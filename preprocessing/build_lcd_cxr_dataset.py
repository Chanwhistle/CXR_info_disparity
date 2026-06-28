#!/usr/bin/env python
"""
Build the JSONL and image metadata files used by the downstream pipeline.
"""

import argparse
import json
import math
import netrc
import os
import shutil
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


TARGET_SIZE = 560
PHYSIONET_HOST = "physionet.org"

PROCEDURE_PRIORITY = {
    "CHEST (PA AND LAT)": 1,
    "CHEST (PRE-OP PA AND LAT)": 1,
    "CHEST (PA AND LAT) PORT": 2,
    "CHEST (PORTABLE AP)": 3,
    "CHEST (SINGLE VIEW)": 4,
    "CHEST (SINGLE VIEW) PORT": 5,
    "DX CHEST PORTABLE PICC LINE PLACEMENT": 6,
    "DX CHEST PORT LINE/TUBE PLCMT 3 EXAMS": 6,
    "CHEST PORT. LINE PLACEMENT": 6,
    "TRAUMA #2 (AP CXR AND PELVIS PORT)": 7,
    "TRAUMA #3 (PORT CHEST ONLY)": 7,
    "ABD PORT LINE/TUBE PLACEMENT 1 EXAM": 8,
    "ABD PORT LINE/TUBE PLACEMENT 1 EXAM PORT": 8,
    "PORTABLE ABDOMEN": 9,
}

VIEW_PRIORITY = {
    "PA": 1,
    "AP": 2,
    "LATERAL": 3,
    "LL": 3,
}

ORIENTATION_PRIORITY = {
    "Erect": 1,
    "Recumbent": 2,
}


def clean_value(value):
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def clean_record(record):
    return {key: clean_value(value) for key, value in record.items()}


def date_to_yyyymmdd(value):
    if value is None or pd.isna(value):
        return None
    return str(pd.to_datetime(value).date()).replace("-", "")


def image_timestamp(image):
    try:
        study_date = datetime.strptime(str(int(image["StudyDate"])), "%Y%m%d")
        study_time = datetime.strptime(f"{int(float(image.get('StudyTime') or 0)):06}", "%H%M%S").time()
        return datetime.combine(study_date, study_time)
    except Exception:
        return datetime.min


def select_best_image(images, admit_time, discharge_time):
    candidates = []
    for image in images:
        ts = image_timestamp(image)
        if admit_time <= ts <= discharge_time:
            proc = image.get("PerformedProcedureStepDescription") or "OTHER"
            view = image.get("ViewPosition") or "OTHER"
            orient = image.get("PatientOrientationCodeSequence_CodeMeaning") or "OTHER"
            candidates.append((
                PROCEDURE_PRIORITY.get(proc, 99),
                VIEW_PRIORITY.get(view, 99),
                ORIENTATION_PRIORITY.get(orient, 99),
                -ts.timestamp(),
                image,
            ))
    if not candidates:
        return None
    return sorted(candidates)[0][-1]


def resize_with_padding(src_path, dst_path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        return
    with Image.open(src_path) as img:
        img = img.convert("RGB")
        width, height = img.size
        scale = min(TARGET_SIZE / width, TARGET_SIZE / height)
        new_size = (int(width * scale), int(height * scale))
        resized = img.resize(new_size, Image.LANCZOS)

        final_img = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), (0, 0, 0))
        x_offset = (TARGET_SIZE - new_size[0]) // 2
        y_offset = (TARGET_SIZE - new_size[1]) // 2
        final_img.paste(resized, (x_offset, y_offset))
        final_img.save(dst_path, "JPEG", quality=95)


def get_physionet_auth():
    if os.environ.get("PHYSIONET_USER") and os.environ.get("PHYSIONET_PASS"):
        return os.environ["PHYSIONET_USER"], os.environ["PHYSIONET_PASS"]
    try:
        auth = netrc.netrc().authenticators(PHYSIONET_HOST)
        if auth:
            return auth[0], auth[2]
    except (FileNotFoundError, netrc.NetrcParseError):
        pass
    return None, None


def download_url(url, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return

    username, password = get_physionet_auth()
    request = urllib.request.Request(url)
    if username and password:
        password_manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        password_manager.add_password(None, url, username, password)
        opener = urllib.request.build_opener(urllib.request.HTTPBasicAuthHandler(password_manager))
        with opener.open(request) as response, open(output_path, "wb") as out_fp:
            shutil.copyfileobj(response, out_fp)
    else:
        with urllib.request.urlopen(request) as response, open(output_path, "wb") as out_fp:
            shutil.copyfileobj(response, out_fp)


def copy_selected_image(
    image,
    split,
    mimic_cxr_jpg_dir,
    output_image_dir,
    download_missing_assets=False,
    cxr_jpg_base_url=None,
):
    rel_path = Path(image["path"])
    name = rel_path.name
    stem = name[:-4] if name.endswith(".jpg") else rel_path.stem
    src_resized = Path(mimic_cxr_jpg_dir) / "files" / rel_path.with_name(f"{stem}_{TARGET_SIZE}_resized.jpg")
    src_original = Path(mimic_cxr_jpg_dir) / "files" / rel_path
    dst = Path(output_image_dir) / split / f"{stem}_{TARGET_SIZE}_resized.jpg"

    if src_resized.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_resized, dst)
    elif src_original.exists():
        resize_with_padding(src_original, dst)
    elif download_missing_assets and cxr_jpg_base_url:
        url = f"{cxr_jpg_base_url.rstrip('/')}/files/{rel_path.as_posix()}"
        download_url(url, src_original)
        resize_with_padding(src_original, dst)
    else:
        raise FileNotFoundError(src_original)


def download_selected_report(image, report_dir, cxr_report_base_url):
    if not report_dir or not cxr_report_base_url:
        return
    rel_parts = Path(image["path"]).parts[:3]
    if len(rel_parts) != 3:
        return
    rel_path = Path(*rel_parts).with_suffix(".txt")
    out_path = Path(report_dir) / rel_path
    url = f"{cxr_report_base_url.rstrip('/')}/files/{rel_path.as_posix()}"
    download_url(url, out_path)


def load_json(path):
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(obj, fp, indent=2, allow_nan=False)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def csv_or_gz(path):
    path = Path(path)
    if path.exists():
        return path
    gz_path = Path(f"{path}.gz")
    if gz_path.exists():
        return gz_path
    raise FileNotFoundError(f"Missing file: {path} or {gz_path}")


def build_debug_features(meta, admission, patient):
    admit_time = str(pd.to_datetime(admission["admittime"]))
    discharge_time = str(pd.to_datetime(admission["dischtime"]))
    debug = {
        "SUBJECT_ID": int(meta["subject_id"]),
        "HADM_ID": int(meta["hadm_id"]),
        "ADMITTIME": admit_time,
        "DISCHTIME": discharge_time,
        "INTIME": admit_time,
        "OUTTIME": discharge_time,
        "RACE": clean_value(admission.get("race")),
        "HOSPITAL_EXPIRE_FLAG": clean_value(admission.get("hospital_expire_flag")),
        "GENDER": clean_value(patient.get("gender")) if patient is not None else None,
        "ANCHOR_AGE": clean_value(patient.get("anchor_age")) if patient is not None else None,
        "ANCHOR_YEAR": clean_value(patient.get("anchor_year")) if patient is not None else None,
        "ANCHOR_YEAR_GROUP": clean_value(patient.get("anchor_year_group")) if patient is not None else None,
        "DOD": clean_value(patient.get("dod")) if patient is not None else None,
        "AGE": clean_value(patient.get("anchor_age")) if patient is not None else None,
        "note_ids": str([meta["note_id"]]),
        "labels": None,
    }
    return debug


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lcd_dir", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--mimic_iv_dir", required=True)
    parser.add_argument("--mimic_cxr_jpg_dir", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--image_output_dir", required=True)
    parser.add_argument("--task_name", default="out_hospital_mortality_30")
    parser.add_argument("--download_missing_assets", action="store_true")
    parser.add_argument("--cxr_jpg_base_url", default="")
    parser.add_argument("--cxr_report_base_url", default="")
    parser.add_argument("--mimic_cxr_report_dir", default="")
    args = parser.parse_args()

    lcd_metadata = load_json(args.metadata_path)
    admissions = pd.read_csv(csv_or_gz(Path(args.mimic_iv_dir) / "hosp" / "admissions.csv"))
    patients_path = csv_or_gz(Path(args.mimic_iv_dir) / "hosp" / "patients.csv")
    patients = pd.read_csv(patients_path) if patients_path.exists() else pd.DataFrame()
    cxr_meta = pd.read_csv(Path(args.mimic_cxr_jpg_dir) / "mimic-cxr-2.0.0-metadata.csv")

    admissions_by_hadm = admissions.set_index("hadm_id", drop=False)
    patients_by_subject = patients.set_index("subject_id", drop=False) if not patients.empty else None
    cxr_by_subject = {sid: group for sid, group in cxr_meta.groupby("subject_id")}

    dataset_metadata = {}

    for split in ["train", "dev", "test"]:
        split_json = load_json(Path(args.lcd_dir) / f"{split}.json")
        jsonl_rows = []
        image_json_rows = []
        missing_admission = missing_images = copied_images = 0

        for inst in tqdm(split_json["data"], desc=f"Building {split}"):
            hash_id = inst["id"]
            meta = lcd_metadata.get(hash_id)
            if not meta or pd.isna(meta.get("hadm_id")):
                missing_admission += 1
                continue

            hadm_id = int(meta["hadm_id"])
            subject_id = int(meta["subject_id"])
            if hadm_id not in admissions_by_hadm.index:
                missing_admission += 1
                continue

            admission = admissions_by_hadm.loc[hadm_id]
            patient = None
            if patients_by_subject is not None and subject_id in patients_by_subject.index:
                patient = patients_by_subject.loc[subject_id]

            debug_features = build_debug_features(meta, admission, patient)
            admit_time = pd.to_datetime(debug_features["ADMITTIME"])
            discharge_time = pd.to_datetime(debug_features["DISCHTIME"])
            discharge_date = date_to_yyyymmdd(debug_features["DISCHTIME"])

            images = []
            for _, row in cxr_by_subject.get(subject_id, pd.DataFrame()).iterrows():
                image = clean_record(row.to_dict())
                if str(image.get("StudyDate")) <= discharge_date:
                    image["path"] = (
                        f"p{str(subject_id)[:2]}/p{subject_id}/"
                        f"s{int(image['study_id'])}/{image['dicom_id']}.jpg"
                    )
                    images.append(image)

            selected = select_best_image(images, admit_time, discharge_time)
            if selected is None:
                missing_images += 1
            else:
                try:
                    copy_selected_image(
                        selected,
                        split,
                        args.mimic_cxr_jpg_dir,
                        args.image_output_dir,
                        args.download_missing_assets,
                        args.cxr_jpg_base_url,
                    )
                    if args.download_missing_assets:
                        try:
                            download_selected_report(
                                selected,
                                args.mimic_cxr_report_dir,
                                args.cxr_report_base_url,
                            )
                        except Exception as exc:
                            print(f"Warning: failed to download report for {selected.get('path')}: {exc}")
                    copied_images += 1
                except FileNotFoundError:
                    missing_images += 1

            label = int(inst[args.task_name])
            debug_features["labels"] = label

            jsonl_rows.append({
                "id": hash_id,
                "label": label,
                "original_note": inst["text"],
            })
            image_json_rows.append({
                "id": meta["note_id"],
                "text": inst["text"],
                args.task_name: label,
                "debug_features": debug_features,
                "images": images,
            })
            dataset_metadata[hash_id] = meta

        split_dir = Path(args.dataset_dir) / f"{split}_summarization"
        write_jsonl(split_dir / f"{split}.jsonl", jsonl_rows)
        write_json(split_dir / f"full-{split}-indent-images.json", {"data": image_json_rows})

        print(
            f"{split}: rows={len(jsonl_rows)}, copied_images={copied_images}, "
            f"missing_images={missing_images}, missing_admission={missing_admission}"
        )

    write_json(Path(args.dataset_dir) / "metadata.json", dataset_metadata)


if __name__ == "__main__":
    main()
