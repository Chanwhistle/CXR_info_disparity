"""
CXR 이미지 메타데이터를 임상 노트 JSON에 추가하는 전처리 스크립트.

MIMIC-CXR-JPG의 metadata CSV를 이용해 각 환자의 입원 기간 내 촬영된 CXR 이미지를
LCD Benchmark JSON 포맷에 매핑합니다.

Usage:
    python add_images_to_json.py <input.json> <mimic-cxr-jpg-dir> <output.json>

    <input.json>         : LCD Benchmark 포맷의 입력 JSON (train/dev/test.json)
    <mimic-cxr-jpg-dir>  : MIMIC-CXR-JPG 루트 디렉토리 (files/, mimic-cxr-2.0.0-metadata.csv 포함)
    <output.json>        : 이미지 메타데이터가 추가된 출력 JSON

Source: Modified from LCD Benchmark (https://github.com/Machine-Learning-for-Medical-Language/long-clinical-doc)
"""

import sys
import os
from os.path import join, exists, basename
import json
import glob
import tqdm

import pandas as pd

def main(args):
    if len(args) < 2:
        sys.stderr.write("Required argument(s): <input file> <mimic-cxr-jpg base dir> <output file>\n")
        sys.exit(-1)

    if not exists(args[0]):
        sys.stderr.write("Input file doesn't exist!\n")
        sys.exit(-1)

    if not args[0].endswith(".json"):
        sys.stderr.write("Input file should end with .json!\n")
        sys.exit(-1)

    if not exists(join(args[1], "files")):
        sys.stderr.write("MIMIC CXR directory does not contain a 'files' directory!\n")
        sys.exit(-1)

    meta_df = pd.read_csv(join(args[1], "mimic-cxr-2.0.0-metadata.csv"))

    with open(args[0], 'rt') as fp:
        json_file = json.load(fp)
        missing_images = 0
        num_insts = 0

        for inst in tqdm.tqdm(json_file["data"]):
            num_insts += 1
            inst_id = inst["id"]
            hadm_id = inst["debug_features"]["HADM_ID"]
            # 퇴원일 이전 이미지만 포함 (YYYYMMDD 형식으로 변환)
            dis_dt = inst["debug_features"]["DISCHTIME"].split(" ")[0].replace("-","")
            inst_images_meta = []
            pt_id = inst_id.split("-")[0]
            pt_shortid = pt_id[:2]
            pt_path = join(args[1], "files", "p"+pt_shortid, "p"+pt_id)
            if not exists(pt_path):
                missing_images += 1
                continue

            for study_dir in os.scandir(pt_path):
                if study_dir.is_file() or study_dir.name.startswith('.'):
                    continue
                study_num = int(study_dir.name[1:])
                images = glob.glob(join(study_dir.path, "*.jpg"))
                for image_path in images:
                    image_name = basename(image_path)
                    dicom_id = image_name[:-4]
                    inst_image_meta = meta_df[meta_df['dicom_id']==dicom_id].to_dict('record')[0]
                    study_dt = str(inst_image_meta["StudyDate"])
                    if study_dt <= dis_dt:
                        # 경로는 files/ 기준 상대경로로 저장
                        inst_image_meta['path'] = "/".join(image_path.split("/")[-4:])
                        inst_images_meta.append(inst_image_meta)

            inst['images'] = inst_images_meta

    print("Processed %d instances and %d did not have corresponding images" % (num_insts, missing_images))

    with open(args[2], 'wt') as fp:
        json.dump(json_file, fp, indent=4)

if __name__ == '__main__':
    main(sys.argv[1:])
