"""
멀티모달 VLM 데이터셋 로더 및 CXR 선택 로직.

VLM_Dataset: 텍스트(퇴원 노트/요약), CXR 이미지, 방사선 보고서를 조합하여
  Llama 모델 입력 포맷(chat template)으로 변환합니다.

CXRDecisionTree: 입원 기간 내 CXR 이미지 중 품질 우선순위에 따라 최적 1장 선택
  우선순위: PA/LAT > Portable AP > Single View > Line placement 등
"""

import os
import json
import random
from datetime import datetime
from typing import Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

to_tensor = transforms.ToTensor()


class VLM_Dataset(Dataset):
    """텍스트/이미지/방사선 보고서 조합을 Llama chat template으로 변환하는 데이터셋."""

    def __init__(self, args,
                 data_list,
                 metadata_image_path,
                 use_cxr_image=False,
                 use_rad_report=False,
                 use_generated_rad_report=False,
                 use_discharge_note=False,
                 shuffle=False,
                 summarize=False):
        self.args = args
        self.data_list = data_list
        self.use_cxr_image = use_cxr_image
        self.use_rad_report = use_rad_report
        self.use_generated_rad_report = use_generated_rad_report
        self.use_discharge_note = use_discharge_note
        self.summarize = summarize

        # 경로 이름에서 split(train/dev/test) 자동 탐지
        metadata_path_lower = metadata_image_path.lower()
        if 'train' in metadata_path_lower:
            self.split = 'train'
        elif 'dev' in metadata_path_lower or 'val' in metadata_path_lower:
            self.split = 'dev'
        elif 'test' in metadata_path_lower:
            self.split = 'test'
        else:
            if hasattr(args, 'train_metadata_image_path') and metadata_image_path == args.train_metadata_image_path:
                self.split = 'train'
            elif hasattr(args, 'dev_metadata_image_path') and metadata_image_path == args.dev_metadata_image_path:
                self.split = 'dev'
            elif hasattr(args, 'test_metadata_image_path') and metadata_image_path == args.test_metadata_image_path:
                self.split = 'test'
            else:
                self.split = None

        self.hash2meta = load_hash2meta_dict(args.metadata_path, metadata_image_path)
        self.Decision_tree = CXRDecisionTree()

        if shuffle:
            random.seed(50)
            random.shuffle(self.data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        if not self.args.summarize:
            sample = {
                "id": item['id'],
                "label": item['label'],
                "summary_type": item['summary_type']
            }
        else:
            sample = {
                "id": item['id'],
                "label": item['label'],
                "original_note": item['text'][0] if isinstance(item['text'], list) else item['text'],
                "summary_type": item['summary_type']
            }

        sample["text"] = (item['text'][0] if isinstance(item['text'], list) else item['text']) if self.use_discharge_note else None

        # CXR 이미지 선택 (CXRDecisionTree 우선순위 기준)
        all_img_data_paths = self.hash2meta[item['id']]['metadata_filtered']
        selected_img_data = self.Decision_tree.select_best_cxr(all_img_data_paths)
        selected_img_data_path = selected_img_data[1]

        sample["image"] = self._load_images(selected_img_data_path) if self.use_cxr_image else None

        # 방사선 보고서 경로 구성 (p10/pXXXXX/sYYYYY 구조)
        if self.use_rad_report or self.use_generated_rad_report:
            original_report_path, generated_report_path = self._get_report_paths(selected_img_data_path)
            sample["rad_report"] = (
                self._load_reports([original_report_path])
                if self.use_rad_report and original_report_path and os.path.exists(original_report_path)
                else None
            )
            sample["generated_rad_report"] = (
                self._load_reports([generated_report_path])
                if self.use_generated_rad_report and generated_report_path and os.path.exists(generated_report_path)
                else None
            )
        else:
            sample["rad_report"] = None
            sample["generated_rad_report"] = None

        if self.args.use_pi:
            sample["personal_information"] = {
                "race": self.hash2meta[item['id']]['race'],
                "age": self.hash2meta[item['id']]['age']
            }
        else:
            sample["personal_information"] = {}

        if self.summarize:
            system_prompt, user_prompt = self._get_prompt_summarize(self.args.summary_type, sample["original_note"])
        else:
            rr_for_prompt = (
                sample.get("generated_rad_report") if self.use_generated_rad_report and sample.get("generated_rad_report")
                else sample["rad_report"]
            )
            system_prompt, user_prompt = self._get_prompt(
                dn=sample["text"], images=sample["image"], rr=rr_for_prompt, pi=sample["personal_information"]
            )

        sample["chat_template"] = self._load_chat_template(system_prompt, user_prompt, sample["image"])
        return sample

    def _get_report_paths(self, img_data_path):
        """이미지 경로에서 방사선 보고서 경로 추출 (original + generated)."""
        if not img_data_path:
            return None, None
        path_parts = img_data_path.split("/")[:3]
        if len(path_parts) != 3:
            return None, None
        rr_relative = '/'.join(path_parts) + ".txt"
        gen_relative = '/'.join(path_parts[:-1]) + "/generated_" + path_parts[-1] + ".txt"
        return (
            os.path.join(self.args.base_rr_dir, rr_relative),
            os.path.join(self.args.base_rr_dir, gen_relative)
        )

    def _load_images(self, mapped_image_path: str) -> list:
        image_path = mapped_image_path.split("/")[-1]
        name, extension = image_path.split(".")
        if "_512_resized" in name:
            real_image_path = os.path.join(self.args.base_img_dir, self.split, image_path)
        else:
            real_image_path = os.path.join(self.args.base_img_dir, self.split, f"{name}_512_resized.{extension}")

        assert os.path.exists(real_image_path), f"Image not found: {real_image_path}"
        img = Image.open(real_image_path).convert("RGB")
        return [to_tensor(img)]

    def _load_reports(self, report_paths: list) -> list:
        reports = []
        for path in report_paths:
            with open(path, "r", encoding='utf-8') as f:
                reports.append(f.read().replace('\n', ' ').strip())
        return reports

    def _load_chat_template(self, system_prompt, user_prompt, images=None):
        if images:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    *[{"type": "image"} for _ in images],
                    {"type": "text", "text": user_prompt}
                ]}
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
            ]
        return messages

    def _get_prompt_summarize(self, prompt_type, dn):
        system_prompt = (
            "You are a highly trained medical assistant specializing in clinical documentation and summarization. "
            "Your task is to produce concise, accurate summaries of discharge notes, relying only on the information explicitly provided in the original document. "
            "Omit any personally identifiable details (e.g., names, ages, races, or ID numbers), and focus strictly on clinically relevant information."
        )

        prompts = {
            "plain": (
                f"Here is the clinical discharge document:\n{dn}\n\n"
                "Please generate a summary of overall ICU discharge note that includes important clinical information. "
                "Write the summary as concise, clear, and well-organized bullet points. "
            ),
            "plain_remove_cxr": (
                f"Here is the clinical discharge document:\n{dn}\n\n"
                "Please generate a summary of overall ICU discharge note that includes important clinical information while excluding all findings related to chest imaging (including x-ray, CT, or other imaging modalities). "
                "Write the summary as concise, clear, and well-organized bullet points."
            ),
            "risk_factor": (
                f"Here is the clinical discharge document:\n{dn}\n\n"
                "Please generate a summary of potential risk factors from the discharge note, including relevant details with a balanced perspective by summarizing both risk factors and signs of stability or a positive prognosis, outlining:"
                "- Clinical Stability: The patient's current physical stability and any lingering critical conditions. "
                "- Past History: The patient's medical history and any pre-existing conditions that could influence recovery. "
                "- New Diagnosis: Any new diagnoses made during the patient's stay. "
                "- Follow-Up: The discharge instructions and any recommended outpatient care. "
                "- Adherence: Any notes or concerns regarding the patient's ability to follow treatment plans. "
                "- Substance Use: Any documented use of drugs, alcohol, or smoking. "
                "- Other Factors: Any additional risks or notable circumstances explicitly mentioned in the discharge note. "
                "Please present your summary in concise, clear, and well-organized bullet points."
            ),
            "risk_factor_remove_cxr": (
                f"Here is the clinical discharge document:\n{dn}\n\n"
                "Please generate a summary of potential risk factors from the discharge note, including relevant details with a balanced perspective by summarizing both risk factors and signs of stability or a positive prognosis, while excluding all findings related to chest imaging (including x-ray, CT, or other imaging modalities). outlining:"
                "- Clinical Stability: The patient's current physical stability and any lingering critical conditions. "
                "- Past History: The patient's medical history and any pre-existing conditions that could influence recovery. "
                "- New Diagnosis: Any new diagnoses made during the patient's stay. "
                "- Follow-Up: The discharge instructions and any recommended outpatient care. "
                "- Adherence: Any notes or concerns regarding the patient's ability to follow treatment plans. "
                "- Substance Use: Any documented use of drugs, alcohol, or smoking. "
                "- Other Factors: Any additional risks or notable circumstances explicitly mentioned in the discharge note. "
                "Please present your summary in concise, clear, and well-organized bullet points."
            ),
            "timeline": (
                f"Here is the clinical discharge document:\n{dn}\n\n"
                "Please generate a timeline summary of the discharge note, outlining: "
                "- Chronological Progression: The chronological progression of the patient's ICU stay from admission to discharge. "
                "- Significant Clinical Changes: Significant changes in the patient's clinical condition over time. "
                "- Timeline of Interventions: The timeline of interventions specific to the ICU setting. "
                "- Notable Complications or Setbacks: Any notable complications or setbacks encountered during the ICU stay. "
                "- Changes in Care Plans: Modifications to the patient's care plan or treatment approach. "
                "Please present your summary in concise, clear, and well-organized bullet points."
            ),
            "timeline_remove_cxr": (
                f"Here is the clinical discharge document:\n{dn}\n\n"
                "Please generate a timeline summary of the discharge note, while excluding all findings related to chest imaging (including x-ray, CT, or other imaging modalities). outlining: "
                "- Chronological Progression: The chronological progression of the patient's ICU stay from admission to discharge. "
                "- Significant Clinical Changes: Significant changes in the patient's clinical condition over time. "
                "- Timeline of Interventions: The timeline of interventions specific to the ICU setting. "
                "- Notable Complications or Setbacks: Any notable complications or setbacks encountered during the ICU stay. "
                "- Changes in Care Plans: Modifications to the patient's care plan or treatment approach. "
                "Please present your summary in concise, clear, and well-organized bullet points."
            ),
        }

        if prompt_type not in prompts:
            raise KeyError(f"Unknown prompt_type: {prompt_type}. Valid: {list(prompts.keys())}")

        return system_prompt, prompts[prompt_type]

    def _get_prompt(self, dn=None, images=None, rr=None, pi=None):
        """모달리티 조합에 따른 시스템/유저 프롬프트 생성."""
        personal_information = f"AGE : {pi['age']}, RACE : {pi['race']}" if len(pi) > 1 else ""

        if dn and not images and not rr:
            system_prompt = (
                "Below is a clinical discharge document. "
                "Based on the given clinical context, assess how likely the patient's out-of-hospital mortality is within 30 days."
            )
            user_prompt = (
                f"Here is the clinical document:\n{personal_information} {dn}\n\n"
                "Based on the clinical information provided, how likely is the patient's out-of-hospital mortality within 30 days? "
                "Please do not explain the reason and respond with one word only: 0:alive, 1:death.\n\nAssistant:"
            )
        elif images and not dn and not rr:
            system_prompt = (
                "A single, most recent chest X-ray (CXR) image from the patient is provided. "
                "Based on the provided CXR image, assess how likely the patient's out-of-hospital mortality is within 30 days."
            )
            user_prompt = (
                "Based on the provided single CXR image, how likely is the patient's out-of-hospital mortality within 30 days? "
                "Please do not explain the reason and respond with one word only: 0:alive, 1:death.\n\nAssistant:"
            )
        elif rr and not dn and not images:
            system_prompt = (
                "A most recent radiology report from the patient is provided. "
                "Based on the provided radiology report, assess how likely the patient's out-of-hospital mortality is within 30 days."
            )
            user_prompt = (
                f"Here is the radiology report:\n{rr}\n\n"
                "Based on the radiology report provided, how likely is the patient's out-of-hospital mortality within 30 days? "
                "Please do not explain the reason and respond with one word only: 0:alive, 1:death.\n\nAssistant:"
            )
        elif dn and rr and not images:
            system_prompt = (
                "A clinical document and a most recent radiology report are provided. "
                "Based on the clinical context and the radiology report, assess how likely the patient's out-of-hospital mortality is within 30 days."
            )
            user_prompt = (
                f"Here is the clinical document:\n{personal_information} {dn}\n\n"
                f"Here is the radiology report:\n{rr}\n\n"
                "Based on the provided clinical information and radiology report, how likely is the patient's out-of-hospital mortality within 30 days? "
                "Please do not explain the reason and respond with one word only: 0:alive, 1:death.\n\nAssistant:"
            )
        elif dn and images and not rr:
            system_prompt = (
                "A clinical document and a single, most recent chest X-ray (CXR) image from the patient are provided. "
                "Based on the clinical context and the provided CXR image, assess how likely the patient's out-of-hospital mortality is within 30 days."
            )
            user_prompt = (
                f"Here is the clinical document:\n{personal_information} {dn}\n\n"
                "Based on the provided clinical information and single CXR image, how likely is the patient's out-of-hospital mortality within 30 days? "
                "Please do not explain the reason and respond with one word only: 0:alive, 1:death.\n\nAssistant:"
            )
        elif images and rr and not dn:
            system_prompt = (
                "A single, most recent chest X-ray (CXR) image from the patient and a radiology report are provided. "
                "Based on the radiology report and the provided CXR image, assess how likely the patient's out-of-hospital mortality is within 30 days."
            )
            user_prompt = (
                f"Here is the radiology report:\n{rr}\n\n"
                "Based on the provided CXR image and radiology report, how likely is the patient's out-of-hospital mortality within 30 days? "
                "Please do not explain the reason and respond with one word only: 0:alive, 1:death.\n\nAssistant:"
            )
        elif dn and images and rr:
            system_prompt = (
                "A clinical document, a single most recent chest X-ray (CXR) image, and a radiology report are provided. "
                "Based on the clinical context, the CXR image, and the radiology report, assess how likely the patient's out-of-hospital mortality is within 30 days."
            )
            user_prompt = (
                f"Here is the clinical document:\n{personal_information} {dn}\n\n"
                f"Here is the radiology report:\n{rr}\n\n"
                "Based on the provided clinical information, CXR image, and radiology report, how likely is the patient's out-of-hospital mortality within 30 days? "
                "Please do not explain the reason and respond with one word only: 0:alive, 1:death.\n\nAssistant:"
            )
        else:
            raise ValueError(
                "Please provide one of the following combinations: "
                "discharge note only, CXR image only, radiology note only, "
                "discharge note + radiology note, discharge note + CXR image, "
                "CXR image + radiology note, discharge note + CXR image + radiology note."
            )

        return system_prompt, user_prompt


def custom_data_collator(processor, use_cxr_image=False, summary_type="plain"):
    """프로세서/토크나이저를 사용한 배치 콜레이터 반환."""
    def collate_fn(examples: list) -> Dict[str, torch.Tensor]:
        if summary_type == "merged":
            filtered_examples = examples
        else:
            filtered_examples = [ex for ex in examples if ex["summary_type"] == summary_type]

        texts, images_nested, ids, labels = [], [], [], []

        for example in filtered_examples:
            texts.append(processor.apply_chat_template(example["chat_template"], tokenize=False))
            ids.append(example["id"])
            labels.append(example["label"])

            if use_cxr_image:
                img = example.get("image")
                images_nested.append([img[0]] if img and isinstance(img, list) and len(img) > 0 else [])

        # 이미지 여부에 따라 프로세서 호출 방식 분기
        if use_cxr_image:
            batch = processor(
                text=texts, images=images_nested,
                return_tensors="pt", padding=True, truncation=False
            )
        else:
            # 텍스트 전용: AutoTokenizer 호환 호출
            batch = processor(
                texts,
                return_tensors="pt", padding=True, truncation=False
            )

        batch["ids"] = ids
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch

    return collate_fn


def load_hash2meta_dict(mapper_path, image_path):
    """metadata.json과 image JSON을 조합하여 hash → (note_id, 이미지 목록, race, age) 매핑 반환."""
    with open(mapper_path, 'r', encoding='utf-8') as f:
        mapper = json.load(f)
    note2hash = {note_info['note_id']: hash_key for hash_key, note_info in mapper.items()}

    with open(image_path, 'r', encoding='utf-8') as f:
        img_path_parser = json.load(f)

    hash2meta = {}
    for metadata in img_path_parser['data']:
        note_id = metadata['id']
        hash_key = note2hash.get(note_id, None)

        try:
            in_time = pd.to_datetime(metadata['debug_features']['INTIME'])
            out_time = pd.to_datetime(metadata['debug_features']['OUTTIME'])
            assert in_time < out_time
        except Exception as e:
            print(f"Error parsing in_time/out_time for note_id {note_id}: {e}")
            continue

        metadata_filtered = []
        for img in metadata.get('images', []):
            study_date = datetime.strptime(str(img['StudyDate']), "%Y%m%d")
            study_time = datetime.strptime(f"{int(img['StudyTime']):06}", "%H%M%S").time()
            final_datetime = pd.Timestamp(datetime.combine(study_date, study_time))

            if in_time <= final_datetime <= out_time:
                metadata_filtered.append((
                    final_datetime,
                    img['path'],
                    img['PerformedProcedureStepDescription'],
                    img['ViewPosition'],
                    img['PatientOrientationCodeSequence_CodeMeaning']
                ))

        metadata_filtered = sorted(set(metadata_filtered), key=lambda x: x[0])

        if hash_key:
            hash2meta[hash_key] = {
                "note_id": note_id,
                "metadata_filtered": metadata_filtered,
                "race": metadata['debug_features']['RACE'],
                "age": metadata['debug_features']['AGE'],
            }

    return hash2meta


class CXRDecisionTree:
    """
    입원 기간 내 CXR 이미지 중 품질 우선순위 기준으로 최적 1장 선택.

    우선순위: PA/LAT(1) > Portable AP(3) > Single View(4) > Line placement(6) > 기타
    동점 시: 가장 최근 이미지 선택 (-timestamp 기준 정렬)
    """

    def __init__(self):
        self.priority_mapping = {
            "PerformedProcedureStepDescription": {
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
                "OTHER": 99
            },
            "ViewPosition": {
                "PA": 1,
                "AP": 2,
                "LATERAL": 3,
                "LL": 3,
                "OTHER": 99
            },
            "PatientOrientationCodeSequence_CodeMeaning": {
                "Erect": 1,
                "Recumbent": 2,
                "OTHER": 99
            }
        }

    def get_priority(self, value, category):
        return self.priority_mapping.get(category, {}).get(value, 99)

    def select_best_cxr(self, data):
        """우선순위 + 최신 순으로 정렬하여 최적 CXR 반환. 데이터 없으면 None."""
        if not data:
            return None
        return sorted(
            data,
            key=lambda x: (
                self.get_priority(x[2], "PerformedProcedureStepDescription"),
                self.get_priority(x[3], "ViewPosition"),
                self.get_priority(x[4], "PatientOrientationCodeSequence_CodeMeaning"),
                -x[0].timestamp(),
                x[1]
            )
        )[0]

    def select_recent_cxr(self, data):
        """가장 최근 CXR 반환."""
        if not data:
            return None
        return sorted(data, key=lambda x: -x[0].timestamp())[0]
