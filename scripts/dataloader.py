import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import pandas as pd
import json, os
import torch
from typing import Dict
import random
from torchvision import transforms
to_tensor = transforms.ToTensor()

class VLM_Dataset(Dataset):
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
        
        # Auto-detect split from metadata_image_path if not provided
        metadata_path_lower = metadata_image_path.lower()
        if 'train' in metadata_path_lower:
            self.split = 'train'
        elif 'dev' in metadata_path_lower or 'val' in metadata_path_lower:
            self.split = 'dev'
        elif 'test' in metadata_path_lower:
            self.split = 'test'
        else:
            # Try to match with args metadata paths
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

        if self.use_discharge_note:
            sample["text"] = item['text'][0] if isinstance(item['text'], list) else item['text']
        else:
            sample["text"] = None
        all_img_data_paths = self.hash2meta[item['id']]['metadata_filtered']
        selected_img_data = self.Decision_tree.select_best_cxr(all_img_data_paths)
        selected_img_data_path = selected_img_data[1]

        if self.use_cxr_image:
            sample["image"] = self._load_images(selected_img_data_path)
        else:
            sample["image"] = None

        if self.use_rad_report or self.use_generated_rad_report:
            if selected_img_data_path:
                path_parts = selected_img_data_path.split("/")[:3]
                if len(path_parts) == 3:
                    # Original radiology report path
                    rr_relative_path = '/'.join(path_parts) + ".txt"
                    original_report_path = os.path.join(self.args.base_rr_dir, rr_relative_path)
                    
                    # Generated radiology report path (same dir, with 'generated_' prefix)
                    report_dir = '/'.join(path_parts[:-1])
                    report_filename = path_parts[-1] + ".txt"
                    generated_rr_relative_path = f"{report_dir}/generated_{report_filename}"
                    generated_report_path = os.path.join(self.args.base_rr_dir, generated_rr_relative_path)
                else:
                    original_report_path = None
                    generated_report_path = None
            else:
                original_report_path = None
                generated_report_path = None
            
            # Load original radiology report
            if self.use_rad_report and original_report_path and os.path.exists(original_report_path):
                sample["rad_report"] = self._load_reports([original_report_path])
            else:
                sample["rad_report"] = None
            
            # Load generated radiology report
            if self.use_generated_rad_report and generated_report_path and os.path.exists(generated_report_path):
                sample["generated_rad_report"] = self._load_reports([generated_report_path])
            else:
                sample["generated_rad_report"] = None
        else:
            sample["rad_report"] = None
            sample["generated_rad_report"] = None

        if self.args.use_pi:
            sample["personal_information"] = {"race": self.hash2meta[item['id']]['race'],
                                            "age": self.hash2meta[item['id']]['age']}
        else:
            sample["personal_information"] = {}

        if self.summarize:
            system_prompt, user_prompt = self._get_prompt_summarize(self.args.summary_type, sample["original_note"])
        else:
            # Determine which radiology report to use for prompt
            # If use_generated_rad_report is True, use generated report; otherwise use original
            if self.use_generated_rad_report and sample.get("generated_rad_report"):
                rr_for_prompt = sample["generated_rad_report"]
            else:
                rr_for_prompt = sample["rad_report"]
            
            system_prompt, user_prompt = self._get_prompt(dn=sample["text"], images=sample["image"], rr=rr_for_prompt, pi=sample["personal_information"])

        sample["chat_template"] = self._load_chat_template(system_prompt, user_prompt, sample["image"])

        return sample

    def _load_images(self, mapped_image_path: str) -> list:
        images = []
        
        image_path = mapped_image_path.split("/")[-1]
        name, extension = image_path.split(".")
        if "_512_resized" in name:
            real_image_path = os.path.join(self.args.base_img_dir, self.split, image_path)
        else:
            real_image_path = os.path.join(self.args.base_img_dir, self.split, f"{name}_512_resized.{extension}")
        
        assert os.path.exists(real_image_path), f"Image not found: {real_image_path}"
                
        img = Image.open(real_image_path).convert("RGB")
        tensor_img = to_tensor(img)
        images.append(tensor_img)

        return images

    def _load_reports(self, report_paths: list) -> list:
        reports = []
        for report_path in report_paths:
            with open(report_path, "r", encoding='utf-8') as f:
                report = f.read()
            reports.append(report.replace('\n', ' ').strip())
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
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt}
                ]}
            ]
        return messages
    
    def _get_prompt_summarize(self, prompt_type, dn):

        system_prompt = (
                "You are a highly trained medical assistant specializing in clinical documentation and summarization. "
                "Your task is to produce concise, accurate summaries of discharge notes, relying only on the information explicitly provided in the original document. "
                "Omit any personally identifiable details (e.g., names, ages, races, or ID numbers), and focus strictly on clinically relevant information."
            )

        if prompt_type == "plain":
            user_prompt = (
                f"Here is the clinical discharge document:\n{dn}\n\n"
                "Please generate a summary of overall ICU discharge note that includes important clinical information. "
                "Write the summary as concise, clear, and well-organized bullet points. "
            )
            
        elif prompt_type == "plain_remove_cxr":
            user_prompt = (
                f"Here is the clinical discharge document:\n{dn}\n\n"
                "Please generate a summary of overall ICU discharge note that includes important clinical information while excluding all findings related to chest imaging (including x-ray, CT, or other imaging modalities). "
                "Write the summary as concise, clear, and well-organized bullet points."
            )

        elif prompt_type == "risk_factor":
            user_prompt = (
                f"Here is the clinical discharge document:\n{dn}\n\n"
                "Please generate a summary of potential risk factors from the discharge note, including relevant details with a balanced perspective by summarizing both risk factors and signs of stability or a positive prognosis, outlining:"
                "- Clinical Stability: The patient's current physical stability and any lingering critical conditions. "
                "- Past History: The patient’s medical history and any pre-existing conditions that could influence recovery. "
                "- New Diagnosis: Any new diagnoses made during the patient's stay. "
                "- Follow-Up: The discharge instructions and any recommended outpatient care. "
                "- Adherence: Any notes or concerns regarding the patient’s ability to follow treatment plans. "
                "- Substance Use: Any documented use of drugs, alcohol, or smoking. "
                "- Other Factors: Any additional risks or notable circumstances exAplicitly mentioned in the discharge note. "
                "Please present your summary in concise, clear, and well-organized bullet points."
            )

        elif prompt_type == "risk_factor_remove_cxr":
            user_prompt = (
                f"Here is the clinical discharge document:\n{dn}\n\n"
                "Please generate a summary of potential risk factors from the discharge note, including relevant details with a balanced perspective by summarizing both risk factors and signs of stability or a positive prognosis, while excluding all findings related to chest imaging (including x-ray, CT, or other imaging modalities). outlining:"
                "- Clinical Stability: The patient's current physical stability and any lingering critical conditions. "
                "- Past History: The patient’s medical history and any pre-existing conditions that could influence recovery. "
                "- New Diagnosis: Any new diagnoses made during the patient's stay. "
                "- Follow-Up: The discharge instructions and any recommended outpatient care. "
                "- Adherence: Any notes or concerns regarding the patient’s ability to follow treatment plans. "
                "- Substance Use: Any documented use of drugs, alcohol, or smoking. "
                "- Other Factors: Any additional risks or notable circumstances exAplicitly mentioned in the discharge note. "
                "Please present your summary in concise, clear, and well-organized bullet points."
            )

        elif prompt_type == "timeline":
            user_prompt = (
                f"Here is the clinical discharge document:\n{dn}\n\n"
                "Please generate a timeline summary of the discharge note, outlining: "
                "- Chronological Progression: The chronological progression of the patient's ICU stay from admission to discharge. "
                "- Significant Clinical Changes: Significant changes in the patient's clinical condition over time. "
                "- Timeline of Interventions: The timeline of interventions specific to the ICU setting. "
                "- Notable Complications or Setbacks: Any notable complications or setbacks encountered during the ICU stay. "
                "- Changes in Care Plans: Modifications to the patient's care plan or treatment approach. "
                "Please present your summary in concise, clear, and well-organized bullet points."
            )

        elif prompt_type == "timeline_remove_cxr":
            user_prompt = (
                f"Here is the clinical discharge document:\n{dn}\n\n"
                "Please generate a timeline summary of the discharge note, while excluding all findings related to chest imaging (including x-ray, CT, or other imaging modalities). outlining: "
                "- Chronological Progression: The chronological progression of the patient's ICU stay from admission to discharge. "
                "- Significant Clinical Changes: Significant changes in the patient's clinical condition over time. "
                "- Timeline of Interventions: The timeline of interventions specific to the ICU setting. "
                "- Notable Complications or Setbacks: Any notable complications or setbacks encountered during the ICU stay. "
                "- Changes in Care Plans: Modifications to the patient's care plan or treatment approach. "
                "Please present your summary in concise, clear, and well-organized bullet points."
            )
        else:
            raise KeyError("Check prompt type for summarization!")
            
        return system_prompt, user_prompt

    def _get_prompt(self, dn=None, images=None, rr=None, pi=None):
        if len(pi) > 1:
            personal_information = f"AGE : {pi['age']}, RACE : {pi['race']}"
        else:
            personal_information = ""

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
        # CXR image only
        elif images and not dn and not rr:
            system_prompt = (
                "A single, most recent chest X-ray (CXR) image from the patient is provided. "
                "Based on the provided CXR image, assess how likely the patient's out-of-hospital mortality is within 30 days."
            )
            user_prompt = (
                "Based on the provided single CXR image, how likely is the patient's out-of-hospital mortality within 30 days? "
                "Please do not explain the reason and respond with one word only: 0:alive, 1:death.\n\nAssistant:"
            )
        # Radiology note only
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
        # Discharge note + Radiology note
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
        # Discharge note + CXR image
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
        # CXR image + Radiology note
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
        # Discharge note + CXR image + Radiology note (all three)
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
            raise ValueError("Please provide one of the following combinations:\n discharge note only,\n CXR image only,\n radiology note only,\n discharge note + radiology note,\n discharge note + CXR image,\n CXR image + radiology note,\n discharge note + CXR image + radiology note.")
            
        return system_prompt, user_prompt


def custom_data_collator(processor, use_cxr_image=False, summary_type="plain"):
    def collate_fn(examples: list) -> Dict[str, torch.Tensor]:
        if summary_type == "merged":
            filtered_examples = examples
        else:
            filtered_examples = [ex for ex in examples if ex["summary_type"] == summary_type]

        texts = []
        images_nested = []  
        ids = []
        labels = []

        for example in filtered_examples:
            texts.append(processor.apply_chat_template(example["chat_template"], tokenize=False))
            ids.append(example["id"])
            labels.append(example["label"])

            if use_cxr_image:
                if example.get("image") is not None and isinstance(example["image"], list) and len(example["image"]) > 0:
                    images_nested.append([example["image"][0]]) 
                else:
                    images_nested.append([])  

        if use_cxr_image:
            batch = processor(
                text=texts,
                images=images_nested,
                return_tensors="pt",
                padding=True,
                truncation=False
            )
        else:
            batch = processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=False
            )

        batch["ids"] = ids
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        
        return batch
    
    return collate_fn


def load_hash2meta_dict(mapper_path, image_path):
    # Load note2hash mapping
    with open(mapper_path, 'r', encoding='utf-8') as f_mapper:
        mapper = json.load(f_mapper)
    note2hash = {note_info['note_id']: hash_key for hash_key, note_info in mapper.items()}

    # Load image path parser
    with open(image_path, 'r', encoding='utf-8') as f_image:
        img_path_parser = json.load(f_image)

    hash2meta = {}
    for metadata in img_path_parser['data']:
        note_id = metadata['id']
        hash_key = note2hash.get(note_id, None)

        # Parse in_time and out_time
        try:
            in_time = pd.to_datetime(metadata['debug_features']['INTIME'])
            out_time = pd.to_datetime(metadata['debug_features']['OUTTIME'])
            assert in_time < out_time, f"INTIME {in_time} must be earlier than OUTTIME {out_time}"
        except Exception as e:
            print(f"Error parsing in_time or out_time for note_id {note_id}: {e}")
            continue

        metadata_filtered = []

        # Process images if present
        for img in metadata.get('images', []):
            # Parse image study date and time
            study_date = datetime.strptime(str(img['StudyDate']), "%Y%m%d")
            study_time = datetime.strptime(f"{int(img['StudyTime']):06}", "%H%M%S").time()
            final_datetime = pd.Timestamp(datetime.combine(study_date, study_time))

            # Filter images within in_time and out_time
            if in_time <= final_datetime <= out_time:
                metadata_filtered.append((
                    final_datetime,
                    img['path'],
                    img['PerformedProcedureStepDescription'],
                    img['ViewPosition'],
                    img['PatientOrientationCodeSequence_CodeMeaning']
                ))

        # Remove duplicates and sort data_paths by datetime
        metadata_filtered = sorted(set(metadata_filtered), key=lambda x: x[0])

        # Populate hash2meta dictionary
        if hash_key:
            hash2meta[hash_key] = {
                "note_id": note_id,
                "metadata_filtered": metadata_filtered,
                "race": metadata['debug_features']['RACE'],
                "age": metadata['debug_features']['AGE'],
            }

    return hash2meta

class CXRDecisionTree:
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
        self.data = data
        if not self.data:
            return None 

        sorted_data = sorted(
            self.data,
            key=lambda x: (
                self.get_priority(x[2], "PerformedProcedureStepDescription"),
                self.get_priority(x[3], "ViewPosition"),
                self.get_priority(x[4], "PatientOrientationCodeSequence_CodeMeaning"),
                -x[0].timestamp(),
                x[1]
            )
        )

        return sorted_data[0]
    
    def select_recent_cxr(self, data):
        self.data = data
        if not self.data:
            return None 

        sorted_data = sorted(
            self.data,
            key=lambda x: -x[0].timestamp()
        )

        return sorted_data[0]