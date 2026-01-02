from llm_experiment_examples.utils import *
import json, os
import torch
import argparse
import logging
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from tqdm import tqdm
import re
import medspacy
from typing import Dict, Set, Union
from sklearn.metrics import *
from huggingface_hub import login
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification, pipeline
from typing import List, Set, Dict, Tuple
from f1chexbert import F1CheXbert

login("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")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
print(f"Log will show from Level: {logger.getEffectiveLevel()}")

class RadiologyEvaluator:
    def __init__(self):
        # CXR 특화 모델 설정
        self.model_name = "microsoft/BiomedVLP-CXR-BERT-specialized"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.f1chexbert = F1CheXbert()
        self.nlp = medspacy.load()
        
        # 임베딩용 모델 (의미적 유사도)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        self.model.to(self.device)
        
        # NER 모델 설정
        self.ner_model_name = "samrawal/bert-base-uncased_clinical-ner"
        # {0: 'B-problem', 1: 'B-treatment', 2: 'I-test', 3: 'I-treatment', 4: 'B-test', 5: 'O', 6: 'I-problem'}
        self.ner_tokenizer = AutoTokenizer.from_pretrained(self.ner_model_name)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(self.ner_model_name)
        self.ner_model.to(self.device)
        
        self.ner_pipeline = pipeline(
            "ner",
            model=self.ner_model,
            tokenizer=self.ner_tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            aggregation_strategy="simple"
        )

    def get_bleu(self, reference, hypothesis):
        reference_tokens = [reference.split()]
        hypothesis_tokens = hypothesis.split()
        smoothing_function = SmoothingFunction().method1
        score = sentence_bleu(reference_tokens, hypothesis_tokens, smoothing_function=smoothing_function)
        return score
        
    def get_rouge(self, reference, hypothesis):
        rouge = Rouge()
        scores = rouge.get_scores(reference, hypothesis, avg=True)
        return scores

    def get_jaccard_similarity(self, gold_report: str, generated_report: str) -> float:
        """자카드 유사도 계산"""
        set1 = set([word.strip() for word in gold_report.split()])
        set2 = set([word.strip() for word in generated_report.split()])
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union != 0 else 0
    
    def get_semantic_similarity(self, gold_report: str, generated_report: str) -> float:
        """두 리포트 간의 의미적 유사도(코사인 유사도) 계산"""
        # 토크나이징
        encoded = self.tokenizer(
            [gold_report, generated_report],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # 임베딩 계산
        with torch.no_grad():
            outputs = self.model(**encoded)
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        # 코사인 유사도 계산
        similarity = torch.nn.functional.cosine_similarity(
            embeddings[0].unsqueeze(0),
            embeddings[1].unsqueeze(0)
        )
        
        return similarity.item()

    def extract_clinical_entities(self, text: str) -> Set[str]:
        """텍스트에서 임상 엔티티 추출"""
        entities = []
        ner_results = self.ner_pipeline(text)
        
        for result in ner_results:
            if result['entity_group'] in ['problem', 'treatment', 'test']:
                entity = f"{result['word'].lower().strip()}"
                entities.append(entity)
        
        return entities

    def calculate_f1_score(self, gold_entities: Set[str], 
                          generated_entities: Set[str]) -> Dict[str, float]:
        """NER 결과의 F1 score 계산"""
        tp = len(gold_entities.intersection(generated_entities))
        fp = len(generated_entities - gold_entities)
        fn = len(gold_entities - generated_entities)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
              
    def evaluate_reports(self, gold_report: str, 
                        generated_report: str) -> Dict[str, Union[float, Dict]]:
        clean_text = lambda x: re.sub("\s+", " ", re.sub("\[.*?\]", "", x).replace("**", "")).strip().lower()
        gold_report = clean_text(gold_report)
        generated_report = clean_text(generated_report)

        bleu = self.get_bleu(gold_report, generated_report)
        rouge = self.get_rouge(gold_report, generated_report)
        similarity = self.get_semantic_similarity(gold_report, generated_report)
        jaccard_scores = self.get_jaccard_similarity(gold_report, generated_report)
        gold_sentences = [sent.text.strip() for sent in self.nlp(gold_report).sents]
        pred_sentences = [sent.text.strip() for sent in self.nlp(generated_report).sents]
        accuracy, accuracy_not_averaged, class_report, class_report_5 = self.f1chexbert(hyps=[gold_report], 
                                                                                        refs=[generated_report])
        
        # 임상 엔티티 추출
        gold_entities = set([ent for gold_sentence in gold_sentences for ent in self.extract_clinical_entities(gold_sentence)])
        generated_entities = set([ent for pred_sentence in pred_sentences for ent in self.extract_clinical_entities(pred_sentence)])
        ner_scores = self.calculate_f1_score(gold_entities, generated_entities)
        
        return {
            'bleu':bleu,
            'rouge':rouge,
            'semantic_similarity': similarity,
            'ner_scores': ner_scores,
            'jaccard_scores': jaccard_scores,
            'f1chexbert': {
                'accuracy':accuracy,
                'accuracy_not_averaged':accuracy_not_averaged.item(),
                # 'class_report':class_report,
                # 'class_report_5':class_report_5
            },
            'entities': {
                'gold': gold_entities,
                'generated': generated_entities
            }
        }

def calculate_averages(args, output):
    total_bleu, total_rouge, total_sim, total_jaccard, total_chexbert, total_ner = 0, 0, 0, 0, 0, 0
    count = 0
    
    for sample in output:
        total_bleu += sample['score']['bleu']
        total_rouge += sample['score']['rouge']['rouge-l']['f']
        total_sim += sample['score']['similarity']
        total_jaccard += sample['score']['jaccard']
        total_chexbert += sample['score']['f1chexbert']['accuracy']
        total_ner += sample['score']['ner']['f1']
        count += 1

    return {
        "model" : args.model_name_or_path,
        "average_bleu": total_bleu / count,
        "average_rouge": total_rouge / count,
        "average_similarity": total_sim / count,      
        "average_jaccard": total_jaccard / count,  
        "average_chexbert": total_chexbert / count,  
        "average_ner": total_ner / count,     
        "total_num": count,
        "output": output
    }


def extract_part(radiology_report):
    re_header = r'(?=\n[A-Z ]+:\s|$)'

    # INDICATION
    ind_pattern = r'(?s)INDICATION:\s*(.*?)' + re_header
    ind_match = re.search(ind_pattern, radiology_report)
    indication = ind_match.group(1).strip() if ind_match else None

    # REASON FOR EXAM (INATION) => 대략 이런 식으로...
    reason_pattern = r'(?s)REASON FOR EXAM(?:INATION)?:\s*(.*?)' + re_header
    reason_match = re.search(reason_pattern, radiology_report)
    reason = reason_match.group(1).strip() if reason_match else None

    # IMPRESSION
    impression_pattern = r'(?s)IMPRESSION:\s*(.*?)' + re_header
    impression_match = re.search(impression_pattern, radiology_report)
    impression = impression_match.group(1).strip() if impression_match else None

    # FINDINGS
    findings_pattern = r'(?s)FINDINGS:\s*(.*?)' + re_header
    findings_match = re.search(findings_pattern, radiology_report)
    findings = findings_match.group(1).strip() if findings_match else None

    return indication, reason, impression, findings



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM zero-shot inference."
    )   
    parser.add_argument(
        "--model_name_or_path", 
        required=True, 
        help="Hugging Face Hub or local path"
    )
    parser.add_argument(
        '--metadata_path', 
        type=str, 
        default="./processed_data/metadata.json",
        help="Path of the dataset to be used"
    )
    parser.add_argument(
        '--metadata_image_path', 
        type=str, 
        default="./processed_data/full-dev-indent-images.json",
        help="Path of the dataset to be used"
    )
    parser.add_argument(
        "--base_img_dir", 
        default="/hdd0/chanhwi/mimic-cxr-jpg-2.1.0.physionet.org/files", 
        help="Path to a CXR img and radiology report folder"
    )   
    parser.add_argument(
        "--base_rr_dir", 
        default="/hdd0/chanhwi/physionet.org/files/mimic-cxr/2.1.0/files", 
        help="Path to a CXR img and radiology report folder"
    )   
    parser.add_argument(
        "--output_path",
        required=False,
        help="Path to save outputs. Default: os.path.join('output-' + args.model_name_or_path.replace('/', '-'))",
    )   

    args = parser.parse_args()

    if args.output_path:
        OUTPUT_DIR = args.output_path
    else:
        OUTPUT_DIR = os.path.join("output-" + args.model_name_or_path.replace("/", "-"))
        logger.warning(f"OUTPUT_DIR set to {OUTPUT_DIR}")

    if os.path.exists(OUTPUT_DIR) and os.path.isdir(OUTPUT_DIR):
        logger.warning(f"Path {OUTPUT_DIR} seems to exist! Not making new dir")
    else:
        os.makedirs(OUTPUT_DIR)
    
    vit_processor = VisionInstructModel(args.model_name_or_path)
    
    torch.no_grad() # This code is for inference only
    rad_repo_comparison = RadiologyEvaluator()
        
    output = []
    
    hash2meta = load_hash2meta_dict(args.metadata_path, args.metadata_image_path)
    
    for index, hash in enumerate(tqdm(hash2meta)):
        meta_data = hash2meta[hash]
        if len(meta_data['data_paths']) == 0:
            continue
        
        system_prompt = '''
        You are an advanced AI model specializing in radiology.

        Your task is to analyze a set of chest X-ray (CXR) images and generate the “FINDINGS” section of a radiology report based on the provided clinical context. You will be given either an “INDICATION” section. Please follow these guidelines:

        1. Thoroughly review and interpret the given CXR images.
        2. Refer to the clinical context outlined in the “INDICATION” section.
        3. Write a concise and accurate “FINDINGS” section that:
        - Summarizes the most significant observations.
        - Highlights critical or urgent abnormalities.
        - Provides clinically actionable interpretations relevant to the provided context.
        4. Maintain a professional and precise tone, avoiding unnecessary details.

        Your output should be clear, focused, and directly aligned with the clinical information provided.
        '''

        for study in meta_data['data_paths']:
            images = load_image(args.base_img_dir, ['/'.join(study[1])])
            rr = load_radiology_report(args.base_rr_dir, ['/'.join(study[1])])
            indication, reason, impression, findings = extract_part(rr)
            
            if ((indication or reason) and findings):

                indication = indication if indication else ""
                reason = reason if reason else ""
                impression = impression if impression else ""
                findings = findings if findings else ""

                INPUT = indication + reason
                GOLD_FINDINGS = impression + findings

                output_dict = {
                    'radiology_note': rr, 
                    'findings': GOLD_FINDINGS, 
                    'model_output': '', 
                    'score': {}
                }
                if "chexagent" in args.model_name_or_path.lower():
                    user_prompt = INPUT
                else:
                    user_prompt = f'''
                    You are given a set of chest X-ray (CXR) images and an “INDICATION” section describing the clinical context for this exam. Based on your analysis of the images and the given background, please provide only the “FINDINGS” section of the radiology report. 
                                                
                    --------------------------------        
                    INDICATION:
                    {INPUT}

                    FINDINGS:
                    {{You have to fill out here based on INDICATION section}}
                    '''

                PRED = vit_processor.run(
                    system_prompt, 
                    user_prompt,
                    images, 
                )                
                
                compare_results = rad_repo_comparison.evaluate_reports(GOLD_FINDINGS, PRED)
                output_dict['model_output'] += PRED
                output_dict['score'].update({'bleu': compare_results['bleu'], 
                                             'rouge': compare_results['rouge'],
                                              'similarity' : compare_results['semantic_similarity'], 
                                              'jaccard': compare_results['jaccard_scores'], 
                                              'f1chexbert': compare_results['f1chexbert'], 
                                              'ner': compare_results['ner_scores']})
                output.append(output_dict)

    averages = calculate_averages(args, output)

    with open(os.path.join(OUTPUT_DIR, f'averages_{args.model_name_or_path.replace("/", "_")}.json'), 'w') as file:
        json.dump(averages, file, indent=2)
