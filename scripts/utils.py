# 표준 라이브러리
import argparse
import json
import copy
import os

# 서드파티 라이브러리
from typing import Optional
import torch
import torch.nn.functional as F
from sklearn.metrics import *
from trl import SFTTrainer
from dataloader import *
import numpy as np

class AdapterOnlySFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        self.use_cxr_image = kwargs.pop('use_cxr_image')
        super().__init__(*args, **kwargs)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n")
        if self.use_cxr_image:
            try:
                torch.save(self.model.base_model.vision_model.get_adapter_state_dict(), os.path.join(output_dir, "vm_adapter.bin"))
                print("Saved vision model LoRA adapter...")
            except:
                torch.save(self.model.base_model.vision_model.state_dict(), os.path.join(output_dir, "vision_encoder.bin"))
                print("No vision model LoRA adapter!")
                print("Saved vision model...")

            torch.save(self.model.base_model.multi_modal_projector.state_dict(), os.path.join(output_dir, "multi_modal_projector.bin"))
            print("Saved multimodal projector...")

        torch.save(self.model.base_model.language_model.get_adapter_state_dict(), os.path.join(output_dir, "lm_adapter.bin"))
        print("Saved language model LoRA adapter...")
        torch.save(self.model.classifier.state_dict(), os.path.join(output_dir, "classifier.bin"))
        print("Saved classification head...")

    # def load_model(self, checkpoint_dir: str):
    #     if self.use_cxr_image:
    #         try:
    #             self.model.base_model.vision_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "vision_encoder.bin"), map_location="cpu", weights_only=True))
    #             print("Loaded vision model.")
    #         except:
    #             print("No vision model!")
            
    #         try:
    #             self.model.base_model.vision_model.load_adapter(peft_model_id=f"{checkpoint_dir}/vm_adapter")
    #             print("Loaded vision model LoRA adapter.")
    #         except:
    #             print("No vision model LoRA adapter!")

    #             self.model.base_model.multi_modal_projector.load_state_dict(torch.load(os.path.join(checkpoint_dir, "multi_modal_projector.bin"), map_location="cpu", weights_only=True))
    #             print("Loaded multimodal projector.")

    #     self.model.classifier.load_state_dict(torch.load(os.path.join(checkpoint_dir, "classifier.bin"), map_location="cpu", weights_only=True))
    #     print("Loaded classification head.")
            
    #     self.model.base_model.language_model.load_adapter(peft_model_id=f"{checkpoint_dir}/lm_adapter")
    #     print("Loaded language model LoRA adapter.")

    # def _load_best_model(self):
    #     best_checkpoint = self.state.best_model_checkpoint
    #     if best_checkpoint is None:
    #         print("No best model checkpoint found.")
    #         return
    #     print(f"Loading best model from {best_checkpoint}")
    #     self.load_model(best_checkpoint)

def extract_first_binary(s: str) -> str:
    """Extract the first occurrence of 0 or 1 in the given string."""
    for char in str(s):
        if char in ['0', '1']:
            return int(char)
    return 0
        
def prepare_loader(summary_type, args, set, processor):
    if set == "train":
        data_path = args.train_data_path
        image_path = args.train_metadata_image_path
    elif set == "dev":
        data_path = args.dev_data_path
        image_path = args.dev_metadata_image_path
    elif set == "test":
        data_path = args.test_data_path
        image_path = args.test_metadata_image_path
    
    data = load_data(data_path, summary_type)
    if args.debug:
        data = data[:30]
    
    dataset = VLM_Dataset(
        args, 
        data, 
        image_path, 
        args.use_cxr_image,
        args.use_rad_report,
        args.use_discharge_note,
        shuffle=False
    )
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,        
        num_workers=0,  # Set to 0 for deterministic behavior
        collate_fn=custom_data_collator(processor, use_cxr_image=args.use_cxr_image, summary_type=summary_type),
        pin_memory=True
    )
    
    return loader

def load_data(path, summary_type):
    dataset = []
    with open(path, "r") as file:
        for line in file:
            sample = json.loads(line)
            if summary_type == "merged":
                for summary_type_partial in ["plain", "risk_factor", "timeline"]:
                    dataset.append({'id' : sample['id'],
                                'label' : sample['label'],
                                'text' : sample[summary_type_partial],
                                'summary_type' : summary_type_partial})
            else:
                dataset.append({'id' : sample['id'],
                            'label' : sample['label'],
                            'text' : sample[summary_type],
                            'summary_type' : summary_type})
    return dataset

def round_numbers(obj, precision=4):
    if isinstance(obj, float):
        return round(obj, precision)
    elif isinstance(obj, list):
        return [round_numbers(x, precision) for x in obj]
    elif isinstance(obj, dict):
        return {k: round_numbers(v, precision) for k, v in obj.items()}
    else:
        return obj
        
def compute_metrics_auroc(eval_pred):
    logits, labels = eval_pred
    probabilities = F.softmax(torch.tensor(logits), dim=-1).numpy()
    predictions = (probabilities[:, 1] >= 0.5)
    auroc = roc_auc_score(labels, probabilities[:, 1])    
    f1 = f1_score(labels, predictions)
    return {"auroc": auroc, "f1": f1}

def load_adapter(current_state_dict, adapter_state):
    for key, value in adapter_state.items():
        if key in current_state_dict:
            current_state_dict[key].copy_(value)

def map_adapter_keys(adapter_state, adapter_name="language_model_adapter"):
    mapped_state = {}
    for key, value in adapter_state.items():
        parts = key.split('.')
        if 'lora_A' in parts or 'lora_B' in parts:
            weight_index = parts.index('weight')
            parts.insert(weight_index, adapter_name)
        mapped_key = '.'.join(parts)
        mapped_state[mapped_key] = value
    return mapped_state

def find_best_f1(labels, probs):
    precisions, recalls, thresholds = precision_recall_curve(labels, [prob[1] for prob in probs])
    precisions, recalls = precisions[:-1], recalls[:-1]
    
    f1_scores = np.zeros_like(thresholds)
    valid_indices = (precisions + recalls) > 0
    f1_scores[valid_indices] = 2 * (precisions[valid_indices] * recalls[valid_indices]) / (precisions[valid_indices] + recalls[valid_indices])
    best_threshold = thresholds[np.argmax(f1_scores)]
    
    return best_threshold

def log_result(labels, probs, best_threshold, output_path, summary_type, set_type):
    pos_probs = [p[1] for p in probs]
    preds = [1 if p >= best_threshold else 0 for p in pos_probs]
    best_positive_f1 = f1_score(labels, preds)
    auroc = float(roc_auc_score(labels, pos_probs))
    auprc = float(average_precision_score(labels, pos_probs))
    total_pos = 100 * sum(1 for ele in preds if ele == 1) / len(preds)
    
    score_file = os.path.join(output_path, "score.txt")
    with open(score_file, "a") as f:
        f.write(f"{set_type} evaluation completed\n")
        f.write(f"Summary type        : {summary_type}\n")
        f.write(f"Positive prediction : {sum(1 for ele in preds if int(ele)==1)}, Negative prediction: {sum(1 for ele in preds if int(ele)==0)}\n")
        f.write(f"Pos/Total           : {total_pos}%\n")
        f.write(f"Threshold           : {best_threshold}\n")
        f.write(f"Positive f1         : {best_positive_f1}\n")
        f.write(f"AUROC               : {auroc}\n")
        f.write(f"AUPRC               : {auprc}\n")
        f.write(classification_report(labels, preds, digits=4))
        f.write(f"Prediction evaluated on {len(preds)} instances. Make sure that this number match with your original dataset!\n\n")

    print(f"Inference complete. Predictions saved to {output_path}")
    
def compute_vote_predictions(predictions):
    """
    세 summary type의 예측 결과에서 soft vote와 any vote 예측값을 계산합니다.
    predictions: (plain_preds, risk_factor_preds, timeline_preds)
    각 예측값은 [prob_negative, prob_positive] 형식이라고 가정합니다.
    """
    plain_preds, risk_factor_preds, timeline_preds = predictions
    soft_votes = []
    any_votes = []
    
    for p, rf, t in zip(plain_preds, risk_factor_preds, timeline_preds):
        p_pos, rf_pos, t_pos = p[1], rf[1], t[1]
        average_pos_prob = (p_pos + rf_pos + t_pos) / 3.0
        soft_votes.append([1 - average_pos_prob, average_pos_prob])
        max_pos_prob = max(p_pos, rf_pos, t_pos)
        any_votes.append([1 - max_pos_prob, max_pos_prob])
        
    return soft_votes, any_votes

def get_args():
    parser = argparse.ArgumentParser(
        description="LLM zero-shot inference."
    )   
    parser.add_argument(
        "--model_name_or_path", 
        default="meta-llama/Llama-3.2-11B-Vision-Instruct",
        help="Hugging Face Hub or local path"
    )
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        default="../finetuned_model",
        help="Hugging Face Hub or local path"
    )
    parser.add_argument(
        "--train_data_path", 
        type=str, 
        default="../dataset/train_summarization/total_output.jsonl",
        help="Path to a folder or json file"
    )
    parser.add_argument(
        "--dev_data_path", 
        type=str, 
        default="../dataset/dev_summarization/total_output.jsonl",
        help="Path to a folder or json file"
    )
    parser.add_argument(
        "--test_data_path", 
        type=str, 
        default="../dataset/test_summarization/total_output.jsonl",
        help="Path to a folder or json file"
    )
    parser.add_argument(
        '--metadata_path', 
        type=str, 
        default="../dataset/metadata.json",
        help="Path of the dataset to be used"
    )
    parser.add_argument(
        '--train_metadata_image_path', 
        type=str, 
        default="../dataset/train_summarization/full-train-indent-images.json",
        help="Path of the dataset to be used"
    )
    parser.add_argument(
        '--dev_metadata_image_path', 
        type=str, 
        default="../dataset/dev_summarization/full-dev-indent-images.json",
        help="Path of the dataset to be used"
    )
    parser.add_argument(
        '--test_metadata_image_path', 
        type=str, 
        default="../dataset/test_summarization/full-test-indent-images.json",
        help="Path of the dataset to be used"
    )
    parser.add_argument(
        "--base_img_dir", 
        default="/hdd2/chanhwi/mimic-cxr-jpg-2.1.0.physionet.org/files", 
        help="Path to a CXR img folder"
    )
    parser.add_argument(
        "--base_rr_dir", 
        default="/hdd2/chanhwi/physionet.org/files/mimic-cxr/2.1.0/files", 
        help="Path to a radiology report folder"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='./finetuned_model', 
        help="Path to save outputs.",
    )
    parser.add_argument(
        "--summary_type", 
        required=False,
        type=str,
        default='plain', 
        help="Discharge note summary type."
    )
    parser.add_argument(
        "--set_name", 
        required=False,
        type=str,
        default='train', 
        help="set type."
    )
    parser.add_argument(
        '--num_epochs', 
        type=int, 
        default=3, 
        help="Use cxr image"
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        default=5e-5, 
        help="Learing rate"
    )
    parser.add_argument(
        '--lora_setting', 
        type=int, 
        default=2, 
        help="Ablation setting for lora"
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=4, 
        help="Use cxr image"
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42, 
        help="Use cxr image"
    )
    parser.add_argument(
        '--gradient_accumulation_steps', 
        type=int, 
        default=4, 
        help="Use cxr image"
    )
    parser.add_argument(
        '--debug', 
        action="store_true",
        help="taining mode"
    )
    parser.add_argument(
        "--finetune", 
        action="store_true",
        help="finetune"
    )
    parser.add_argument(
        "--summarize", 
        action="store_true",
        help="summarize"
    )
    parser.add_argument(
        "--zeroshot", 
        action="store_true",
        help="inference zeroshot"
    )
    parser.add_argument(
        '--wandb', 
        action="store_true",
        help="taining mode"
    )
    parser.add_argument(
        '--use_cxr_image', 
        action="store_true",
        help="Use cxr image"
    )
    parser.add_argument(
        '--use_rad_report', 
        action="store_true",
        help="Use radiology report"
    )
    parser.add_argument(
        '--use_discharge_note', 
        action="store_true",
        help="Use discharge note during inference"
    )
    parser.add_argument(
        '--use_pi', 
        action="store_true",
        help="Use personal information like gender, age, race"
    )

    return parser.parse_args()

