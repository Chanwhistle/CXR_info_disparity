#!/usr/bin/env python
import wandb
import torch
import numpy as np
import os
import json
from tqdm import tqdm
from model import load_model
from torch.utils.data import DataLoader
from utils import *
from dataloader import VLM_Dataset, custom_data_collator
from sklearn.metrics import *
import torch.nn.functional as F
import warnings

warnings.filterwarnings(
    "ignore", 
    message="Could not find a config file in", 
    category=UserWarning
)
import os, random, numpy as np, torch

# Set all random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)

# Set seed
set_seed(random.randint(1, 10000))

from torchvision import transforms

eval_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # (Normalize 등도 순서·값 그대로)
])


def inference(model, device, data_loader, args, set_type, summary_type_override=None, best_threshold_override=None):
    summary_type = summary_type_override if summary_type_override is not None else args.summary_type

    output_dict = {"data": []}
    predictions_file = os.path.join(args.output_path, f'{set_type}_predictions_{summary_type}.txt')
    os.makedirs(os.path.dirname(predictions_file), exist_ok=True)
    with open(predictions_file, 'w') as f:
        for batch in tqdm(data_loader, total=len(data_loader)):
            batch_device = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            with torch.no_grad():
                outputs = model(**batch_device)
            logits = outputs["logits"]
            probabilities = F.softmax(logits, dim=-1)
            
            for i in range(len(probabilities)):
                sample_id = batch["ids"][i]
                label = int(batch["labels"][i].item()) if "labels" in batch else None
                prob = probabilities[i].cpu().tolist()
                logit = logits[i].cpu().tolist()
                output = {
                    "id": sample_id,
                    "label": label,
                    "logits": logit,
                    "probabilities": prob
                }
                # rounded_output = round_numbers(output, precision=6)
                # f.write(json.dumps(rounded_output) + "\n")
                # output_dict["data"].append(rounded_output)
                if set_type == "test":
                    output['prediction'] = 0 if best_threshold_override > prob[1] else 1
                f.write(json.dumps(output) + "\n")
                output_dict["data"].append(output)

    labels, probs = zip(*[(ele['label'], ele['probabilities']) for ele in output_dict['data']])
    if not best_threshold_override:
        best_threshold = find_best_f1(labels, probs)
    else:
        best_threshold = best_threshold_override

    log_result(labels, probs, best_threshold, args.output_path, summary_type, set_type)

    return best_threshold, labels, probs 

def process_vote(model, device, processor, set_type, soft_threshold=None, any_threshold=None):
    labels, probs = run_inference_for_vote(model, device, processor, set_type)
    
    soft_vote_preds, any_vote_preds = compute_vote_predictions(probs)
    
    if soft_threshold is None and any_threshold is None:
        best_threshold_sv = find_best_f1(labels, soft_vote_preds)
        best_threshold_av = find_best_f1(labels, any_vote_preds)
    else:
        best_threshold_sv = soft_threshold
        best_threshold_av = any_threshold
    
    log_result(labels, soft_vote_preds, best_threshold_sv, args.output_path, "Soft voting", set_type)
    log_result(labels, any_vote_preds, best_threshold_av, args.output_path, "Any voting", set_type)
    
    return best_threshold_sv, best_threshold_av

def run_inference_for_vote(model, device, processor, set_type, custom_thresholds=None):

    all_probs = []
    for idx, partial_summary_type in enumerate(["plain", "risk_factor", "timeline"]):
        print(f"Evaluating on {partial_summary_type} for {set_type} set...")
        
        _, labels, probs = inference(
            model, device, prepare_loader(partial_summary_type, args, set_type, processor), args, 
            set_type=set_type, 
            summary_type_override=partial_summary_type,
        )
        
        all_probs.append(probs)

    return labels, all_probs
    
def test(args):    
    model, processor = load_model(
        args,
        model_id=args.model_name_or_path, 
        inference=True
    )
    
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    if args.summary_type == "merged":
        dev_soft_threshold, dev_any_threshold = process_vote(model, device, processor, "dev")
        _, _ = process_vote(model, device, processor, "test", dev_soft_threshold, dev_any_threshold)
    else:
        dev_loader = prepare_loader(args.summary_type, args, "dev", processor)
        test_loader = prepare_loader(args.summary_type, args, "test", processor)
        # print("Inferencing Dev set to find best threshold...")
        # dev_threshold, _, _ = inference(model, device, dev_loader, args, set_type="dev")
        dev_threshold = 0.1
        _, labels, probs = inference(model, device, test_loader, args, set_type="test", best_threshold_override=dev_threshold)

if __name__ == "__main__":
    args = get_args()
    print(f"Base model    : {args.model_name_or_path}")
    print(f"Summary type  : {args.summary_type}")
    print(f"Use CXR image      : {args.use_cxr_image}")
    print(f"Use Radiology note : {args.use_rad_report}")
    print(f"Use Discharge note : {args.use_discharge_note}")
    test(args)