#!/usr/bin/env python
import wandb
import torch
import numpy as np
import os
import json
from tqdm import tqdm
from model import load_model
from utils import *
from sklearn.metrics import *
import torch.nn.functional as F
import warnings

def inference(model, device, data_loader, args, set_type):

    output_dict = {"data": []}
    predictions_file = os.path.join(args.output_path, f'{args.summary_type}{"_add_pi" if args.use_pi else ""}_{set_type}_predictions.txt')
    os.makedirs(os.path.dirname(predictions_file), exist_ok=True)
    with open(predictions_file, 'w') as f:
        for batch in tqdm(data_loader, total=len(data_loader)):
            batch_device = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            with torch.no_grad():
                outputs = model(**batch_device)
            logits = outputs["logits"]
            probabilities = F.softmax(logits.float(), dim=-1)
            
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
                f.write(json.dumps(output) + "\n")
                output_dict["data"].append(output)

    labels, probs = zip(*[(ele['label'], ele['probabilities']) for ele in output_dict['data']])

    log_result(args, labels, probs, args.output_path, set_type)


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
    model.config.use_cache = True
    
    # dev_loader = prepare_loader(args.summary_type, args, "dev", processor)
    test_loader = prepare_loader(args.summary_type, args, "test", processor)
    # inference(model, device, dev_loader, args, set_type="dev")
    inference(model, device, test_loader, args, set_type="test")

if __name__ == "__main__":
    args = get_args()
    set_seed(random.randint(1, 10000))
    test(args)