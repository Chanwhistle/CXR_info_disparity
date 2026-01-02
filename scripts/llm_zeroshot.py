import json, os
import torch
import argparse
import logging
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import *
from utils import *
from dataloader import *
from model import load_model

from huggingface_hub import login
login("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")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
print(f"Log will show from Level: {logger.getEffectiveLevel()}")


def get_prompt_end_index(token_ids, prompt_token_id):
    indices = (token_ids == prompt_token_id).nonzero(as_tuple=True)[0]
    if len(indices) > 0:
        return indices[-1].item() + 1 
    else:
        return 0 
    
def inference(args):    
    model, processor = load_model(
        args,
        model_id=args.model_name_or_path, 
        inference=False
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dev_loader = prepare_loader(args.summary_type, args, "dev", processor)
    test_loader = prepare_loader(args.summary_type, args, "test", processor)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    output_dict = {"data": []}
    predictions_file = os.path.join(args.output_path, f'dev_predictions_{args.summary_type}.txt')
    with open(predictions_file, 'w') as f:
        for batch in tqdm(dev_loader, total=len(dev_loader)):
            batch_device = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            with torch.no_grad():
                outputs = model(**batch_device)
                
            decoded = processor.decode(outputs, skip_special_tokens=True)
            
            output = {
                'label': int(batch_device['labels'][0]), 
                'prediction': decoded,
                'binary': extract_first_binary(decoded)
            }
            
            output_dict['data'].append(output)
            
            f.write(str(json.dumps(output)) + '\n')
            
        f.flush()

    preds, labels = zip(*[(ele['binary'], ele['label']) for ele in output_dict['data']])

    assert len(preds)==len(labels)

    with open(os.path.join(args.output_path, "score.txt"), "a") as f:
        f.write(f"dev evaluation completed\n")
        f.write(f"Summary type        : {args.summary_type}\n")
        f.write(f"Use CXR image      : {args.use_cxr_image}\n")
        f.write(f"Use Radiology note : {args.use_rad_report}\n")
        f.write(f"Use Discharge note : {args.use_discharge_note}\n")
        f.write(f"Positive prediction : {sum(1 for ele in preds if int(ele)==1)}, Negative prediction: {sum(1 for ele in preds if int(ele)==0)}\n")
        f.write(f"Positive f1         : {f1_score(preds, labels)}\n")
        f.write(classification_report(labels, preds, digits=4))
        f.write(f"Prediction evaluated on {len(preds)} instances. Make sure that this number match with your original dataset!\n\n")

    print(f"Inference complete. Predictions saved to {args.output_path}")


    predictions_file = os.path.join(args.output_path, f'test_predictions_{args.summary_type}.txt')
    with open(predictions_file, 'w') as f:
        for batch in tqdm(test_loader, total=len(test_loader)):
            batch_device = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            with torch.no_grad():
                outputs = model(**batch_device)
                
            decoded = processor.decode(outputs, skip_special_tokens=True)
            
            output = {
                'label': int(batch_device['labels'][0]), 
                'prediction': decoded,
                'binary': extract_first_binary(decoded)
            }
            
            output_dict['data'].append(output)
            
            f.write(str(json.dumps(output)) + '\n')
            
        f.flush()

    preds, labels = zip(*[(ele['binary'], ele['label']) for ele in output_dict['data']])

    assert len(preds)==len(labels)

    with open(os.path.join(args.output_path, "score.txt"), "a") as f:
        f.write(f"test evaluation completed\n")
        f.write(f"Summary type        : {args.summary_type}\n")
        f.write(f"Use CXR image      : {args.use_cxr_image}\n")
        f.write(f"Use Radiology note : {args.use_rad_report}\n")
        f.write(f"Use Discharge note : {args.use_discharge_note}\n")
        f.write(f"Positive prediction : {sum(1 for ele in preds if int(ele)==1)}, Negative prediction: {sum(1 for ele in preds if int(ele)==0)}\n")
        f.write(f"Positive f1         : {f1_score(preds, labels)}\n")
        f.write(classification_report(labels, preds, digits=4))
        f.write(f"Prediction evaluated on {len(preds)} instances. Make sure that this number match with your original dataset!\n\n")

    print(f"Inference complete. Predictions saved to {args.output_path}")
        
if __name__ == "__main__":
    args = get_args()
    inference(args)
