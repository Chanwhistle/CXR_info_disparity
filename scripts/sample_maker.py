#!/usr/bin/env python
import os
from utils import load_data, get_args
from dataloader import VLM_Dataset
import warnings
from tqdm import tqdm

warnings.filterwarnings(
    "ignore", 
    message="Could not find a config file in", 
    category=UserWarning
)

def train(args):
    train_data = load_data(args.train_data_path, args.summary_type)
    eval_data = load_data(args.dev_data_path, args.summary_type)
    test_data = load_data(args.test_data_path, args.summary_type)
    print(f"TOTAL TRAIN DATASET : {len(train_data)}")
    print(f"TOTAL EVAL  DATASET : {len(eval_data)}")
    print(f"TOTAL EVAL  DATASET : {len(test_data)}")

    print(args.summary_type)
    

    train_dataset = VLM_Dataset(args, 
                                train_data, 
                                args.train_metadata_image_path, 
                                args.use_cxr_image,
                                args.use_rad_report,
                                args.use_discharge_note,
                                shuffle=False)
    eval_dataset = VLM_Dataset(args, 
                               eval_data, 
                               args.dev_metadata_image_path, 
                               args.use_cxr_image,
                               args.use_rad_report,
                               args.use_discharge_note,
                               shuffle=False)
    test_dataset = VLM_Dataset(args, 
                               test_data, 
                               args.test_metadata_image_path, 
                               args.use_cxr_image,
                               args.use_rad_report,
                               args.use_discharge_note,
                               shuffle=False)
    
    train_output, dev_output, test_output = [], [], []
    for sample in tqdm(train_dataset):
        sample_dict = {
            "id": sample['id'],
            "summary_type": sample["summary_type"],
            "text": sample['chat_template'][1]['content'][0]['text'],
            "label": sample['label']
        }
        train_output.append(sample_dict)
        
    for sample in tqdm(eval_dataset):
        sample_dict = {
            "id": sample['id'],
            "summary_type": sample["summary_type"],
            "text": sample['chat_template'][1]['content'][0]['text'],
            "label": sample['label']
        }
        dev_output.append(sample_dict)

    for sample in tqdm(test_dataset):
        sample_dict = {
            "id": sample['id'],
            "summary_type": sample["summary_type"],
            "text": sample['chat_template'][1]['content'][0]['text'],
            "label": sample['label']
        }
        test_output.append(sample_dict)

    import json
    with open(os.path.join(args.output_path, "train", f"train_{args.summary_type}.json"), "w") as f:
        json.dump(train_output, f, indent=4)
    with open(os.path.join(args.output_path, "dev", f"dev_{args.summary_type}.json"), "w") as f:
        json.dump(dev_output, f, indent=4)
    with open(os.path.join(args.output_path, "test", f"test_{args.summary_type}.json"), "w") as f:
        json.dump(test_output, f, indent=4)

if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.output_path, exist_ok=True)

    print(f"Base model           : {args.model_name_or_path}")
    print(f"Base learning rate   : {args.lr}")
    print(f"Use CXR image        : {args.use_cxr_image}")
    print(f"Use Radiology note   : {args.use_rad_report}")
    print(f"Use Discharge note   : {args.use_discharge_note}")
    print(f"LoRA Settting        : {args.lora_setting}")
    train(args)
