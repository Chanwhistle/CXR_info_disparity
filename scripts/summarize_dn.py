import json
import os
import torch
import logging
from tqdm import tqdm
from utils import *
from model import *
from huggingface_hub import login

login("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")

def summarize(args):

    model, processor = load_model(
        args,
        model_id=args.model_name_or_path, 
        inference=False
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    model.config.temperature = 0.1
    data = load_data(f"/ssd1/chanhwi/long-clinical-doc/dataset/{args.set_name}_summarization/{args.set_name}.jsonl", "original_note")

    if args.set_name == "train":
        data_image_path = args.train_metadata_image_path
    if args.set_name == "dev":
        data_image_path = args.dev_metadata_image_path
    if args.set_name == "test":
        data_image_path = args.test_metadata_image_path

    dataset = VLM_Dataset(args, 
                        data, 
                        data_image_path, 
                        args.use_cxr_image,
                        args.use_rad_report,
                        args.use_discharge_note,
                        shuffle=False,
                        summarize=True)
    
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,        
        collate_fn=custom_data_collator(processor, use_cxr_image=args.use_cxr_image, summary_type="original_note"),
    )
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    summary_file = os.path.join(args.output_path, f'{args.summary_type}_output.jsonl')
    with open(summary_file, 'w') as f:
        for idx, sample in enumerate(tqdm(data_loader, total=len(data_loader))):
            sample_device = {k: v.to(device) for k, v in sample.items() if isinstance(v, torch.Tensor)}
            with torch.no_grad():
                outputs = model(**sample_device)
                
            decoded = processor.decode(outputs, skip_special_tokens=True)

            output = {
                'id': sample['ids'][0],
                'label': int(sample_device['labels'][0]), 
                'original_note': [dataset[idx]['original_note'], len(processor.tokenizer.encode(dataset[idx]['original_note']))], 
                f'{args.summary_type}': [decoded[9:], len(outputs[2:])],
            }
            
            f.write(str(json.dumps(output)) + '\n')
            f.flush()

if __name__ == "__main__":
    args = get_args()
    summarize(args)