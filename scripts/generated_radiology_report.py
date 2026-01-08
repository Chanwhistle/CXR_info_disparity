#!/usr/bin/env python
"""
Generate radiology reports using CheXagent 8B model.
This script processes CXR images from train/dev/test sets and generates 
radiology reports, saving them with 'generated_' prefix in the same location as original reports.
"""

import os
import sys
import argparse
import json
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

# Import from local modules
from dataloader import load_hash2meta_dict, CXRDecisionTree


def get_args():
    parser = argparse.ArgumentParser(description="Generate radiology reports using CheXagent 8B model.")
    
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="StanfordAIMI/CheXagent-8b",
        help="HuggingFace model name or local path for CheXagent"
    )
    parser.add_argument(
        "--base_img_dir",
        type=str,
        default="../saved_images",
        help="Base directory for CXR images"
    )
    parser.add_argument(
        "--base_rr_dir",
        type=str,
        default="../physionet.org/files/mimic-cxr/2.1.0/files",
        help="Base directory for radiology reports"
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="../dataset/metadata.json",
        help="Path to metadata.json"
    )
    parser.add_argument(
        "--train_metadata_image_path",
        type=str,
        default="../dataset/train_summarization/full-train-indent-images.json",
        help="Path to train metadata image json"
    )
    parser.add_argument(
        "--dev_metadata_image_path",
        type=str,
        default="../dataset/dev_summarization/full-dev-indent-images.json",
        help="Path to dev metadata image json"
    )
    parser.add_argument(
        "--test_metadata_image_path",
        type=str,
        default="../dataset/test_summarization/full-test-indent-images.json",
        help="Path to test metadata image json"
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "dev", "test"],
        help="Which splits to process (train, dev, test)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with limited samples"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip if generated report already exists"
    )
    
    return parser.parse_args()


def load_model(model_name_or_path, device):
    """Load CheXagent model, processor, and generation config."""
    print(f"Loading model: {model_name_or_path}")
    
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    )
    
    generation_config = GenerationConfig.from_pretrained(model_name_or_path)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None
    )
    
    if device == "cuda" and not hasattr(model, 'hf_device_map'):
        model = model.to(device)
    
    model.eval()
    print(f"Model loaded successfully on {device}")
    
    return model, processor, generation_config, dtype


def get_image_report_pairs(args, split):
    """Get all image-report path pairs for a given split."""
    
    # Select appropriate metadata path based on split
    if split == "train":
        metadata_image_path = args.train_metadata_image_path
    elif split == "dev":
        metadata_image_path = args.dev_metadata_image_path
    elif split == "test":
        metadata_image_path = args.test_metadata_image_path
    else:
        raise ValueError(f"Unknown split: {split}")
    
    # Load hash2meta dictionary
    hash2meta = load_hash2meta_dict(args.metadata_path, metadata_image_path)
    decision_tree = CXRDecisionTree()
    
    pairs = []
    
    for hash_key, meta_info in hash2meta.items():
        metadata_filtered = meta_info.get('metadata_filtered', [])
        
        if not metadata_filtered:
            continue
        
        # Select best CXR using decision tree
        selected_img_data = decision_tree.select_best_cxr(metadata_filtered)
        
        if selected_img_data is None:
            continue
        
        selected_img_data_path = selected_img_data[1]  # Path is at index 1
        
        if not selected_img_data_path:
            continue
        
        # Build image path
        image_filename = selected_img_data_path.split("/")[-1]
        name, extension = image_filename.split(".")
        
        if "_512_resized" in name:
            real_image_path = os.path.join(args.base_img_dir, split, image_filename)
        else:
            real_image_path = os.path.join(args.base_img_dir, split, f"{name}_512_resized.{extension}")
        
        # Build report paths (original and generated)
        path_parts = selected_img_data_path.split("/")[:3]
        
        if len(path_parts) == 3:
            rr_relative_path = '/'.join(path_parts) + ".txt"
            original_report_path = os.path.join(args.base_rr_dir, rr_relative_path)
            
            # Generated report path: same directory, with 'generated_' prefix on filename
            report_dir = os.path.dirname(original_report_path)
            report_filename = os.path.basename(original_report_path)
            generated_report_path = os.path.join(report_dir, f"generated_{report_filename}")
            
            pairs.append({
                'hash_key': hash_key,
                'image_path': real_image_path,
                'original_report_path': original_report_path,
                'generated_report_path': generated_report_path
            })
    
    return pairs


def generate_report(model, processor, generation_config, image_path, device, dtype):
    """
    Generate radiology report using CheXagent with the optimal 2-stage pipeline:
    1. Generate detailed FINDINGS (Image -> Text)
    2. Summarize into IMPRESSION (Text -> Text)
    """
    
    # 1. Load and process image
    image = Image.open(image_path).convert("RGB")
    
    # =========================================================================
    # STAGE 1: Findings Generation (Image -> Findings)
    # =========================================================================
    
    # CheXagent specific prompt for findings (Instruction Tuning Trigger)
    findings_prompt = "Describe the findings in the chest X-ray image."
    
    # Format: "USER: <image>\n{prompt} ASSISTANT:" 
    # Note: <image> token is crucial for the processor to insert visual embeddings
    formatted_findings_prompt = f"USER: <image>\n{findings_prompt} ASSISTANT:"
    
    inputs_findings = processor(
        images=image,
        text=formatted_findings_prompt,
        return_tensors="pt"
    ).to(device=device, dtype=dtype)
    
    with torch.no_grad():
        output_findings_ids = model.generate(
            **inputs_findings, 
            pad_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=256,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3, 
            do_sample=False,    
            early_stopping=True,
            generation_config=generation_config
        )[0]
        
    # Decode findings
    findings_text = processor.tokenizer.decode(output_findings_ids, skip_special_tokens=True)
    
    # Post-processing: Extract only the assistant's response
    if "ASSISTANT:" in findings_text:
        findings_text = findings_text.split("ASSISTANT:")[-1].strip()
    
    # =========================================================================
    # STAGE 2: Impression Generation (Findings -> Impression)
    # =========================================================================
    
    # Prompt to summarize findings into impression
    impression_prompt = f"Summarize the following findings into an impression: {findings_text}"
    formatted_impression_prompt = f"USER: {impression_prompt} ASSISTANT:"
    
    # For text-only input (Stage 2), we usually don't need the image, 
    # but maintaining the processor flow prevents errors. 
    # Ideally, we pass the image again or handle text-only if processor supports it.
    # Here we pass the image again to be safe with the multimodal encoder structure.
    inputs_impression = processor(
        images=image, 
        text=formatted_impression_prompt,
        return_tensors="pt"
    ).to(device=device, dtype=dtype)
    
    with torch.no_grad():
        output_impression_ids = model.generate(
            **inputs_impression, 
            pad_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=128,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3, 
            do_sample=False,    
            early_stopping=True,
            generation_config=generation_config,
        )[0]
        
    impression_text = processor.tokenizer.decode(output_impression_ids, skip_special_tokens=True)
    
    if "ASSISTANT:" in impression_text:
        impression_text = impression_text.split("ASSISTANT:")[-1].strip()

    # =========================================================================
    # Final Output Formatting
    # =========================================================================
    
    final_report = f"FINDINGS:\n{findings_text}\n\nIMPRESSION:\n{impression_text}"
    
    if not findings_text and not impression_text:
        return "[No report generated]"
        
    return final_report


def main():
    args = get_args()
    
    print("=" * 60)
    print("CheXagent Radiology Report Generator")
    print("=" * 60)
    print(f"Model: {args.model_name_or_path}")
    print(f"Device: {args.device}")
    print(f"Splits to process: {args.splits}")
    print(f"Skip existing: {args.skip_existing}")
    print("=" * 60)
    
    # Load model
    model, processor, generation_config, dtype = load_model(args.model_name_or_path, args.device)
    
    total_generated = 0
    total_skipped = 0
    total_failed = 0
    
    for split in args.splits:
        print(f"\n{'='*60}")
        print(f"Processing {split} split...")
        print(f"{'='*60}")
        
        # Get image-report pairs
        pairs = get_image_report_pairs(args, split)
        print(f"Found {len(pairs)} image-report pairs for {split}")
        
        if args.debug:
            pairs = pairs[:10]
            print(f"Debug mode: limiting to {len(pairs)} samples")
        
        split_generated = 0
        split_skipped = 0
        split_failed = 0
        
        for pair in tqdm(pairs, desc=f"Generating reports for {split}"):
            image_path = pair['image_path']
            generated_report_path = pair['generated_report_path']
            
            # # Skip if already exists
            # if args.skip_existing and os.path.exists(generated_report_path):
            #     split_skipped += 1
            #     continue
            
            # Check if image exists
            if not os.path.exists(image_path):
                print(f"\nWarning: Image not found: {image_path}")
                split_failed += 1
                continue
            
            try:
                # Generate report
                generated_report = generate_report(
                    model, processor, generation_config, image_path, 
                    args.device, dtype
                )
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(generated_report_path), exist_ok=True)
                
                # Save generated report
                with open(generated_report_path, 'w', encoding='utf-8') as f:
                    f.write(generated_report)
                
                split_generated += 1
                
            except Exception as e:
                print(f"\nError generating report for {image_path}: {str(e)}")
                split_failed += 1
                continue
        
        print(f"\n{split} split summary:")
        print(f"  Generated: {split_generated}")
        print(f"  Skipped (existing): {split_skipped}")
        print(f"  Failed: {split_failed}")
        
        total_generated += split_generated
        total_skipped += split_skipped
        total_failed += split_failed
    
    print(f"\n{'='*60}")
    print("Overall Summary")
    print(f"{'='*60}")
    print(f"Total Generated: {total_generated}")
    print(f"Total Skipped: {total_skipped}")
    print(f"Total Failed: {total_failed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
