#!/usr/bin/env python
"""
Extract all images from train/dev/test datasets and save them as JPG files.
"""
import os
import json
import argparse
from PIL import Image
from tqdm import tqdm
from dataloader import load_hash2meta_dict, CXRDecisionTree
import shutil


def load_data_ids(path):
    """Load all data IDs from a JSONL file."""
    data_ids = []
    with open(path, "r") as file:
        for line in file:
            sample = json.loads(line)
            data_ids.append(sample['id'])
    return data_ids


def extract_and_save_images(args):
    """
    Extract images from train/dev/test datasets and save them to output folder.
    """
    # Initialize decision tree for image selection
    decision_tree = CXRDecisionTree()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each dataset split
    splits = ['train', 'dev', 'test']
    
    for split in splits:
        print(f"\n{'='*60}")
        print(f"Processing {split.upper()} set...")
        print(f"{'='*60}")
        
        # Get data and metadata paths
        if split == 'train':
            data_path = args.train_data_path
            metadata_image_path = args.train_metadata_image_path
        elif split == 'dev':
            data_path = args.dev_data_path
            metadata_image_path = args.dev_metadata_image_path
        else:  # test
            data_path = args.test_data_path
            metadata_image_path = args.test_metadata_image_path
        
        # Load data IDs
        print(f"Loading data from: {data_path}")
        data_ids = load_data_ids(data_path)
        print(f"Found {len(data_ids)} samples in {split} set")
        
        # Load hash2meta mapping
        print(f"Loading metadata from: {metadata_image_path}")
        hash2meta = load_hash2meta_dict(args.metadata_path, metadata_image_path)
        
        # Create split-specific output directory
        split_output_dir = os.path.join(args.output_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)
        
        # Track statistics
        successful_copies = 0
        failed_copies = 0
        missing_images = 0
        
        # Process each sample
        for idx, sample_id in enumerate(tqdm(data_ids, desc=f"Extracting {split} images")):
            try:
                # Get metadata for this sample
                if sample_id not in hash2meta:
                    print(f"\nWarning: Sample ID {sample_id} not found in metadata")
                    missing_images += 1
                    continue
                
                # Get all image paths for this sample
                all_img_data_paths = hash2meta[sample_id]['metadata_filtered']
                
                if not all_img_data_paths:
                    print(f"\nWarning: No images found for sample ID {sample_id}")
                    missing_images += 1
                    continue
                
                # Select the best CXR image based on decision tree
                selected_img_data = decision_tree.select_best_cxr(all_img_data_paths)
                
                if selected_img_data is None:
                    print(f"\nWarning: Could not select image for sample ID {sample_id}")
                    missing_images += 1
                    continue
                
                # Get the image path
                img_relative_path = selected_img_data[1]
                img_full_path = os.path.join(args.base_img_dir, img_relative_path)
                
                # Check for resized version
                if "512_resized" not in img_full_path.lower():
                    base, ext = img_full_path.rsplit('.', 1)
                    img_full_path_resized = f"{base}_512_resized.{ext}"
                    if os.path.exists(img_full_path_resized):
                        img_full_path = img_full_path_resized
                
                # Check if source image exists
                if not os.path.exists(img_full_path):
                    print(f"\nWarning: Image file not found: {img_full_path}")
                    failed_copies += 1
                    continue
                
                # Create output filename using sample ID
                output_filename = f"{sample_id}.jpg"
                output_path = os.path.join(split_output_dir, output_filename)
                
                # Load and save as JPG
                try:
                    with Image.open(img_full_path) as img:
                        # Convert to RGB (in case of RGBA or other formats)
                        rgb_img = img.convert('RGB')
                        # Save as JPG
                        rgb_img.save(output_path, 'JPEG', quality=95)
                    successful_copies += 1
                    
                except Exception as e:
                    print(f"\nError processing image {img_full_path}: {str(e)}")
                    failed_copies += 1
                    continue
                
            except Exception as e:
                print(f"\nError processing sample {sample_id}: {str(e)}")
                failed_copies += 1
                continue
        
        # Print statistics for this split
        print(f"\n{split.upper()} Set Statistics:")
        print(f"  Total samples: {len(data_ids)}")
        print(f"  Successfully saved: {successful_copies}")
        print(f"  Failed to process: {failed_copies}")
        print(f"  Missing images: {missing_images}")
        print(f"  Output directory: {split_output_dir}")
    
    print(f"\n{'='*60}")
    print(f"All images have been extracted to: {args.output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Extract all images from train/dev/test datasets')
    
    # Data paths
    parser.add_argument(
        '--train_data_path', 
        type=str, 
        default="../dataset/train_summarization2/plain_output.jsonl",
        help="Path to train data JSONL file"
    )
    parser.add_argument(
        '--dev_data_path', 
        type=str, 
        default="../dataset/dev_summarization2/plain_output.jsonl",
        help="Path to dev data JSONL file"
    )
    parser.add_argument(
        '--test_data_path', 
        type=str, 
        default="../dataset/test_summarization2/plain_output.jsonl",
        help="Path to test data JSONL file"
    )
    
    # Metadata paths
    parser.add_argument(
        '--metadata_path', 
        type=str, 
        default="../dataset/metadata.json",
        help="Path to metadata JSON file"
    )
    parser.add_argument(
        '--train_metadata_image_path', 
        type=str, 
        default="../dataset/train_summarization/full-train-indent-images.json",
        help="Path to train metadata image JSON file"
    )
    parser.add_argument(
        '--dev_metadata_image_path', 
        type=str, 
        default="../dataset/dev_summarization/full-dev-indent-images.json",
        help="Path to dev metadata image JSON file"
    )
    parser.add_argument(
        '--test_metadata_image_path', 
        type=str, 
        default="../dataset/test_summarization/full-test-indent-images.json",
        help="Path to test metadata image JSON file"
    )
    
    # Image directory
    parser.add_argument(
        "--base_img_dir", 
        type=str,
        default="../mimic-cxr-jpg-2.1.0.physionet.org/files",
        help="Path to base CXR image directory"
    )
    
    # Output directory
    parser.add_argument(
        "--output_dir", 
        type=str,
        default="../extracted_images",
        help="Directory to save extracted images"
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("Configuration:")
    print(f"  Train data: {args.train_data_path}")
    print(f"  Dev data: {args.dev_data_path}")
    print(f"  Test data: {args.test_data_path}")
    print(f"  Base image directory: {args.base_img_dir}")
    print(f"  Output directory: {args.output_dir}")
    print()
    
    # Extract and save images
    extract_and_save_images(args)


if __name__ == "__main__":
    main()
