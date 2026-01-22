import os
import sys
from PIL import Image
from pathlib import Path
from tqdm import tqdm

TARGET_SIZE = 560

def resize_with_padding(img_path, output_path):
    """
    Resize image to 560x560 with black padding if needed.
    Maintains aspect ratio and centers the image on a black background.
    """
    # Open image
    img = Image.open(img_path)
    
    # Get original dimensions
    original_width, original_height = img.size
    
    # Calculate scaling factor to fit within 560x560 while maintaining aspect ratio
    scale = min(TARGET_SIZE / original_width, TARGET_SIZE / original_height)
    
    # Calculate new dimensions
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Resize image
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    
    # Create a new 560x560 black image
    final_img = Image.new('RGB', (TARGET_SIZE, TARGET_SIZE), (0, 0, 0))
    
    # Calculate position to center the resized image
    x_offset = (TARGET_SIZE - new_width) // 2
    y_offset = (TARGET_SIZE - new_height) // 2
    
    # Paste the resized image onto the black background
    final_img.paste(resized_img, (x_offset, y_offset))
    
    # Save the final image
    final_img.save(output_path, "JPEG", quality=95)

def main():
    source_dir = Path("saved_images")
    target_dir = Path("saved_images_2")
    
    if not source_dir.exists():
        print(f"Error: {source_dir} directory does not exist!")
        sys.exit(1)
    
    # Create target directory if it doesn't exist
    target_dir.mkdir(exist_ok=True)
    
    # Find all jpg files
    jpg_files = list(source_dir.rglob("*.jpg"))
    
    if not jpg_files:
        print(f"No JPG files found in {source_dir}")
        sys.exit(1)
    
    print(f"Found {len(jpg_files)} images to process...")
    
    # Process each image
    for img_path in tqdm(jpg_files, desc="Resizing images"):
        # Calculate relative path from source directory
        relative_path = img_path.relative_to(source_dir)
        
        # Create output path maintaining directory structure
        output_path = target_dir / relative_path
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Resize and save
        try:
            resize_with_padding(img_path, output_path)
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            continue
    
    print(f"\nDone! All images have been resized to {TARGET_SIZE}x{TARGET_SIZE} and saved to {target_dir}")

if __name__ == "__main__":
    main()
