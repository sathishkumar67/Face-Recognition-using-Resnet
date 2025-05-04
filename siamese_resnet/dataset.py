from __future__ import annotations
import os
import shutil
import cv2
import random
from tqdm import tqdm
from mtcnn import MTCNN


def split_dataset(
    root_dir: str,
    save_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    target_size: tuple = (224, 224)) -> None:
    """
    Splits a facial image dataset into train/val/test sets, detects and crops faces,
    and saves processed images in organized directories.

    Args:
        root_dir: Directory containing identity subfolders with images
        save_dir: Directory to save processed images
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        target_size: Size to resize detected faces to
    """
    
    # Validate input ratios
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-9, "Ratios must sum to 1.0"
    
    # Initialize face detector
    face_detector = MTCNN()

    # Get and shuffle valid identities
    identities = _get_valid_identities(root_dir)
    random.shuffle(identities)

    # Split identities into groups
    split_groups = _split_identities(identities, train_ratio, val_ratio, test_ratio)
    splits = ["train", "val", "test"]

    # Process each split group
    for split_name, identity_group in zip(splits, split_groups):
        print(f"\nProcessing {split_name} set:")
        processed_count = 0
        skipped_images = 0

        # Main progress bar for identities
        identity_pbar = tqdm(identity_group, desc="Identities", leave=False)
        for identity in identity_pbar:
            identity_path = os.path.join(root_dir, identity)
            image_paths = _get_image_paths(identity_path)
            
            # Update identity progress bar description
            identity_pbar.set_postfix_str(f"Processing: {identity[:15]}...")
            
            # Create output directory for this identity
            output_dir = os.path.join(save_dir, split_name, identity)
            os.makedirs(output_dir, exist_ok=True)

            # Images progress bar for current identity
            image_pbar = tqdm(image_paths, 
                            desc=f"Images ({identity[:12]}...)", 
                            leave=False,
                            dynamic_ncols=True)
            for image_path in image_pbar:
                result = _process_and_save_image(
                    image_path=image_path,
                    output_dir=output_dir,
                    detector=face_detector,
                    target_size=target_size
                )
                
                if result == "skipped":
                    skipped_images += 1
                else:
                    processed_count += 1
                
                # Update image progress bar postfix
                image_pbar.set_postfix_str(f"Saved: {processed_count} | Skipped: {skipped_images}")
            
            image_pbar.close()

        identity_pbar.close()

        # Print statistics
        _print_split_stats(split_name, processed_count, skipped_images, save_dir)

    # Cleanup original data
    shutil.rmtree(root_dir)
    print("\n✅ Original dataset directory removed.")
def _get_valid_identities(root_dir: str) -> list:
    """Get identities with valid image directories"""
    return [
        identity for identity in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, identity))
    ]

def _split_identities(
    identities: list,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float)-> tuple:
    """Split identities into train/val/test groups"""
    total = len(identities)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)
    
    return (
        identities[:train_end],
        identities[train_end:val_end],
        identities[val_end:]
    )

def _get_image_paths(identity_path: str) -> list:
    """Get valid image paths for an identity"""
    return [
        os.path.join(identity_path, img)
        for img in os.listdir(identity_path)
        if img.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]

def _process_and_save_image(
    image_path: str,
    output_dir: str,
    detector: MTCNN,
    target_size: tuple)-> str:
    """Process single image and save cropped face"""
    try:
        # Read and convert image
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = detector.detect_faces(image)
        if not faces:
            return "skipped"

        # Select best face
        best_face = max(faces, key=lambda x: x['confidence'])
        x, y, w, h = best_face['box']
        
        # Crop and resize
        face_region = image[y:y+h, x:x+w]
        resized_face = cv2.resize(face_region, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Save image
        filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_cropped.jpg"
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, cv2.cvtColor(resized_face, cv2.COLOR_RGB2BGR))
        
        return "saved"

    except Exception as e:
        print(f"⚠️ Error processing {image_path}: {str(e)}")
        return "skipped"

def _print_split_stats(split_name: str, valid_count: int, skipped_count: int, save_dir: str) -> None:
    """Print statistics for processed split"""
    print(f"\n{split_name.capitalize()} Set Summary:")
    print(f"✅ Successfully saved faces: {valid_count}")
    print(f"⏭️ Skipped images (no face detected/errors): {skipped_count}")
    print(f"📊 Total images processed: {valid_count + skipped_count}")
    print(f"📁 Output directory: {os.path.abspath(save_dir)}/{split_name}\n")