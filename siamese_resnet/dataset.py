from __future__ import annotations
import os
import shutil
import cv2
import random
import numpy as np
import torch
from tqdm import tqdm
from mtcnn import MTCNN
from torch.utils.data import Dataset
from collections import defaultdict



def triplet_collate_fn(batch):
    """Optimized collate function for triplet face recognition
    """
    # Separate components
    anchors, positives, negatives = [], [], []
    
    # Process each item in the batch
    # and append to separate lists
    for item in batch:
        anchors.append(item['anchor'])
        positives.append(item['positive'])
        negatives.append(item['negative'])
    
    # Convert to tensor (no copy)
    return {
        'anchor': torch.stack(anchors, dim=0),
        'positive': torch.stack(positives, dim=0),
        'negative': torch.stack(negatives, dim=0)
    }


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
            
            # Process each image
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

    
    

class TripletDatasetGenerator:
    """Professional-grade dataset handler with optimal splits and caching"""
    
    def __init__(self, data_root, split_ratios=(0.8, 0.1, 0.1), triplets_per_identity=10):
        self.data_root = data_root
        self.split_ratios = split_ratios
        self.triplets_per_identity = triplets_per_identity
        self.identity_map = self._build_identity_map()
        self.identities = list(self.identity_map.keys())
        random.shuffle(self.identities)
        
    def _build_identity_map(self):
        """Efficient directory scanning with validity checks"""
        identity_map = defaultdict(list)
        for identity in os.listdir(self.data_root):
            identity_dir = os.path.join(self.data_root, identity)
            if os.path.isdir(identity_dir):
                valid_images = [f for f in os.listdir(identity_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if len(valid_images) >= 2:  # Minimum for triplet creation
                    identity_map[identity] = valid_images
        return identity_map
    
    def create_splits(self):
        """Stratified split preserving class distribution"""
        total = len(self.identities)
        train_end = int(total * self.split_ratios[0])
        val_end = train_end + int(total * self.split_ratios[1])
        
        return {
            'train': self.identities[:train_end],
            'val': self.identities[train_end:val_end],
            'test': self.identities[val_end:]
        }
    
    def generate_triplets(self, split_identities):
        """Efficient triplet generation with hard negative mining preparation"""
        triplets = []
        identity_list = list(split_identities)
        
        for identity in identity_list:
            samples = self.identity_map[identity]
            for _ in range(self.triplets_per_identity):
                # Anchor-Positive pair
                anchor, positive = random.sample(samples, 2)
                
                # Hard Negative: Different identity from same split
                neg_identity = random.choice(identity_list)
                while neg_identity == identity:
                    neg_identity = random.choice(identity_list)
                negative = random.choice(self.identity_map[neg_identity])
                
                triplets.append((
                    os.path.join(self.data_root, identity, anchor),
                    os.path.join(self.data_root, identity, positive),
                    os.path.join(self.data_root, neg_identity, negative)
                ))
        return triplets



import cv2
import torch
from torch.utils.data import Dataset
class TripletDataset(Dataset):
    """Optimized dataset loader with zero in-memory storage"""
    
    def __init__(self, triplets):
        self.triplets = triplets
        self.num_triplets = len(triplets)
        self.triplets_list = []

        # Preload the triplet images
        for i in range(self.num_triplets):
            a_path, p_path, n_path = self.triplets[i]
            anchor_img, positive_img, negative_img = cv2.cvtColor(cv2.imread(a_path), cv2.COLOR_BGR2RGB), cv2.cvtColor(cv2.imread(p_path), cv2.COLOR_BGR2RGB), cv2.cvtColor(cv2.imread(n_path), cv2.COLOR_BGR2RGB)
            if anchor_img.shape != (224, 224, 3):
                anchor_img = cv2.resize(anchor_img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
            if positive_img.shape != (224, 224, 3):
                positive_img = cv2.resize(positive_img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
            if negative_img.shape != (224, 224, 3):
                negative_img = cv2.resize(negative_img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
                
            # convert to tensor (no copy)
            anchor_img = torch.as_tensor(anchor_img, dtype=torch.float32).permute(2, 0, 1)
            positive_img = torch.as_tensor(positive_img, dtype=torch.float32).permute(2, 0, 1)
            negative_img = torch.as_tensor(negative_img, dtype=torch.float32).permute(2, 0, 1)
            
            # normalize to [0, 1] (no copy)
            anchor_img.div_(255)
            positive_img.div_(255)
            negative_img.div_(255)
            
            # Append to the list
            self.triplets_list.append((anchor_img, positive_img, negative_img))
        
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        """OpenCV-based loading with minimal preprocessing"""
        # Load the triplet images
        anchor_img, positive_img, negative_img = self.triplets_list[idx]
        # Return the triplet images as a dictionary
        return {
            'anchor': anchor_img,
            'positive': positive_img,
            'negative': negative_img
        }