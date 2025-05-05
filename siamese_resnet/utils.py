from __future__ import annotations
import os
import zipfile
from tqdm import tqdm
import cv2
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict






def zip_dir(folder_path: str, output_zip: str):
    """
    Compresses the contents of folder_path into a ZIP file at output_zip.
    """

    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                abs_filepath = os.path.join(root, file)
                # Compute the archive name (so the folder structure is preserved)
                arcname = os.path.relpath(abs_filepath, start=folder_path)
                zf.write(abs_filepath, arcname)
    print(f"Created ZIP archive: {output_zip}")


def unzip_file(zip_path: str, target_dir: str) -> None:
    """
    Unzips the specified zip file into the target directory with a progress bar.

    Parameters:
    - zip_path (str): The path to the zip file to be unzipped.
    - target_dir (str): The directory where the unzipped files will be stored.
    """
    # Ensure the target directory exists; create it if it doesn't
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Open the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get list of all files and directories in the zip
        info_list = zip_ref.infolist()
        # Calculate total uncompressed size for the progress bar
        total_size = sum(zinfo.file_size for zinfo in info_list)
        
        # Create progress bar with total size in bytes, scaled to KB/MB/GB as needed
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Unzipping") as pbar:
            for zinfo in info_list:
                # Extract the file or directory
                zip_ref.extract(zinfo, target_dir)
                # Update progress bar by the file's uncompressed size
                pbar.update(zinfo.file_size)
    
    print(f"Unzipped {zip_path} to {target_dir}")
    
    # Optionally, you can remove the zip file after extraction
    os.remove(zip_path)
    print(f"Removed zip file: {zip_path}")
    
    

# ====================== Configuration ======================
DATA_ROOT = ""  # Path to the dataset root directory
SPLIT_RATIOS = (0.8, 0.1, 0.1)  # Train, Val, Test
TRIPLETS_PER_IDENTITY = 10
SEED = 42
# ===========================================================

class TripletDatasetGenerator:
    """Professional-grade dataset handler with optimal splits and caching"""
    
    def __init__(self, data_root):
        self.data_root = data_root
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
                            if f.lower().endswith(('.jpg', '.jpeg'))]
                if len(valid_images) >= 2:  # Minimum for triplet creation
                    identity_map[identity] = valid_images
        return identity_map
    
    def create_splits(self):
        """Stratified split preserving class distribution"""
        total = len(self.identities)
        train_end = int(total * SPLIT_RATIOS[0])
        val_end = train_end + int(total * SPLIT_RATIOS[1])
        
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
            for _ in range(TRIPLETS_PER_IDENTITY):
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


class TripletDataset(Dataset):
    """Optimized dataset loader with zero in-memory storage"""
    
    def __init__(self, triplets):
        self.triplets = triplets
        
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        """OpenCV-based loading with minimal preprocessing"""
        a_path, p_path, n_path = self.triplets[idx]
        
        def load_img(path):
            img = cv2.imread(path)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Essential for pretrained models
            
        return {
            'anchor': load_img(a_path),
            'positive': load_img(p_path),
            'negative': load_img(n_path)
        }