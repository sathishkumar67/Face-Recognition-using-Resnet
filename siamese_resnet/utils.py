from __future__ import annotations
import os
import zipfile
from tqdm import tqdm

def zip_directory_with_progress(src_dir: str, output_zip: str) -> None:
    """
    Zips the contents of src_dir into output_zip, showing a progress bar.
    """
    # Gather all file paths
    file_paths = []
    for root, _, files in os.walk(src_dir):
        for file in files:
            full_path = os.path.join(root, file)
            file_paths.append(full_path)

    # Create the zip file and write files with progress
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in tqdm(file_paths, desc="Zipping files"):
            # Write file to the zip archive, preserving directory structure
            arcname = os.path.relpath(file, start=src_dir)
            zipf.write(file, arcname)


def zip_dir(folder_path: str, output_zip: str) -> None:
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