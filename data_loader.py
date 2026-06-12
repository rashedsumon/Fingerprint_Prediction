import os
import shutil
import kagglehub

def load_dataset():
    """
    Downloads the SASH-VPV dataset from KaggleHub and 
    returns the path where the raw files are located.
    """
    print("Checking/Downloading SASH-VPV dataset...")
    # Downloads the latest version of the dataset
    download_path = kagglehub.dataset_download("sashinoventures/sash-vpv-subcutaneous-vascular-palm-vein-data")
    print(f"Dataset successfully available at: {download_path}")
    return download_path

if __name__ == "__main__":
    # Test download functionality locally
    path = load_dataset()