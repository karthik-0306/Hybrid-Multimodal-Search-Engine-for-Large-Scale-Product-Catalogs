import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_BUCKET_NAME]):
    print("ERROR: Missing AWS credentials in .env file.")
    sys.exit(1)

# Initialize AWS S3 Client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

project_root = Path(__file__).resolve().parent.parent
images_dir = project_root / "Processed_Data" / "images"
dataset_file = project_root / "Processed_Data" / "products.parquet"


def upload_file(local_path: Path, s3_key: str, content_type: str = "image/jpeg"):
    """Uploads a single file to S3, skipping if it already exists."""
    try:
        # Skip if already uploaded
        try:
            s3_client.head_object(Bucket=AWS_BUCKET_NAME, Key=s3_key)
            return True
        except:
            pass

        s3_client.upload_file(
            Filename=str(local_path),
            Bucket=AWS_BUCKET_NAME,
            Key=s3_key,
            ExtraArgs={"ContentType": content_type}
        )
        return True
    except Exception as e:
        return f"Failed {local_path.name}: {str(e)}"


def upload_all():
    print(f"Connecting to AWS S3 Bucket: {AWS_BUCKET_NAME}")
    
    # 1. Upload the main parquet dataset
    if dataset_file.exists():
        print(f"\nUploading dataset {dataset_file.name}...")
        upload_file(dataset_file, "dataset/products.parquet", "application/octet-stream")
        print("Dataset uploaded successfully!")
    else:
        print("WARNING: products.parquet not found. Skipping.")

    # 2. Collect all images
    if not images_dir.exists():
        print(f"ERROR: Images directory not found at {images_dir}")
        return

    images = list(images_dir.rglob("*.jpg"))
    total_images = len(images)
    print(f"\nFound {total_images} images to upload.")

    if total_images == 0:
        return

    # 3. Multi-threaded upload
    print("Starting multi-threaded upload (20 concurrent threads)...")
    success_count = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=20) as executor:
        # Submit all tasks
        futures = {
            executor.submit(upload_file, img, f"images/{img.name}", "image/jpeg"): img 
            for img in images
        }

        # Process with progress bar
        with tqdm(total=total_images, unit="file", desc="Uploading") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is True:
                    success_count += 1
                else:
                    tqdm.write(result)  # Print error without breaking progress bar
                pbar.update(1)
                if pbar.n % 500 == 0:
                    print(f"Update: {pbar.n} / {total_images} processed...", flush=True)

    elapsed_mins = (time.time() - start_time) / 60
    print(f"\nUpload complete! Successfully uploaded {success_count}/{total_images} images in {elapsed_mins:.1f} minutes.")
    print(f"Your images are now live at: https://{AWS_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/images/...")


if __name__ == "__main__":
    upload_all()
