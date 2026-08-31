import os
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, SparseVector

# Load environment variables
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

if not all([QDRANT_URL, QDRANT_API_KEY, AWS_BUCKET_NAME]):
    print("ERROR: Missing Qdrant or AWS credentials in .env file.")
    sys.exit(1)

project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "Processed_Data"
parquet_path = data_dir / "products.parquet"
dense_path = data_dir / "image_embeddings.npy"
sparse_path = data_dir / "sparse_text_embeddings.json"

COLLECTION_NAME = "abo_products"


def migrate_to_cloud():
    print(f"Connecting to Qdrant Cloud at {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    print("Loading dataset and pre-computed embeddings...")
    df = pd.read_parquet(parquet_path)
    dense_embeddings = np.load(dense_path)
    with open(sparse_path, "r") as f:
        sparse_embeddings = json.load(f)

    # Note: We do NOT create the collection here. 
    # Qdrant Cloud UI allows you to create it, or you can run indexing/build_index.py 
    # slightly modified. Let's assume you've set it up or we create it if missing.
    if not client.collection_exists(COLLECTION_NAME):
        from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams
        print(f"Creating collection '{COLLECTION_NAME}' in cloud...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"dense_image": VectorParams(size=768, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse_text": SparseVectorParams()},
        )

    # Base S3 URL for images
    s3_base_url = f"https://{AWS_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/images/"

    print("Starting upsert to Qdrant Cloud...")
    batch_size = 250
    points = []

    for i, row in df.iterrows():
        # Replace the local image path with the public S3 URL
        payload = row.to_dict()
        image_filename = Path(payload["main_image_path"]).name
        payload["image_url"] = s3_base_url + image_filename
        
        # Remove the local path
        payload.pop("main_image_path", None)

        point = PointStruct(
            id=i,
            vector={
                "dense_image": dense_embeddings[i].tolist(),
                "sparse_text": SparseVector(
                    indices=sparse_embeddings[i]["indices"],
                    values=sparse_embeddings[i]["values"]
                )
            },
            payload=payload
        )
        points.append(point)

        if len(points) >= batch_size:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            points = []
            print(f"Upserted {i + 1}/{len(df)} points...", flush=True)

    # Upsert remaining
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        
    print(f"Successfully migrated all {len(df)} products to Qdrant Cloud!")


if __name__ == "__main__":
    migrate_to_cloud()
