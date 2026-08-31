import os
import argparse
from pathlib import Path
import torch
from transformers import AutoProcessor, AutoModel
from peft import PeftModel

def export_to_onnx(base_model_name: str, lora_adapter_path: str, output_dir: str):
    print(f"Loading base model: {base_model_name}")
    processor = AutoProcessor.from_pretrained(base_model_name)
    base_model = AutoModel.from_pretrained(base_model_name)

    if os.path.exists(lora_adapter_path):
        print(f"Applying LoRA adapter from {lora_adapter_path}")
        model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        # Merge LoRA weights into base model for export
        model = model.merge_and_unload()
    else:
        print("No LoRA adapter found, using base model.")
        model = base_model

    model.eval()
    
    # We only need the text model for inference on the search server
    text_model = model.text_model

    print("Tracing text model to ONNX...")
    
    # Create dummy inputs
    dummy_text = ["This is a test query for ONNX export."]
    inputs = processor(text=dummy_text, padding="max_length", truncation=True, return_tensors="pt")
    
    onnx_path = Path(output_dir) / "siglip_text.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export using PyTorch ONNX exporter
    torch.onnx.export(
        text_model,
        (inputs["input_ids"],),
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_ids"],
        output_names=["pooler_output"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "pooler_output": {0: "batch_size"},
        }
    )
    
    print(f"ONNX export successful! Saved to {onnx_path}")
    print("This lightweight ONNX model will be used for Render.com deployment to fit within 512MB RAM.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="google/siglip-base-patch16-224")
    parser.add_argument("--lora", type=str, default="models/siglip_finetuned")
    parser.add_argument("--output", type=str, default="models/onnx")
    args = parser.parse_args()
    
    export_to_onnx(args.base_model, args.lora, args.output)
