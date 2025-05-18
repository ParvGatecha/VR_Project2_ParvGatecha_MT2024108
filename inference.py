import argparse
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from transformers import ViltProcessor, ViltForQuestionAnswering
from peft import PeftModel
import gdown
import zipfile
import shutil

BASE_MODEL_NAME = "dandelin/vilt-b32-finetuned-vqa"
LOCAL_ADAPTER_DOWNLOAD_DIR = "./downloaded_vilt_vqa_lora_adapters"

TARGET_IMAGE_HEIGHT = 384
TARGET_IMAGE_WIDTH = 384

def download_and_extract_adapters(gdrive_url, download_to_dir):
    os.makedirs(download_to_dir, exist_ok=True)
    zip_filename = "lora_adapters.zip"
    zip_filepath = os.path.join(download_to_dir, zip_filename)

    print(f"Attempting to download LoRA adapters from Google Drive URL: {gdrive_url}")
    print(f"This may take a few moments depending on the file size and network speed.")

    try:
        gdown.download(url=gdrive_url, output=zip_filepath, quiet=False, fuzzy=True)
        print(f"Successfully downloaded to {zip_filepath}")

        print(f"Extracting {zip_filepath} to {download_to_dir}...")
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(download_to_dir)
        print(f"Successfully extracted adapters to {download_to_dir}")

        os.remove(zip_filepath)
        print(f"Removed temporary zip file: {zip_filepath}")

        for root, dirs, files in os.walk(download_to_dir):
            if "adapter_config.json" in files:
                print(f"Found adapter_config.json in: {root}")
                return root

        print(f"Warning: Could not find 'adapter_config.json' after extraction in {download_to_dir} or its subdirectories.")
        return download_to_dir

    except Exception as e:
        print(f"Error during download or extraction from Google Drive: {e}")
        print("Please ensure the Google Drive URL is correct, public, and points to a ZIP file.")
        return None


def load_model_and_processor(base_model_name, lora_adapter_path, device):
    processor = ViltProcessor.from_pretrained(base_model_name)
    base_model = ViltForQuestionAnswering.from_pretrained(base_model_name)
    if lora_adapter_path and os.path.exists(lora_adapter_path) and os.path.exists(os.path.join(lora_adapter_path, "adapter_config.json")):
        print(f"Loading LoRA adapters from local path: {lora_adapter_path}")
        model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        print("Successfully loaded LoRA adapters.")
    else:
        print(f"Warning: LoRA adapter path '{lora_adapter_path}' not found locally or 'adapter_config.json' missing.")
        print("Falling back to using the base model without LoRA adapters.")
        model = base_model

    model.to(device)
    model.eval()
    return model, processor

def main():
    parser = argparse.ArgumentParser(description="VQA Inference Script with LoRA fine-tuned ViLT model.")
    parser.add_argument('--image_dir', type=str, required=True, help='Path to the directory containing images.')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the input CSV file with image_name and question columns.')
    parser.add_argument('--gdrive_lora_url', type=str, help='Google Drive URL for the ZIP file of LoRA adapters.')
    parser.add_argument('--local_adapter_path', type=str, help='Path to a local directory containing LoRA adapters (overrides GDrive download if provided and valid).')
    parser.add_argument('--output_csv_path', type=str, default="results.csv", help='Path to save the output CSV with generated answers.')

    args = parser.parse_args()
    args.gdrive_lora_url = "https://drive.google.com/file/d/1HmTKzVATt5fFaTplcQL5SNE0SnVWOL9h/view?usp=sharingg"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    lora_path_to_use = None

    if args.local_adapter_path and os.path.exists(args.local_adapter_path) and os.path.exists(os.path.join(args.local_adapter_path, "adapter_config.json")):
        print(f"Using provided local adapter path: {args.local_adapter_path}")
        lora_path_to_use = args.local_adapter_path
    elif args.gdrive_lora_url:
        print(f"Attempting to download adapters from Google Drive URL: {args.gdrive_lora_url}")
        if os.path.exists(LOCAL_ADAPTER_DOWNLOAD_DIR):
            print(f"Cleaning up existing download directory: {LOCAL_ADAPTER_DOWNLOAD_DIR}")
            shutil.rmtree(LOCAL_ADAPTER_DOWNLOAD_DIR) # Remove old downloads

        lora_path_to_use = download_and_extract_adapters(args.gdrive_lora_url, LOCAL_ADAPTER_DOWNLOAD_DIR)
        if not lora_path_to_use:
            print("Failed to download or extract adapters from Google Drive. Will attempt to use base model only.")
    else:
        print("No LoRA adapter path or Google Drive URL provided. Will attempt to use base model only.")


    model, processor = load_model_and_processor(BASE_MODEL_NAME, lora_path_to_use, device)

    if hasattr(model, 'base_model'):
        model_config = model.base_model.model.config
    else:
        model_config = model.config

    if not hasattr(model_config, 'id2label') or not model_config.id2label:
        print("Error: model.config.id2label not found or empty. Cannot map prediction indices to answers.")
        print("Attempting to load config from base model name directly.")
        from transformers import AutoConfig
        temp_config = AutoConfig.from_pretrained(BASE_MODEL_NAME)
        if not hasattr(temp_config, 'id2label') or not temp_config.id2label:
             raise ValueError("Could not load id2label from model config. This is required for VQA.")
        model_id2label = temp_config.id2label
    else:
        model_id2label = model_config.id2label

    try:
        df = pd.read_csv(args.csv_path)
        if 'image_name' not in df.columns or 'question' not in df.columns:
            raise ValueError("Input CSV must contain 'image_name' and 'question' columns.")
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {args.csv_path}")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    generated_answers = []
    print(f"Processing {len(df)} samples...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating Answers"):
        image_filename = str(row['image_name'])
        image_path = os.path.join(args.image_dir, image_filename)
        question = str(row['question'])

        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize((TARGET_IMAGE_WIDTH, TARGET_IMAGE_HEIGHT), Image.Resampling.BICUBIC)
            encoding = processor(images=image, text=question, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**encoding)
                logits = outputs.logits
                predicted_idx = logits.argmax(-1).item()
                answer_str = model_id2label[predicted_idx]
        except FileNotFoundError:
            answer_str = "error_image_not_found"
        except Exception as e:
            answer_str = "error_processing_sample"

        processed_answer = str(answer_str).split()[0].lower() if answer_str else "error_empty_answer"
        generated_answers.append(processed_answer)

    df["generated_answer"] = generated_answers
    df.to_csv(args.output_csv_path, index=False)
    print(f"Inference complete. Results saved to {args.output_csv_path}")

if __name__ == "__main__":
    main()