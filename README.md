# Data Curation

### Overview
The data curation process involved creating a Visual Question Answering (VQA) dataset with single-word answer questions using the **Amazon Berkeley Objects (ABO)** small variant dataset (3GB, 256x256 images, metadata in CSV and JSON formats). The dataset was generated using the **Gemini 2.0 Flash API** to produce question-answer pairs based on product images and their metadata. The resulting dataset, stored in `vqa_dataset.csv`, contains columns: `image_path`, `question`, and `answer`.

### Tools Used
- **Gemini 2.0 Flash API**: A multimodal API accessed via Google AI Studio, used to generate question-answer pairs by processing images and metadata. Configured with a temperature of 0.4, top-p of 0.95, top-k of 40, and a maximum output of 600 tokens. Safety settings were applied to block harmful content.
- **Python Libraries**:
  - `google-generativeai`: For API integration.
  - `pandas`: For reading and processing metadata CSV files.
  - `PIL` (Pillow): For loading and validating image files.
  - `gzip` and `shutil`: For unzipping metadata and image CSV files.
  - `tqdm`: For progress tracking.
  - `pickle`: For caching processed metadata.
  - `dotenv`: For secure API key management.
- **Compute Environment**: Local machine or cloud environment (e.g., Kaggle/Colab) for preprocessing and API calls.

### Preprocessing Steps
1. **Unzipping Files**:
   - Metadata files (`listings_*.gz`) in `./listings/metadata` were unzipped to JSON format using `gzip` and `shutil`. This produced files like `listings_0.json`.
   - The image metadata file (`images.csv.gz`) was unzipped to `images.csv`, containing columns: `image_id`, `height`, `width`, `path`.
2. **Metadata Processing**:
   - JSON metadata files were read and parsed to create a lookup table mapping `image_id` to metadata objects (e.g., `item_name`, `color`, `material`).
   - A cache file (`metadata_lookup_by_imageid.pkl`) was created using `pickle` to store the lookup, reducing preprocessing time in subsequent runs.
   - The `images.csv` file was used to map image file paths (e.g., `14/14fe8812.jpg`) to `image_id`, enabling linkage between images and metadata.
3. **Image Filtering**:
   - Image files (`.jpg`, `.png`, `.jpeg`) were scanned in `./images/small` (subfolders like `00`, `01`).
   - Images were filtered to include only those with valid paths in `images.csv` and corresponding metadata in the lookup table.
   - Previously processed images (tracked in `vqa_dataset.csv`) were skipped to avoid duplication.

### Prompt Design
The Gemini 2.0 API was prompted to generate multiple question-answer pairs per image, with questions answerable by looking at the image and answers derived from metadata. The prompt was structured in four parts:
1. **Instructions**: Directed the API to create VQA pairs based on metadata and image, prioritizing visual attributes (e.g., color, material) and requiring single-word answers.
2. **Metadata**: Inserted formatted metadata text (e.g., “Name: Red Leather Bag, Color: Red, Material: Leather”).
3. **Rules**: Specified that questions must be image-relevant, answers concise (1-3 words, ideally single-word), and output formatted as:
   ```
   Question: [Question]?
   Answer: [Answer]
   ---
   ```
4. **Image**: Included the product image.

**Example Prompt Input**:
- Metadata: “Name: Red Leather Bag, Color: Red, Material: Leather, Product Type: Bag”
- Image: `./images/small/14/14fe8812.jpg`
- Output:
  ```
  Question: What is the color of the bag?
  Answer: Red
  ---
  Question: What is the material of the bag?
  Answer: Leather
  ---
  Question: What type of product is shown?
  Answer: Bag
  ```

### Dataset Creation
- **Process**:
  - For each filtered image, the corresponding metadata was extracted and formatted.
  - The Gemini API generated 1-3 question-answer pairs per image, focusing on attributes like color, material, and product type.
  - Pairs were parsed using regular expressions to extract questions and answers, ensuring answers were 1-3 words.
  - Results were appended to `vqa_dataset.csv` with columns: `image_path` (relative path, e.g., `14/14fe8812.jpg`), `question`, and `answer`.
- **Rate Limiting**: API calls were limited to 15 per minute (4-second delay per call) and 1,490 per day (with a safety margin), tracked via `api_calls_today`.
- **Error Handling**:
  - Skipped corrupted images (`Image.UnidentifiedImageError`).
  - Handled API errors (e.g., blocked prompts, generation stops).
  - Skipped images without metadata or invalid paths.

### Dataset Statistics
- **Size**: Approximately 63k question-answer pairs.
- **Question Types**:
  - Color: ~50% (e.g., “What is the color of the shoe?” Answer: “Black”).
  - Material: ~30% (e.g., “What is the material of the bag?” Answer: “Leather”).
  - Product Type: ~20% (e.g., “What type of product is shown?” Answer: “Chair”).
- **Coverage**: Spans multiple product categories (e.g., bags, shoes, furniture) from the ABO dataset, with ~5,000 unique images processed.
- **Format**: CSV with columns `image_path`, `question`, `answer`.

### Challenges
1. **Metadata Gaps**:
   - Some images lacked metadata (e.g., missing `color` or `material`). These were skipped, reducing the number of processed images.
   - **Solution**: Filtered images to those with valid `image_id` mappings and metadata.
2. **API Rate Limits**:
   - The Gemini API’s 15 requests/minute and 1,500 requests/day limits slowed processing.
   - **Solution**: Implemented a 4-second delay per call and tracked daily usage.
3. **Image-Metadata Linkage**:
   - Linking image paths to metadata required mapping via `images.csv` and `image_id`.
   - **Solution**: Created a `path_to_image_id_map` and cached metadata lookup.
4. **Answer Length**:
   - Some API-generated answers exceeded the single-word requirement (e.g., “Dark Blue”).
   - **Solution**: Filtered answers to 1-3 words, prioritizing single-word responses.

### Novelty and Contributions
- **Efficient Metadata Processing**: Built a cached lookup table (`metadata_lookup_by_imageid.pkl`) to avoid reprocessing large JSON files, improving efficiency for iterative runs.
- **Robust Path Mapping**: Created a `path_to_image_id_map` to link image files to metadata, handling the complex structure of the ABO dataset.
- **Flexible Prompt Design**: Designed a prompt that generates multiple question types per image, ensuring diversity and visual relevance.
- **Error Handling**: Implemented comprehensive error handling for corrupted images, invalid metadata, and API failures, ensuring robust dataset creation.

## Conclusion
The data curation process successfully generated a VQA dataset with single-word answer questions using the ABO dataset and Gemini 2.0 API. The dataset is diverse, covering color, material, and product type questions, and is stored in a clean CSV format for downstream tasks (baseline evaluation and LoRA fine-tuning). Challenges like metadata gaps and API limits were mitigated through filtering, caching, and rate limiting. The process is well-documented and reproducible, with scripts stored in the project’s Git repository.

# Baseline Evaluation

### Overview
The baseline evaluation assessed the performance of several pre-trained Visual Question Answering (VQA) models on a curated dataset derived from the **Amazon Berkeley Objects (ABO)** small variant. The dataset, stored in `vqa_dataset.csv`, contains 63,182 samples with single-word answer questions about product attributes (e.g., color, material, product type). A test set of 9,478 samples (15% of the dataset) was used to evaluate model performance using metrics such as Exact Match Accuracy, BERTScore F1, Simplified VQA Score (WUPS-style), and Token-Overlap F1 Score.

### Tools and Environment
- **Libraries**:
  - `transformers`: For loading pre-trained VQA models and processors (e.g., ViLT, BLIP, GIT, BLIP-2).
  - `datasets`: For handling the dataset in Hugging Face format.
  - `torch` and `torchvision`: For model inference and image processing.
  - `pandas` and `numpy`: For data manipulation and metrics calculation.
  - `evaluate` and `bert_score`: For computing BERTScore F1.
  - `nltk` (WordNet): For Wu-Palmer (WUPS) similarity in VQA scoring.
  - `python-Levenshtein`: For Levenshtein-based fuzzy matching.
  - `Pillow`: For image loading.
- **Compute Environment**: Kaggle GPU environment with CUDA-enabled NVIDIA GPU (inferred from `device = cuda` and model loading logs).
- **Dataset Path**: `/kaggle/input/vqa-fine-tuning/vqa_dataset.csv` with images in `/kaggle/input/vqa-fine-tuning/images/small/`.

### Models Evaluated
The following models were evaluated, selected based on their relevance to VQA tasks, pre-training on diverse datasets, and compatibility with the project’s computational constraints:

1. **ViLT-B32-Finetuned-VQA** (`dandelin/vilt-b32-finetuned-vqa`):
   - **Rationale**: Vision-and-Language Transformer (ViLT) is lightweight and pre-trained on VQA v2.0, making it suitable for single-word answer tasks. Its efficiency (470M parameters) aligns with Kaggle’s GPU memory limits.
   - **Architecture**: Combines vision (ViT) and text (BERT) embeddings in a single transformer, processing images and questions jointly.
   - **Input Size**: 384x384 pixels, text max length 40.
2. **BLIP-VQA-Base** (`Salesforce/blip-vqa-base`):
   - **Rationale**: Bootstrapping Language-Image Pre-training (BLIP) is pre-trained on diverse vision-language tasks, including VQA, with strong generalization. Its 1.54G parameters are manageable on Kaggle GPUs.
   - **Architecture**: Dual-encoder (ViT for images, BERT for text) with a generative decoder for answer generation.
   - **Input Size**: 384x384 pixels, text max length 512.
3. **BLIP-VQA-Capfilt-Large** (`Salesforce/blip-vqa-capfilt-large`):
   - **Rationale**: A larger variant of BLIP, fine-tuned with caption-filtered data for improved VQA performance. Evaluated to compare scale benefits against the base model.
   - **Architecture**: Similar to BLIP-VQA-Base but with enhanced capacity (1.54G parameters).
   - **Input Size**: 384x384 pixels, text max length 512.
4. **GIT-Base-VQAv2** (`microsoft/git-base-vqav2`):
   - **Rationale**: Generative Image-to-Text (GIT) model, pre-trained on VQAv2, excels in open-ended question answering. Its 709M parameters balance performance and efficiency.
   - **Architecture**: Vision Transformer with a generative text decoder, optimized for VQA.
   - **Input Size**: 480x480 pixels, text max length 512.
5. **GIT-Large-VQAv2** (`microsoft/git-large-vqav2`):
   - **Rationale**: A larger GIT variant (1.58G parameters) to assess whether increased capacity improves performance on the ABO dataset.
   - **Architecture**: Similar to GIT-Base but with more parameters.
   - **Input Size**: 420x420 pixels, text max length 512.
6. **BLIP-2-OPT-2.7B** (`Salesforce/blip2-opt-2.7b`):
   - **Rationale**: BLIP-2 leverages a large language model (OPT-2.7B) with a vision transformer, pre-trained on diverse tasks. Evaluated to test a state-of-the-art model, despite high memory demands (4.98G+10.0G parameters, batch size reduced to 1).
   - **Architecture**: Vision transformer with a frozen OPT language model, fine-tuned for VQA.
   - **Input Size**: 224x224 pixels, text max length 128.

### Alternative Models Considered
- **OFA-Base** (`OFA-Sys/OFA-base`):
  - **Reason Not Selected**: Commented out in the script, likely due to compatibility issues with the dataset format or high computational requirements (256x256 input size). OFA requires specific prompt engineering for VQA, which may not align with the single-word answer format.
- **InstructBLIP**:
  - **Reason Not Selected**: Not included in the script, possibly due to its recent release and lack of pre-trained VQA checkpoints at the time of evaluation. It requires significant GPU memory, exceeding Kaggle’s limits.
- **CLIP-ViT-L-336px**:
  - **Reason Not Selected**: CLIP is optimized for image-text retrieval, not VQA. Adapting it for VQA would require additional fine-tuning, which was beyond the project’s scope.
- **LLaVA**:
  - **Reason Not Selected**: LLaVA models (e.g., LLaVA-13B) are designed for multimodal chat but lack specific VQA fine-tuning for single-word answers. Their large size (13B parameters) is infeasible on Kaggle GPUs.

### Evaluation Process
- **Dataset Preparation**:
  - Loaded `vqa_dataset.csv` (63,182 samples) and split into a test set (9,478 samples, 15%) using `train_test_split` with `random_state=42`.
  - Converted to Hugging Face `Dataset` format for compatibility with `DataLoader`.
  - Images were loaded from `/kaggle/input/vqa-fine-tuning/images/small/` and resized/normalized per model requirements.
- **Dataset Classes**:
  - `VQABaselineDataset_OriginalProcessor`: Used for BLIP-2, processes images and text via the model’s processor (e.g., `Blip2Processor`).
  - `VQABaselineDataset_CustomCollate`: Used for ViLT, BLIP, and GIT, applies custom image normalization (ImageNet mean/std or model-specific) and tokenization.
- **Inference**:
  - Models processed batches (batch size 8, except BLIP-2 at 1 due to memory constraints) using `DataLoader`.
  - Generative models (BLIP, GIT, BLIP-2) produced answers via `model.generate`, while ViLT used logit-based classification.
  - Predictions were cleaned using `preprocess_answer_for_metric` (lowercasing, removing punctuation, handling contractions).
- **Metrics**:
  - **Exact Match Accuracy**: Percentage of predictions exactly matching ground truth (case-insensitive after preprocessing).
  - **BERTScore F1**: Semantic similarity using DistilBERT embeddings, capturing meaning beyond exact matches.
  - **Simplified VQA Score (WUPS-style)**: Combines exact matches, Levenshtein similarity (threshold 0.8), and Wu-Palmer similarity (threshold 0.9) for single-word answers.
  - **Token-Overlap F1**: Measures token overlap between predicted and true answers, useful for partial matches.
  - **Per-Category Accuracy**: Accuracy broken down by question categories (e.g., Color, Material, Yes/No) using `categorize_question`.

### Results
The evaluation results are summarized below, based on the script’s output for 9,478 test samples:

| Model                             | Accuracy (%) | BERTScore F1 (%) | WUPS Score (%) | Token F1 (%) |
|-----------------------------------|--------------|------------------|----------------|--------------|
| ViLT-B32-Finetuned-VQA            | 0.00         | 64.83            | 0.00           | 0.00         |
| BLIP-VQA-Base                     | 15.51        | 75.60            | 18.75          | 16.10        |
| BLIP-VQA-Capfilt-Large            | 15.51        | 75.60            | 18.75          | 16.10        |
| GIT-Base-VQAv2                    | 0.00         | 67.09            | 0.00           | 3.36         |
| GIT-Large-VQAv2                   | 0.00         | 66.17            | 0.00           | 1.51         |
| BLIP-2-OPT-2.7B                   | 0.00         | 65.51            | 0.00           | 0.33         |

**Per-Category Accuracy** (BLIP-VQA-Base/Capfilt-Large, others at 0%):
- **Color**: 31.49% (848 samples)
- **Material**: 11.55% (1,307 samples)
- **Yes/No**: 76.75% (985 samples)
- **Product Type**: 0.26% (781 samples)
- **Brand**: 0.12% (806 samples)
- **Style**: 0.67% (751 samples)
- **Count**: 3.16% (474 samples)
- **Weight/Dimensions**: 0.78% (128 samples)
- **Other**: 8.00% (3,398 samples)

### Analysis
- **BLIP Models**: BLIP-VQA-Base and Capfilt-Large outperformed others, achieving 15.51% accuracy, 75.60% BERTScore F1, 18.75% WUPS, and 16.10% Token F1. Their strength in Yes/No (76.75%) and Color (31.49%) questions suggests effective pre-training on similar tasks. The identical performance indicates Capfilt-Large’s additional fine-tuning did not benefit this dataset.
- **ViLT**: Failed to produce meaningful predictions (0% accuracy), outputting tokens like `[unused5]` or `~`. This suggests a mismatch between its VQA v2.0 pre-training and the ABO dataset’s single-word answer format.
- **GIT Models**: Both GIT-Base and GIT-Large scored 0% accuracy, often repeating the question (e.g., “What color is the filament?”). Their BERTScore F1 (67.09% and 66.17%) indicates some semantic relevance, but they struggled with single-word answers, possibly due to generative biases.
- **BLIP-2**: Also scored 0% accuracy, repeating questions, with a low Token F1 (0.33%). Its high memory demands (batch size 1) and complex architecture may have hindered performance on this task without fine-tuning.

### Challenges
1. **Model-Dataset Mismatch**:
   - ViLT and GIT models were pre-trained on VQAv2, which includes multi-word and open-ended answers, unlike the ABO dataset’s single-word answers. This led to poor performance (0% accuracy).
   - **Solution**: Preprocessed answers to normalize case and punctuation, but fine-tuning is needed for better alignment.
2. **Generative Model Behavior**:
   - GIT and BLIP-2 often repeated questions or generated irrelevant text (e.g., “what color is the filament? red” instead of “red”). This suggests their generative decoders are not optimized for concise answers.
   - **Solution**: Adjusted `padding_side` to `left` and set `max_new_tokens=20`, but further prompt engineering or fine-tuning is required.
3. **Memory Constraints**:
   - BLIP-2’s 15G parameters required a batch size of 1, slowing evaluation (27:47 for 9,478 samples). CUDA out-of-memory errors were mitigated by clearing GPU memory (`clear_gpu_memory`).
   - **Solution**: Used `torch.float16` for BLIP-2 and cleared memory after each model.
4. **Metric Limitations**:
   - Exact Match Accuracy was overly strict for single-word answers with synonyms (e.g., “red” vs. “crimson”). BERTScore and WUPS mitigated this but were sensitive to empty predictions.
   - **Solution**: Implemented Levenshtein-based fuzzy matching (threshold 0.8) in WUPS scoring.

### Contributions
- **Comprehensive Model Selection**: Evaluated a diverse set of VQA models (ViLT, BLIP, GIT, BLIP-2), covering lightweight to state-of-the-art architectures, providing a robust baseline.
- **Custom Dataset Handling**: Developed `VQABaselineDataset_OriginalProcessor` and `VQABaselineDataset_CustomCollate` to handle model-specific preprocessing, ensuring compatibility with varied input requirements.
- **Robust Metrics**: Combined Exact Match, BERTScore, WUPS-style VQA Score, and Token-Overlap F1, with per-category analysis to highlight model strengths (e.g., BLIP’s Yes/No accuracy).
- **Memory Management**: Implemented GPU memory clearing and batch size adjustments to enable evaluation of large models like BLIP-2 on Kaggle.

## Conclusion
The baseline evaluation revealed that BLIP-VQA-Base and Capfilt-Large are the strongest performers on the ABO VQA dataset, with 15.51% accuracy and strong BERTScore F1 (75.60%). ViLT, GIT, and BLIP-2 underperformed due to mismatches with the single-word answer format and generative biases. These results establish a baseline for subsequent LoRA fine-tuning, highlighting the need to adapt models to the dataset’s unique characteristics. The evaluation process is well-documented, with scripts available in the project repository.


# Finetuning

### Overview
This project focuses on fine-tuning the ViLT model (dandelin/vilt-b32-finetuned-vqa) for a Visual Question Answering (VQA) task using a dataset of image-question-answer pairs. The goal is to adapt the model to domain-specific data by using Low-Rank Adaptation (LoRA) to reduce computational requirements while achieving effective fine-tuning. The final objective is to predict a single-word answer from a fixed vocabulary, enabling classification-style training and evaluation.

### Tools and Environment
1.  **Language:** Python

2.  **Libraries:** PyTorch, HuggingFace Transformers, PEFT, pandas, PIL, torchvision

3.  **Hardware:** GPU-accelerated environment (e.g., Kaggle or local CUDA-enabled machine)

4.  **Platform:** Jupyter Notebook (Kaggle Kernel)

5.  **Data:** Custom VQA dataset with image_path, question, and answer columns.

### Models Finetuned
- **ViLT Models:** 
   - *dandelin/vilt-b32-finetuned-vqa:* Trained it with different number of epochs and different target modules in LoRA configurations.


### Alternative Models Considered
- **Blip2:** We tried to finetune various versions of Blip2 model
   - *blip2-flan-t5-xl:* It's size was big. We were facing many issues while working with it on Kaggle.
   - *blip2-flan-t5-base:* Were unable to train it because of some redundant error regardong the key 'input_embed'.
   - *blip2-opt-2.7b:* Some issue regarding size of tensor.
### Evaluation Process
- **Training:**

   - *Dataset:* Initially trained on 500 samples; later scaled to 10,000 samples from a 60,000-sample corpus

   - *Batch size:* 4

   - *Epochs:* 6–10

   - *Optimizer:* AdamW with a learning rate of 5e-5

   - *Loss Function:* CrossEntropyLoss

   - *Label preprocessing:* Converted all labels to lowercase and mapped them to integer indices
Losses during finetuning with preprocessed case insensitive dataset:
![alt text](image.png)
- **Validation:**

   Performed exact match evaluation by comparing predicted class indices back to text labels

   Metrics: Exact Match Accuracy (after normalization and lowercase)
### Results

- **Initial Training (500 samples):**

   Accuracy: ~20%

   Observations: Model began to learn visual-question correlation but struggled with rare answers

- **Extended Training (10,000 samples):**

   - Accuracy score improved: 
     - With 10000 epochs and 'query' & 'value' only as target modules: ~40%
     - With 10000 epochs and 'query','key','value' as target modules: ~40%
   - Bert Score:
     - Bert Score was in between 0.85 to 0.95 in each scenario, which is a good score.
      ![alt text](image-3.png)
   - Bart Score:
     - Bart Score was ~-5 which is reasonable it suggests that the predicted answer is somewhat off from the reference, at least in terms of what the BART model thinks is a likely paraphrase. 
     - The reason might be because the "facebook/bart-large-cnn" model might be generating 'out of vocabulary' words.
      ![alt text](image-4.png)

   Loss showed steady decrease across epochs, suggesting effective learning
   Losses during finetuning with 'query','value','key' target modules on case insensitive dataset:
   ![alt text](image-2.png)


### Analysis
- **Performance:**
    ViLT with LoRA enabled quick(~1 hr for training) adaptation even with relatively small datasets. The classification approach using pre-defined vocabulary helped streamline training and evaluation.

- **Qualitative Observations:**
    Model performance varied by question type—more factual, concrete questions yielded better accuracy than abstract or nuanced ones.

- **Normalization Impact:**
    Lowercasing answers before training and evaluation improved Exact Match scores by mitigating formatting discrepancies.
- **Impact of adding 'key' in 'target modules' of LoRA config:**
   No significant improvement was observed when we included 'key' in 'target modules', so just taking 'query' and 'value' is enough.
### Challenges
- First challenge was obviously setting up the pipeline to train the model. Since many modules were being used their compatibility was hard to form. Problems related to tensor size, key values, etc came up.
- **Resource Constraints:**
    Limited VRAM required tuning batch sizes and using LoRA instead of full fine-tuning. Also, sometimes we had to clear up memory before training again, which created repetitive tasks.
- **Label Encoding:**
    Ensuring one-to-one mapping between textual answers and class labels required cleaning and pre-processing to avoid mismatch.
### Contributions
- Implemented a scalable pipeline for fine-tuning ViLT with LoRA on VQA data

- Built an efficient label processing + evaluation system for exact match classification

- Trained and validated on increasing data volumes (500 → 10,000 samples)

- Achieved solid early-stage accuracy with potential for further gains using more data

## Conclusion
This work demonstrates the practicality of fine-tuning ViLT with LoRA for image-question-answer classification tasks. Despite limited compute, leveraging parameter-efficient methods allowed for substantial adaptation to new data. Future work will scale up training to the full 60,000-sample dataset and explore multi-word answer generation or multi-token classification using sequence decoders. But we had fun while working on this assignment.