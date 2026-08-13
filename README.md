# DRIFT-SENSE: AI-POWERED NAVIGATION-ERROR RECOVERY FOR WAFER INSPECTION TOOLS
##Hackathon: Applied Materials Problem Statement (SEMICON India 2026)

## Objective: 
Automatically locate a high-resolution (100x) reference pattern inside a larger, noisier low-resolution (10x) search image to correct wafer inspection stage drift.

## 🧠 About the Solution
This repository contains a complete, end-to-end deep learning pipeline to solve the navigation-error recovery problem. Instead of relying on classical feature matching (which struggles with heavy SEM noise and repetitive structures), this solution uses a Deep Siamese Convolutional Neural Network (CNN) combined with Normalized Cross-Correlation (NCC).

## Key Features:

Scale & Rotation Robustness: The model is trained on dynamically generated synthetic data with scales ranging from 9:1 to 11:1 and random rotations of ±2°, preventing overfitting to a fixed 10x scale.

### Siamese Architecture: 
A custom CNN backbone extracts deep features from both images and projects them into a shared embedding space.

### Tie-Breaker Logic:
If the highly repetitive pattern (e.g., DRAM grids, FinFETs) yields multiple valid matches, the inference engine automatically selects the match closest to the absolute center of the search image, exactly as required.

Coordinate System: Adheres strictly to the standard image coordinate system: Origin (0, 0) is at the top-left, with X increasing to the right and Y increasing downward.

# 📂 Repository Structure
The folder structure strictly follows the recommended hackathon guidelines:

Plaintext
```
submission/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── solution_presentation.pptx          # 12-slide mandatory presentation
├── generate_dataset.py                 # Synthetic SEM data generator
├── localize.py                         # Evaluation and inference script
├── references/                         # Public sources and citations
├── results/                            # Outputs: CSV manifests, JSON summaries, overlays
├── model/                              # Contains the trained PyTorch checkpoint
│   └── drift_sense_model.pth           
└── src/                                # Source code for training
    └── dl_training.py
```
# ⚙️ Environment Setup & Hardware
This code is written in Python and uses PyTorch. It is optimized to run dynamically on CUDA (NVIDIA GPUs), MPS (Apple Silicon / M-Series Chips), or standard CPUs.

### (Note: The primary model was trained and evaluated on an Apple MacBook Air M4 utilizing MPS hardware acceleration, achieving ~12ms inference latency).

## Create a virtual environment (optional but recommended):

```
python -m venv drift_env
source drift_env/bin/activate  # On Windows use: drift_env\Scripts\activate
```
Install dependencies:
```
pip install -r requirements.txt
```
### ⚖️ Evaluation Instructions (For Judges)
This repository is designed for easy evaluation. You can test custom data in three different ways depending on your preferred format.

1. Batch Folder Mode (Easiest)
If you have a folder of images without a CSV file, use this mode. The script will automatically pair files containing ref and search in their names.

```
python localize.py --batch-dir /path/to/your/custom_folder/ --weights model/drift_sense_model.pth
```
### 2. Standard Dataset Evaluation (Uses CSV)
To run the standard evaluation using a CSV manifest, use the --data-dir argument. (Note: The target folder must contain the images and a labels.csv file formatted identically to the provided sample data).
```
python localize.py --data-dir /path/to/your/custom_dataset --split val --weights model/drift_sense_model.pth
```
(Required Folder Structure for this mode):
```
Plaintext
custom_dataset/
└── val/
    ├── labels.csv       <-- Must be named labels.csv
    ├── ref_1.png
    └── search_1.png
```
3. Quick Test (Single Image Pair)
Test the drift matching on a single pair of images via the command line:
```
python localize.py --ref-img /path/to/reference.png --search-img /path/to/search.png --weights model/drift_sense_model.pth
```
## 🚀 Reproduction Steps (From Scratch)
Step 1: Generate the Synthetic Dataset
Run the data generator to create realistic grayscale SEM image pairs (DRAM, FinFET, and Via Arrays) with noise, shading, and dynamic scale/rotation augmentations.

```
python generate_dataset.py --train-samples 5000 --val-samples 200 --out-dir dataset
```
Output: Creates a dataset/ folder containing train/ and val/ subdirectories, along with the ground-truth labels.csv manifests and meta.json.

Step 2: Train the Model (Optional)
If you wish to train the model from scratch instead of using the provided weights:

```
python src/dl_training.py
```
Output: Trains the Siamese network for 45 epochs and saves the best weights.

Step 3: Evaluate on the Validation Set
Run the automated evaluation harness over the generated dataset to calculate the threshold metrics (5px, 4px, 2px, 1px pass rates):

```
python localize.py --data-dir dataset --split val --weights model/drift_sense_model.pth --out-dir results
```
Output: Generates predictions_manifest.csv, eval_summary.json, and an annotated image of the worst-performing pair inside the results/ folder.

## ⚠️ Assumptions & Limitations
Input Format: The model expects 1000×1000 grayscale images.

Nominal Scale: While the model is robust to 9x-11x variations, the inference pipeline actively resizes the reference image by a hardcoded nominal factor of 10.0 to force scale-invariant feature matching.

Device Fallback: If a dedicated GPU (CUDA/MPS) is not detected, the code will seamlessly fall back to CPU execution, which will naturally increase latency.

Key changes made:
Added Markdown code blocks (```bash) so the terminal commands render correctly on GitHub.

Added the missing --batch-dir command to the judges' section.

Ensured --weights model/drift_sense_model.pth is explicitly included in the commands so it doesn't fail if the user runs it from the root directory.

Separated the "For Judges" section entirely from the "Reproduction Steps" so a judge doesn't have to read through the dataset generation instructions just to figure out how to evaluate the model.
