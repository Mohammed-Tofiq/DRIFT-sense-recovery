# Drift-Sense: AI-Powered Navigation-Error Recovery
## Hackathon: Applied Materials Problem Statement (SEMICON India 2026)

### Objective: Automatically locate a high-resolution (100x) reference pattern inside a larger, noisier low-resolution (10x) search image to correct wafer inspection stage drift.

## 🧠 About the Solution
This repository contains a complete, end-to-end deep learning pipeline to solve the navigation-error recovery problem.

Instead of relying on classical feature matching (which struggles with heavy SEM noise and repetitive structures), this solution uses a Deep Siamese Convolutional Neural Network (CNN) combined with Normalized Cross-Correlation (NCC).

## Key Features:
Scale & Rotation Robustness: The model is trained on dynamically generated synthetic data with scales ranging from 9:1 to 11:1 and random rotations of ±2°, preventing overfitting to a fixed 10x scale.

## Siamese Architecture: 
A custom CNN backbone extracts deep features from both images and projects them into a shared embedding space.

## Tie-Breaker Logic: If the highly repetitive pattern (e.g., DRAM grids, FinFETs) yields multiple valid matches, the inference engine automatically selects the match closest to the absolute center of the search image, exactly as required.

##Coordinate System: Adheres strictly to the standard image coordinate system: Origin (0, 0) is at the top-left, with X increasing to the right and Y increasing downward.

# 📂 Repository Structure
The folder structure strictly follows the recommended hackathon guidelines:
```text
Plaintext
submission/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── solution_presentation.pptx          # 12-slide mandatory presentation
├── generate_dataset.py                 # Synthetic SEM data generator
├── localize.py                         # Evaluation and inference script
├── references/                         # Public sources and citations
├── results/                            # Outputs: CSV manifests, JSON summaries, overlay images
├── model/                              # Contains the trained PyTorch checkpoint
│   └── drift_sense_model.pth           
└── src/                                # Source code for training
    └── dl_training.py
```
# ⚙️ Environment Setup & Hardware
This code is written in Python and uses PyTorch. It is optimized to run dynamically on CUDA (NVIDIA GPUs), MPS (Apple Silicon / M-Series Chips), or standard CPUs.

(Note: The primary model was trained and evaluated on an Apple MacBook Air M4 utilizing MPS hardware acceleration, achieving ~12ms inference latency).

1. Create a virtual environment (optional but recommended):
```text
python -m venv drift_env
source drift_env/bin/activate  # On Windows use: drift_env\Scripts\activate
```
2. Install dependencies:
```text
pip install -r requirements.txt
```

# 🚀 How to Run the Code (Step-by-Step)
### Step 1: Generate the Synthetic Dataset
Run the data generator to create realistic grayscale SEM image pairs (DRAM, FinFET, and Via Arrays) with noise, shading, and dynamic scale/rotation augmentations.
```text
python generate_dataset.py --train-samples 5000 --val-samples 200 --out-dir dataset
```
Output: Creates a dataset/ folder containing train/ and val/ subdirectories, along with the ground-truth labels.csv manifests and meta.json.

### Step 2: Train the Model (Optional)
If you wish to train the model from scratch instead of using the provided weights in the model/ folder:
```text
python src/dl_training.py
```
Output: Trains the Siamese network for 45 epochs and saves the best weights to drift_sense_model.pth.

### Step 3: Evaluate on the Validation Set
To run the automated evaluation harness over the generated dataset and calculate the threshold metrics (5px, 4px, 2px, 1px pass rates):
```text
python localize.py --data-dir dataset --split val --weights model/drift_sense_model.pth --out-dir results
```
Output: Generates predictions_manifest.csv, eval_summary.json, and an annotated image of the worst-performing pair inside the results/ folder.

### Step 4: Real-World Inference (Single Image Pair)
To test the model on a single, real-world image pair without needing a CSV manifest or dataset directory:
```text
python localize.py --ref-img path/to/reference.png --search-img path/to/search.png --weights model/drift_sense_model.pth
```
Output: Prints the exact predicted (x, y) sub-pixel coordinates of the target center in the terminal.

# ⚠️ Assumptions & LimitationsInput Format:
The model expects $1000 \times 1000$ grayscale images.Nominal Scale: While the model is robust to 9x-11x variations, the inference pipeline actively resizes the reference image by a hardcoded nominal factor of 10.0 to force scale-invariant feature matching.Device Fallback: If a dedicated GPU (CUDA/MPS) is not detected, the code will seamlessly fall back to CPU execution, which will increase latency.



