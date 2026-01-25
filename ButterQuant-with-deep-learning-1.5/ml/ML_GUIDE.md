# ButterQuant ML Workflow 🧠

This directory contains the machine learning pipeline for predicting the success probability of butterfly strategies.

## Do I need to keep the computer on 24/7 for training?
**No.** 
Training is an "offline" process. You only need to run the GPU and your computer when you want to **update** the model with new data. 
- **Inference** (The actual AI scoring in the app) uses a pre-trained "brain" (ONNX model) which is very fast and does not require constant training.
- You can re-train once a week or once a month to keep the model sharp.

---

## How to use the Python scripts:

### Step 1: Data Preparation
Daily market data is saved to the database by the scanner. Once you have enough data (e.g., after a week), run:
```bash
python ml/features.py
```
*   **What it does:** Reads the database, downloads historical prices to calculate results, and creates `ml/training_data.parquet`.

### Step 2: Model Training
Use your GPU and the Anaconda environment to train:
```bash
python ml/train_model.py
```
*   **What it does:** Trains the neural network. Saves `ml/models/success_model.pth` and `ml/models/scaler.joblib`.

### Step 3: Deployment (ONNX Export)
The backend requires a high-performance ONNX model:
```bash
python ml/export_onnx.py
```
*   **What it does:** Converts the `.pth` model to `.onnx`.

### Step 4: Final Update to Backend
Copy the new files to the backend directory:
```bash
cp ml/models/success_model.onnx backend/models/
cp ml/models/scaler.joblib backend/models/
```
(Restart your Flask backend to apply the new model).
