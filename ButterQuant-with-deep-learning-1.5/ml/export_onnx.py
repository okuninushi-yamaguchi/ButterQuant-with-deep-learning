import torch
import torch.nn as nn
import os
import joblib

# Must match the architecture in train_model.py
class SuccessClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SuccessClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def export():
    # 1. Configuration
    model_pth = 'ml/models/success_model.pth'
    scaler_path = 'ml/models/scaler.joblib'
    onnx_path = 'ml/models/success_model.onnx'
    
    if not os.path.exists(model_pth) or not os.path.exists(scaler_path):
        print("Error: Trained model (.pth) or scaler (.joblib) not found.")
        print("Please run 'python ml/train_model.py' first.")
        return

    # 2. Get input dimension from scaler
    scaler = joblib.load(scaler_path)
    input_dim = scaler.n_features_in_
    print(f"Detected input dimension: {input_dim}")

    # 3. Load Model
    model = SuccessClassifier(input_dim)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device('cpu')))
    model.eval()

    # 4. Export to ONNX
    # Create a dummy input (matching expected input shape)
    dummy_input = torch.randn(1, input_dim)
    
    print(f"Exporting model to {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print("✅ Export successful!")
    print("\nTo deploy to backend:")
    print(f"1. Copy {onnx_path} to backend/models/")
    print(f"2. Copy {scaler_path} to backend/models/")

if __name__ == "__main__":
    export()
