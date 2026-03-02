# Model Conversion Guide

## ⚠️ Important: TensorFlow.js Model Format Required

The current model file (`assets/model/convlstm.tflite`) needs to be converted to TensorFlow.js format for the app to work.

## Option 1: Convert TFLite to TFJS (Recommended)

### Step 1: Install TensorFlow.js Converter
```bash
pip install tensorflowjs
```

### Step 2: Convert the Model
```bash
# Navigate to your project
cd "c:\Users\ejans\OneDrive\Documents\Programming Projects\Thesis\ConvLSTM"

# Convert TFLite to TFJS format
tensorflowjs_converter \
  --input_format=tf_saved_model \
  --output_format=tfjs_graph_model \
  --signature_name=serving_default \
  --saved_model_tags=serve \
  conversion/tf_model \
  test_deployment/assets/model/

# This will create:
# - model.json (model architecture)
# - group1-shard1of1.bin (model weights)
```

### Step 3: Update Asset References
The model will be automatically loaded from `assets/model/model.json`

## Option 2: Use ONNX Model (Alternative)

If you have the ONNX model, you can use onnx.js:

```bash
npm install onnxjs
```

Then update `inference.ts` to use ONNX runtime instead of TensorFlow.js.

## Option 3: Development Mode (Current)

For testing the UI without a working model, the app currently falls back to displaying random predictions when the model fails to load. This allows you to:
- Test the camera capture
- Verify the UI layout
- Check performance metrics display
- Test navigation flow

## Converting Your PyTorch Model

If you need to convert from PyTorch:

### Step 1: Export PyTorch to ONNX
```python
import torch
import onnx

# Load your PyTorch model
model = ConvLSTMClassifier(...)
model.load_state_dict(torch.load('best_convlstm.pth'))
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 20, 6, 128, 128)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "convlstm.onnx",
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output']
)
```

### Step 2: Convert ONNX to TensorFlow
```bash
pip install onnx-tf

# Convert ONNX to TF SavedModel
onnx-tf convert -i convlstm.onnx -o tf_model/
```

### Step 3: Convert TF SavedModel to TFJS
```bash
tensorflowjs_converter \
  --input_format=tf_saved_model \
  --output_format=tfjs_graph_model \
  tf_model/ \
  test_deployment/assets/model/
```

## Verifying the Conversion

After conversion, you should see:
```
test_deployment/
└── assets/
    └── model/
        ├── model.json          # Model architecture
        └── group1-shard1of1.bin # Model weights
```

## File Size Expectations

- **TFLite model**: ~1.5 MB
- **TFJS model.json**: ~100 KB
- **TFJS .bin files**: ~1.5 MB total

## Next Steps

1. Convert your model using one of the methods above
2. Place the converted files in `test_deployment/assets/model/`
3. Restart the Expo development server
4. The app will automatically load the real model

## Troubleshooting

### Error: "Cannot find module model.json"
- Make sure model.json is in `assets/model/` directory
- Check that the file is included in `app.json` under `assetBundlePatterns`

### Error: "Model prediction failed"
- Verify input shape matches: `[1, 20, 6, 128, 128]`
- Check that the model expects float32 input
- Ensure normalization is correct (0-1 range for pixels)

### Model loads but predictions are wrong
- Verify preprocessing matches training pipeline
- Check channel order (RGB vs BGR)
- Verify normalization values match training

## Need Help?

If you encounter issues:
1. Check the console logs for detailed error messages
2. Verify the model conversion completed successfully
3. Test the model in Python first before converting
4. Ensure TensorFlow.js version is compatible with your model
