# EluSEEdate - Mobile App

React Native/Expo mobile application for real-time turn direction prediction using the ConvLSTM model.

## Overview

This app uses a ConvLSTM (Convolutional Long Short-Term Memory) model to predict vehicle turn directions in real-time using the device camera. The model outputs one of three directions: **Front**, **Left**, or **Right**.

**Based on**: Prototype 10 (with ONNX export support)

## Target Device

**Redmi Note 13 Pro 5G**
- Screen: 1080 x 2400 pixels (portrait)
- Camera: 200MP main (rear)
- Expected inference time: ~100ms

## Design

Minimalistic black & white palette for a clean, distraction-free interface.

## Features

- **Main Menu**: Simple start button
- **Live Camera**: Real-time camera preview with automatic prediction
- **Direction Label**: Large direction indicator at the bottom
- **Performance Overlay**: Inference time and latency metrics at top-left (in ms)
- **Auto-Prediction**: Sliding window inference - captures at 20 FPS, gives new prediction every 50ms after initial 1-second buffer
- **Fast Response**: First prediction after 1 second (20 frames collected), then continuous predictions using sliding window

## Project Structure

```
test_deployment/
├── App.tsx                          # Main entry point
├── package.json                     # Dependencies
├── app.json                         # Expo configuration
├── tsconfig.json                    # TypeScript config
├── babel.config.js                  # Babel config
├── assets/
│   └── model/
│       └── convlstm.tflite          # TFLite model file
└── src/
    ├── config/
    │   └── modelConfig.ts           # Model & device configuration
    ├── navigation/
    │   └── types.ts                 # Navigation type definitions
    ├── screens/
    │   ├── MainMenuScreen.tsx       # Main menu with Start button
    │   └── CameraScreen.tsx         # Camera with inference
    └── services/
        ├── preprocessor.ts          # Frame preprocessing (TypeScript)
        └── inference.ts             # TFLite model inference
```

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Input Shape | [1, 20, 6, 128, 128] |
| Sequence Length | 20 frames |
| FPS | 20 frames/second |
| Duration | 1 second |
| Channels | 6 (3 RGB + 3 Intent) |
| Frame Size | 128 x 128 |
| Output Classes | 3 (Front, Left, Right) |
| Model Size | ~1.5 MB |
| Prediction Interval | New prediction every 50ms (sliding window) |

## Intent Channels

The model expects 6 channels per frame:
- Channels 0-2: RGB color channels [0, 1]
- Channels 3-5: Intent channels (all zeros for 'no intent' mode)

In this app, we always use "no intent" (all zeros for intent channels).

## Getting Started

### Prerequisites

- Node.js 18+
- Expo CLI
- Android Studio (for Android development)

### Installation

```bash
# Navigate to project directory
cd test_deployment

# Install dependencies
npm install

# Start Expo development server
npx expo start

# Run on Android device/emulator
npx expo run:android
```

### Building for Production

```bash
# Create production build using EAS Build (recommended)
npx eas build --platform android --profile preview

# For production release
npx eas build --platform android --profile production
```

## Usage

1. Launch the app
2. Tap the **Start** button on the main menu
3. Point the rear camera at the road ahead
4. The app will automatically:
   - Capture frames at 20 FPS
   - Buffer 20 frames (1 second of video)
   - Run first prediction after 1 second
   - Continue with sliding window predictions every 50ms
   - Display the predicted direction at the bottom
   - Show performance metrics at the top-left

## Performance Metrics

The app displays the following metrics in the top-left corner:

- **Inference**: Time taken by the TFLite model (in ms)
- **Preprocess**: Time taken to prepare frames (in ms)
- **Total**: Combined latency (in ms)
- **FPS**: Effective frames per second

## Development Notes

### TFLite Integration

The current implementation includes a mock inference function for development. To use the actual TFLite model:

1. Install react-native-tflite or equivalent
2. Update `src/services/inference.ts` to use the actual TFLite interpreter
3. Ensure the model file is correctly bundled in assets

### Frame Capture & Sliding Window

The CameraScreen uses `expo-camera` with a sliding window approach:
- **Continuous capture**: 20 FPS (50ms intervals) for real-time processing
- **Buffer management**: Maintains rolling buffer of last 20 frames
- **First prediction**: Uses frame padding if needed (faster initial response)
- **Subsequent predictions**: Sliding window - uses frames [n-19 to n] for each new frame n
- **Prediction frequency**: New prediction every 50ms after initial buffer is filled

For production optimization, consider:
- Using `expo-gl` with shaders for more efficient frame processing
- Implementing native modules for direct YUV frame access
- Using `vision-camera` with frame processors for real-time manipulation

## Architecture

Based on **Prototype 10** - Mobile-Optimized ConvLSTM with Global Average Pooling + ONNX Export:
- 2-layer ConvLSTM: hidden_dim = [64, 32]
- Global Average Pooling (reduces model size by ~80%)
- Dropout (0.5) for regularization
- 3-class classification output

## License

Part of thesis project for ConvLSTM Turn Prediction.
