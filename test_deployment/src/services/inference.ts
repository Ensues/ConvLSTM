/**
 * TensorFlow Lite Inference Service
 * 
 * Handles model loading and inference for ConvLSTM turn prediction
 * Uses TensorFlow.js for efficient on-device inference
 * 
 * NOTE: Running in CPU-only DEMO mode for Expo Go compatibility
 * Model loading is disabled until TFJS model files are provided.
 * The app will work with simulated predictions for testing.
 */

import { NUM_CLASSES, CLASS_NAMES, MODEL_CONFIG, ClassId, PredictionClass } from '../config/modelConfig';
import { ProcessedTensor } from './preprocessor';

// TensorFlow.js is available but we use demo mode for Expo Go
// Real inference would require a development build with native modules
let tf: any = null;

// Try to load TensorFlow.js (may fail in some environments)
try {
  tf = require('@tensorflow/tfjs');
} catch (e) {
  console.warn('[TFLite] TensorFlow.js not available, using demo mode');
}

/**
 * Prediction result from model inference
 */
export interface PredictionResult {
  classId: ClassId;           // Predicted class (0, 1, 2)
  className: PredictionClass; // Human-readable class name
  confidence: number;         // Prediction confidence (0-1)
  probabilities: number[];    // All class probabilities
  inferenceTimeMs: number;    // Time taken for inference
}

/**
 * Performance metrics for tracking
 */
export interface PerformanceMetrics {
  preprocessingTimeMs: number;
  inferenceTimeMs: number;
  totalLatencyMs: number;
  fps: number;
}

/**
 * TFLite Model Manager
 * Handles loading and running inference with the ConvLSTM model
 */
class TFLiteModelManager {
  private isLoaded: boolean = false;
  private model: any = null;
  private demoMode: boolean = true;

  /**
   * Load the TFLite model
   * Must be called before running inference
   */
  async loadModel(): Promise<boolean> {
    if (this.isLoaded && this.model) {
      return true;
    }

    try {
      console.log('[TFLite] Checking TensorFlow.js availability...');
      
      if (tf) {
        // Initialize TensorFlow.js with CPU backend (Expo Go compatible)
        await tf.ready();
        console.log('[TFLite] Backend ready:', tf.getBackend());
      } else {
        console.log('[TFLite] TensorFlow.js not available');
      }
      
      console.log('[TFLite] ⚠️  Running in DEMO MODE (Expo Go)');
      console.log('[TFLite] 📱 Camera and UI work, predictions are simulated');
      console.log('[TFLite] 📝 For real inference, create a development build with:');
      console.log('[TFLite]    npx expo prebuild && npx expo run:android');
      
      // In Expo Go, we can't load native TF models - run in demo mode
      // Real model loading requires a development build with expo-gl
      this.isLoaded = false;
      this.demoMode = true;
      return true; // Return true so app continues to function in demo mode
      
    } catch (error) {
      console.error('[TFLite] Failed to initialize:', error);
      this.demoMode = true;
      return true; // Still allow app to run in demo mode
    }
  }

  /**
   * Check if model is loaded
   */
  isModelLoaded(): boolean {
    return this.isLoaded;
  }

  /**
   * Run inference on preprocessed tensor
   * 
   * @param tensor - Preprocessed frame sequence tensor
   * @returns Prediction result with class and confidence
   */
  async runInference(tensor: ProcessedTensor): Promise<PredictionResult> {
    const startTime = performance.now();

    try {
      let output: number[];
      
      if (this.demoMode || !this.isLoaded || !this.model) {
        // Fallback: Use simulated predictions for development/testing
        output = await this.simulateInference();
      } else if (tf) {
        // Real inference with loaded model (requires dev build)
        const inputTensor = tf.tensor(tensor.data, tensor.shape);
        const outputTensor = this.model.predict(inputTensor);
        
        // Get output data as array
        const outputData = await outputTensor.data();
        output = Array.from(outputData);
        
        // Cleanup tensors to prevent memory leaks
        inputTensor.dispose();
        outputTensor.dispose();
      } else {
        // TensorFlow.js not available, use simulation
        output = await this.simulateInference();
      }
      
      const inferenceTimeMs = performance.now() - startTime;
      console.log('[TFLite] Inference completed in', inferenceTimeMs.toFixed(2), 'ms');

      // Apply softmax to get probabilities (if model doesn't output probabilities already)
      const probabilities = this.softmax(output);
      
      // Get predicted class
      const classId = this.argmax(probabilities) as ClassId;
      const className = CLASS_NAMES[classId] as PredictionClass;
      const confidence = probabilities[classId];
      
      console.log('[TFLite] Prediction:', className, 'confidence:', (confidence * 100).toFixed(1) + '%');

      return {
        classId,
        className,
        confidence,
        probabilities,
        inferenceTimeMs
      };
    } catch (error) {
      console.error('[TFLite] Inference failed:', error);
      throw error;
    }
  }

  /**
   * Simulate inference for development mode (when model not loaded)
   * Generates realistic-looking predictions for testing UI
   */
  private async simulateInference(): Promise<number[]> {
    // Simulate processing delay (80-120ms)
    await new Promise(resolve => setTimeout(resolve, 80 + Math.random() * 40));
    
    // Generate more realistic logits (favor "Front" direction slightly)
    const logits: number[] = [
      Math.random() * 2 + 0.5,  // Front (slightly higher)
      Math.random() * 2 - 0.5,  // Left
      Math.random() * 2 - 0.5,  // Right
    ];
    
    return logits;
  }

  /**
   * Warm up the model with a dummy inference
   * Reduces latency for first real prediction
   */
  private async warmUp(): Promise<void> {
    try {
      const dummyTensor: ProcessedTensor = {
        data: new Float32Array(20 * 6 * 128 * 128), // SEQ_LEN * CHANNELS * HEIGHT * WIDTH
        shape: [1, 20, 6, 128, 128],
        processingTimeMs: 0
      };
      
      await this.runInference(dummyTensor);
      console.log('[TFLite] Warm-up inference successful');
    } catch (error) {
      console.warn('[TFLite] Warm-up failed (non-critical):', error);
    }
  }

  /**
   * Softmax activation function
   * Converts logits to probabilities
   */
  private softmax(logits: number[]): number[] {
    const maxLogit = Math.max(...logits);
    const expValues = logits.map(x => Math.exp(x - maxLogit));
    const sumExp = expValues.reduce((a, b) => a + b, 0);
    return expValues.map(x => x / sumExp);
  }

  /**
   * Find index of maximum value
   */
  private argmax(arr: number[]): number {
    return arr.reduce((maxIdx, val, idx, arr) => 
      val > arr[maxIdx] ? idx : maxIdx, 0);
  }

  /**
   * Unload model and free resources
   */
  async unloadModel(): Promise<void> {
    if (this.model && typeof this.model.dispose === 'function') {
      this.model.dispose();
      this.model = null;
      console.log('[TFLite] Model disposed and memory freed');
    }
    this.isLoaded = false;
    this.demoMode = true;
    console.log('[TFLite] Model unloaded');
  }
}

/**
 * Singleton model manager instance
 */
let modelManager: TFLiteModelManager | null = null;

export function getModelManager(): TFLiteModelManager {
  if (!modelManager) {
    modelManager = new TFLiteModelManager();
  }
  return modelManager;
}

/**
 * High-level inference function
 * Combines preprocessing and inference with performance tracking
 */
export async function runPrediction(tensor: ProcessedTensor): Promise<{
  prediction: PredictionResult;
  metrics: PerformanceMetrics;
}> {
  const manager = getModelManager();
  
  if (!manager.isModelLoaded()) {
    await manager.loadModel();
  }

  const prediction = await manager.runInference(tensor);

  const metrics: PerformanceMetrics = {
    preprocessingTimeMs: tensor.processingTimeMs,
    inferenceTimeMs: prediction.inferenceTimeMs,
    totalLatencyMs: tensor.processingTimeMs + prediction.inferenceTimeMs,
    fps: 1000 / (tensor.processingTimeMs + prediction.inferenceTimeMs)
  };

  return { prediction, metrics };
}

/**
 * Initialize the model (call on app startup)
 */
export async function initializeModel(): Promise<boolean> {
  const manager = getModelManager();
  return manager.loadModel();
}

/**
 * Cleanup model resources (call on app close)
 */
export async function cleanupModel(): Promise<void> {
  const manager = getModelManager();
  await manager.unloadModel();
}
