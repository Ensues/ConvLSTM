/**
 * TensorFlow Lite Inference Service
 * 
 * Handles model loading and inference for ConvLSTM turn prediction
 * Uses react-native-tflite for efficient on-device inference
 */

import { NUM_CLASSES, CLASS_NAMES, MODEL_CONFIG, ClassId, PredictionClass } from '../config/modelConfig';
import { ProcessedTensor } from './preprocessor';

// Note: react-native-tflite needs to be installed and configured
// For Expo, you may need to use a development build or expo-dev-client

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
  private interpreter: any = null;
  private modelPath: string = 'convlstm.tflite';

  /**
   * Load the TFLite model
   * Must be called before running inference
   */
  async loadModel(): Promise<boolean> {
    if (this.isLoaded) {
      return true;
    }

    try {
      // For a real implementation with react-native-tflite:
      // const model = await loadTensorflowModel(require('../../assets/model/convlstm.tflite'));
      // this.interpreter = model;
      
      console.log('[TFLite] Loading model:', this.modelPath);
      
      // Simulate model loading for development
      // In production, replace with actual TFLite loading
      await new Promise(resolve => setTimeout(resolve, 100));
      
      this.isLoaded = true;
      console.log('[TFLite] Model loaded successfully');
      
      // Warm up model with dummy inference to initialize engine
      console.log('[TFLite] Warming up model...');
      await this.warmUp();
      console.log('[TFLite] Model warm-up complete');
      
      return true;
    } catch (error) {
      console.error('[TFLite] Failed to load model:', error);
      return false;
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
    if (!this.isLoaded) {
      throw new Error('Model not loaded. Call loadModel() first.');
    }

    const startTime = performance.now();

    try {
      // For real implementation:
      // const outputTensor = await this.interpreter.run([tensor.data]);
      // const output = outputTensor[0];
      
      // Mock inference for development
      // Replace this with actual TFLite inference
      const output = await this.mockInference(tensor);
      
      const inferenceTimeMs = performance.now() - startTime;

      // Apply softmax to get probabilities
      const probabilities = this.softmax(output);
      
      // Get predicted class
      const classId = this.argmax(probabilities) as ClassId;
      const className = CLASS_NAMES[classId] as PredictionClass;
      const confidence = probabilities[classId];

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
    } catch (error) {
      console.warn('[TFLite] Warm-up failed (non-critical):', error);
    }
  }

  /**
   * Mock inference for development/testing
   * Replace with actual TFLite inference in production
   */
  private async mockInference(tensor: ProcessedTensor): Promise<number[]> {
    // Simulate inference delay
    await new Promise(resolve => setTimeout(resolve, 80 + Math.random() * 40));
    
    // Generate mock logits (replace with real model output)
    // In production, this would be the actual model output
    const logits: number[] = [];
    for (let i = 0; i < NUM_CLASSES; i++) {
      // Generate random logits for testing
      logits.push(Math.random() * 2 - 1);
    }
    
    return logits;
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
    if (this.interpreter) {
      // In production: await this.interpreter.dispose();
      this.interpreter = null;
    }
    this.isLoaded = false;
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
