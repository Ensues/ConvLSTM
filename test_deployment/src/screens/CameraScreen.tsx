/**
 * Camera Screen - EluSEEdate
 * 
 * Live camera view with real-time turn prediction
 * - Captures frames silently from rear camera (no sound/flash)
 * - Uses sliding window: frames [1-20] → prediction, [3-22] → prediction, [5-24] → prediction, etc.
 * - Gives a new prediction every 2nd frame after initial 20-frame buffer (10 predictions/sec)
 * - Shows predicted direction at bottom
 * - Shows inference/latency metrics at top-left
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  StatusBar,
  TouchableOpacity,
  Dimensions,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/types';
import {
  FrameBuffer,
  VideoPreprocessor,
  FrameData,
} from '../services/preprocessor';
import {
  runPrediction,
  initializeModel,
  PredictionResult,
  PerformanceMetrics,
} from '../services/inference';
import { SEQ_LEN, DEVICE_CONFIG } from '../config/modelConfig';

type CameraScreenProps = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'Camera'>;
};

export default function CameraScreen({ navigation }: CameraScreenProps) {
  // Camera permission state
  const [permission, requestPermission] = useCameraPermissions();
  
  // Camera reference for frame capture
  const cameraRef = useRef<CameraView>(null);
  
  // Frame buffer for storing captured frames
  const frameBufferRef = useRef<FrameBuffer>(new FrameBuffer(DEVICE_CONFIG.cameraFps));
  
  // Preprocessor instance
  const preprocessorRef = useRef<VideoPreprocessor>(new VideoPreprocessor());
  
  // Prediction state
  const [currentPrediction, setCurrentPrediction] = useState<PredictionResult | null>(null);
  const [directionLabel, setDirectionLabel] = useState<string>('--');
  const [confidence, setConfidence] = useState<number>(0);
  
  // Performance metrics state
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    preprocessingTimeMs: 0,
    inferenceTimeMs: 0,
    totalLatencyMs: 0,
    fps: 0,
  });
  
  // Processing state
  const [isModelLoaded, setIsModelLoaded] = useState<boolean>(false);
  const [isCapturing, setIsCapturing] = useState<boolean>(false);
  const [frameCount, setFrameCount] = useState<number>(0);
  const [predictionCount, setPredictionCount] = useState<number>(0);
  
  // Inference lock to prevent concurrent inferences
  const isInferencingRef = useRef<boolean>(false);
  
  // Frame counter for prediction interval (predict every Nth frame)
  const predictionFrameCounterRef = useRef<number>(0);
  
  // Capture interval reference
  const captureIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  /**
   * Initialize model on screen mount
   */
  useEffect(() => {
    const initModel = async () => {
      console.log('[Camera] Initializing model...');
      const loaded = await initializeModel();
      setIsModelLoaded(loaded);
      if (loaded) {
        console.log('[Camera] Model initialized successfully');
      } else {
        Alert.alert('Error', 'Failed to load model');
      }
    };
    
    initModel();
    
    // Cleanup on unmount
    return () => {
      stopCapture();
    };
  }, []);

  /**
   * Start continuous frame capture when model is loaded
   */
  useEffect(() => {
    if (isModelLoaded && permission?.granted) {
      startCapture();
    }
    
    return () => {
      stopCapture();
    };
  }, [isModelLoaded, permission?.granted]);

  /**
   * Start continuous frame capture
   */
  const startCapture = useCallback(() => {
    if (captureIntervalRef.current) return;
    
    setIsCapturing(true);
    console.log('[Camera] Starting continuous capture at 20 FPS...');
    
    // Capture 20 frames per second (50ms interval) for 1 second buffer
    const captureInterval = 1000 / DEVICE_CONFIG.cameraFps; // 50ms for 20fps
    
    captureIntervalRef.current = setInterval(async () => {
      await captureFrame();
    }, captureInterval);
  }, []);

  /**
   * Stop frame capture
   */
  const stopCapture = useCallback(() => {
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }
    setIsCapturing(false);
    console.log('[Camera] Capture stopped');
  }, []);

  /**
   * Capture a single frame from camera
   */
  const captureFrame = async () => {
    if (!cameraRef.current) return;
    
    try {
      // Capture frame silently (no sound, no animation)
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.3,           // Lower quality for faster capture
        base64: true,
        skipProcessing: true,
        shutterSound: false,    // Disable shutter sound
      });
      
      if (!photo || !photo.base64) return;
      
      // Convert base64 to frame data
      // Note: In a real implementation, you'd decode the base64 to pixel data
      // For now, we'll create mock frame data for demonstration
      const frameData: FrameData = {
        data: new Uint8Array(DEVICE_CONFIG.screenWidth * DEVICE_CONFIG.screenHeight * 4),
        width: photo.width,
        height: photo.height,
        timestamp: Date.now(),
      };
      
      // Add frame to buffer (sliding window - keeps last 20 frames)
      const wasAdded = frameBufferRef.current.addFrame(frameData);
      
      if (wasAdded) {
        setFrameCount((prev: number) => prev + 1);
        predictionFrameCounterRef.current += 1;
        
        const buffer = frameBufferRef.current;
        const isBufferReady = buffer.isReady();
        
        // Wait for full buffer, then predict every Nth frame based on predictionInterval
        const shouldPredict = isBufferReady && 
                            (predictionFrameCounterRef.current % DEVICE_CONFIG.predictionInterval === 0);
        
        if (shouldPredict && !isInferencingRef.current) {
          await runInference();
        }
      }
    } catch (error) {
      // Silently ignore capture errors (camera busy, etc.)
    }
  };

  /**
   * Run model inference on buffered frames (sliding window)
   * Uses the most recent 20 frames from the buffer for prediction
   * Predictions occur every Nth frame based on predictionInterval config
   */
  const runInference = async () => {
    const buffer = frameBufferRef.current;
    
    if (isInferencingRef.current || !buffer.isReady()) {
      return;
    }
    
    isInferencingRef.current = true;
    
    try {
      // Get the most recent 20 frames (sliding window)
      const frames = buffer.getFrames();
      
      // Preprocess frames
      const preprocessor = preprocessorRef.current;
      const tensor = preprocessor.preprocessFrameSequence(frames);
      
      // Run prediction
      const { prediction, metrics: newMetrics } = await runPrediction(tensor);
      
      // Update state
      setCurrentPrediction(prediction);
      setDirectionLabel(prediction.className);
      setConfidence(prediction.confidence);
      setMetrics(newMetrics);
      setPredictionCount((prev: number) => prev + 1);
      
      console.log(`[Camera] Prediction #${predictionCount + 1}: ${prediction.className} (${(prediction.confidence * 100).toFixed(1)}%)`);
      console.log(`[Camera] Latency: ${newMetrics.totalLatencyMs.toFixed(1)}ms`);
      
      // Don't clear buffer - sliding window keeps frames for next prediction
      // Buffer auto-removes oldest frame when new frame is added
    } catch (error) {
      console.error('[Camera] Inference error:', error);
    } finally {
      isInferencingRef.current = false;
    }
  };

  /**
   * Handle back button press
   */
  const handleBack = () => {
    stopCapture();
    navigation.goBack();
  };

  // Permission not determined yet
  if (!permission) {
    return (
      <SafeAreaView style={styles.container}>
        <Text style={styles.permissionText}>Requesting camera permission...</Text>
      </SafeAreaView>
    );
  }

  // Permission denied
  if (!permission.granted) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.permissionContainer}>
          <Text style={styles.permissionText}>Camera access is required</Text>
          <TouchableOpacity style={styles.permissionButton} onPress={requestPermission}>
            <Text style={styles.permissionButtonText}>Grant Permission</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="transparent" translucent />
      
      {/* Camera View - Silent capture mode (no children allowed) */}
      <CameraView
        ref={cameraRef}
        style={styles.camera}
        facing="back"
        animateShutter={false}
        flash="off"
      />
      
      {/* Overlay Container - Absolute positioned on top of camera */}
      <View style={styles.overlayContainer}>
        {/* Performance Overlay (Top-Left) */}
        <View style={styles.performanceOverlay}>
          <Text style={styles.performanceTitle}>Performance</Text>
          <Text style={styles.performanceText}>
            Inference: {metrics.inferenceTimeMs.toFixed(0)} ms
          </Text>
          <Text style={styles.performanceText}>
            Preprocess: {metrics.preprocessingTimeMs.toFixed(0)} ms
          </Text>
          <Text style={styles.performanceText}>
            Total: {metrics.totalLatencyMs.toFixed(0)} ms
          </Text>
          <Text style={styles.performanceText}>
            FPS: {metrics.fps.toFixed(1)}
          </Text>
          <View style={styles.performanceDivider} />
          <Text style={styles.performanceText}>
            Frames: {frameCount}
          </Text>
          <Text style={styles.performanceText}>
            Predictions: {predictionCount}
          </Text>
        </View>

        {/* Back Button (Top-Right) */}
        <TouchableOpacity style={styles.backButton} onPress={handleBack}>
          <Text style={styles.backButtonText}>✕</Text>
        </TouchableOpacity>

        {/* Status Indicator */}
        <View style={styles.statusIndicator}>
          <View style={[
            styles.statusDot,
            { backgroundColor: isCapturing ? '#ffffff' : '#666666' }
          ]} />
          <Text style={styles.statusText}>
            {isCapturing ? 'Capturing' : 'Paused'}
          </Text>
          {!isModelLoaded && (
            <Text style={styles.statusText}> | Loading...</Text>
          )}
        </View>

        {/* Direction Label (Bottom) */}
        <View style={styles.directionContainer}>
          <Text style={styles.directionLabel}>{directionLabel}</Text>
          {currentPrediction && (
            <Text style={styles.confidenceText}>
              {(confidence * 100).toFixed(1)}%
            </Text>
          )}
        </View>

        {/* Frame Buffer Progress */}
        <View style={styles.bufferProgress}>
          <View style={styles.bufferContainer}>
            <View 
              style={[
                styles.bufferFill,
                { width: `${(frameBufferRef.current.getFrameCount() / SEQ_LEN) * 100}%` }
              ]} 
            />
          </View>
          <Text style={styles.bufferText}>
            Buffer: {frameBufferRef.current.getFrameCount()}/{SEQ_LEN}
          </Text>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
  },
  
  camera: {
    flex: 1,
  },

  // Overlay container - positioned absolutely on top of camera
  overlayContainer: {
    ...StyleSheet.absoluteFillObject,
    pointerEvents: 'box-none',
  },

  // Performance Overlay (Top-Left)
  performanceOverlay: {
    position: 'absolute',
    top: 50,
    left: 16,
    backgroundColor: 'rgba(0, 0, 0, 0.85)',
    padding: 10,
    borderRadius: 4,
    minWidth: 120,
    borderWidth: 1,
    borderColor: '#333333',
  },
  performanceTitle: {
    fontSize: 10,
    fontWeight: '500',
    color: '#888888',
    marginBottom: 4,
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  performanceText: {
    fontSize: 11,
    color: '#ffffff',
    fontFamily: 'monospace',
    lineHeight: 16,
  },
  performanceDivider: {
    height: 1,
    backgroundColor: '#333333',
    marginVertical: 4,
  },

  // Back Button (Top-Right)
  backButton: {
    position: 'absolute',
    top: 50,
    right: 16,
    width: 40,
    height: 40,
    backgroundColor: 'rgba(0, 0, 0, 0.85)',
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#333333',
  },
  backButtonText: {
    fontSize: 18,
    color: '#ffffff',
    fontWeight: '300',
  },

  // Status Indicator
  statusIndicator: {
    position: 'absolute',
    top: 56,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
  },
  statusDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    marginRight: 6,
  },
  statusText: {
    fontSize: 11,
    color: '#ffffff',
    fontWeight: '400',
  },

  // Direction Label (Bottom)
  directionContainer: {
    position: 'absolute',
    bottom: 80,
    left: 20,
    right: 20,
    paddingVertical: 20,
    paddingHorizontal: 20,
    borderRadius: 8,
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.85)',
    borderWidth: 1,
    borderColor: '#333333',
  },
  directionLabel: {
    fontSize: 40,
    fontWeight: '300',
    color: '#ffffff',
    letterSpacing: 6,
  },
  confidenceText: {
    fontSize: 12,
    color: '#888888',
    marginTop: 6,
  },

  // Buffer Progress
  bufferProgress: {
    position: 'absolute',
    bottom: 30,
    left: 20,
    right: 20,
    alignItems: 'center',
  },
  bufferContainer: {
    width: '100%',
    height: 2,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    borderRadius: 1,
    overflow: 'hidden',
  },
  bufferFill: {
    height: '100%',
    backgroundColor: '#ffffff',
    borderRadius: 1,
  },
  bufferText: {
    fontSize: 10,
    color: 'rgba(255, 255, 255, 0.5)',
    marginTop: 4,
  },

  // Permission States
  permissionContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 40,
  },
  permissionText: {
    fontSize: 16,
    color: '#ffffff',
    textAlign: 'center',
    marginBottom: 20,
  },
  permissionButton: {
    backgroundColor: '#ffffff',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 20,
  },
  permissionButtonText: {
    fontSize: 14,
    color: '#000000',
    fontWeight: '500',
  },
});
