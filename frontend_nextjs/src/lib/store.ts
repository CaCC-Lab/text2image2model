/**
 * Zustand store for application state
 */
import { create } from 'zustand';
import type { CheckpointType, Engine3DType, ImageEngineType, MeshQualityType } from '@/types/api';

interface GenerationResult {
  prompt: string;
  generatedImage: string;
  processedImage: string;
  meshObjUrl: string;
  meshGlbUrl: string;
  processingTime: number;
  timestamp: number;
  engine3d?: string;
}

export type InputMode = 'text' | 'image';

interface AppState {
  // Input state
  prompt: string;
  checkpoint: CheckpointType;
  removeBackground: boolean;
  foregroundRatio: number;
  mcResolution: number;
  engine3d: Engine3DType;
  imageEngine: ImageEngineType;
  meshQuality: MeshQualityType;

  // UI state
  inputMode: InputMode;  // 'text' or 'image' input mode
  isLoading: boolean;
  generationStage: 'idle' | 'image' | '3d';  // Progressive generation stage
  isSegmenting: boolean;  // Part segmentation in progress
  error: string | null;
  backendConnected: boolean;

  // Output state
  generatedImage: string | null;
  processedImage: string | null;
  meshObjUrl: string | null;
  meshGlbUrl: string | null;
  segmentedMeshUrl: string | null;  // P3-SAM segmented mesh
  partCount: number | null;  // Number of detected parts
  processingTime: number | null;
  // Multi-view images (for gemini_mv/auto_mv engines)
  multiviewFront: string | null;
  multiviewLeft: string | null;
  multiviewRight: string | null;
  multiviewBack: string | null;

  // History
  history: GenerationResult[];

  // Actions
  setPrompt: (prompt: string) => void;
  setCheckpoint: (checkpoint: CheckpointType) => void;
  setRemoveBackground: (value: boolean) => void;
  setForegroundRatio: (value: number) => void;
  setMcResolution: (value: number) => void;
  setEngine3d: (engine: Engine3DType) => void;
  setImageEngine: (engine: ImageEngineType) => void;
  setMeshQuality: (quality: MeshQualityType) => void;
  setInputMode: (mode: InputMode) => void;
  setIsLoading: (value: boolean) => void;
  setGenerationStage: (stage: 'idle' | 'image' | '3d') => void;
  setError: (error: string | null) => void;
  setGeneratedImage: (image: string | null) => void;
  setBackendConnected: (value: boolean) => void;
  setGenerationResult: (result: GenerationResult) => void;
  setIsSegmenting: (value: boolean) => void;
  setSegmentedMesh: (url: string | null, partCount: number | null) => void;
  setMultiviewImages: (front: string | null, left: string | null, right: string | null, back: string | null) => void;
  clearResult: () => void;
  reset: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  // Initial state
  prompt: '',
  checkpoint: '4-Step',
  removeBackground: true,
  foregroundRatio: 0.90,  // Match config.py default_foreground_ratio
  mcResolution: 256,      // Match config.py default_mc_resolution
  engine3d: 'hunyuan3d',
  imageEngine: 'sdxl',
  meshQuality: 'balanced',
  inputMode: 'text',
  isLoading: false,
  generationStage: 'idle',
  isSegmenting: false,
  error: null,
  backendConnected: false,
  generatedImage: null,
  processedImage: null,
  meshObjUrl: null,
  meshGlbUrl: null,
  segmentedMeshUrl: null,
  partCount: null,
  processingTime: null,
  multiviewFront: null,
  multiviewLeft: null,
  multiviewRight: null,
  multiviewBack: null,
  history: [],

  // Actions
  setPrompt: (prompt) => set({ prompt }),
  setCheckpoint: (checkpoint) => set({ checkpoint }),
  setRemoveBackground: (removeBackground) => set({ removeBackground }),
  setForegroundRatio: (foregroundRatio) => set({ foregroundRatio }),
  setMcResolution: (mcResolution) => set({ mcResolution }),
  setEngine3d: (engine3d) => set({ engine3d }),
  setImageEngine: (imageEngine) => set({ imageEngine }),
  setMeshQuality: (meshQuality) => set({ meshQuality }),
  setInputMode: (inputMode) => set((state) => {
    // If switching to text mode and image-only MV engine is selected, switch to hunyuan3d
    if (inputMode === 'text' && (state.engine3d === 'hunyuan3d_mv' || state.engine3d === 'gemini_mv')) {
      return { inputMode, engine3d: 'hunyuan3d' };
    }
    // If switching to image mode and text-only auto_mv is selected, switch to gemini_mv
    if (inputMode === 'image' && state.engine3d === 'auto_mv') {
      return { inputMode, engine3d: 'gemini_mv' };
    }
    return { inputMode };
  }),
  setIsLoading: (isLoading) => set({ isLoading }),
  setGenerationStage: (generationStage) => set({ generationStage }),
  setError: (error) => set({ error }),
  setGeneratedImage: (generatedImage) => set({ generatedImage }),
  setBackendConnected: (backendConnected) => set({ backendConnected }),

  setGenerationResult: (result) =>
    set((state) => ({
      generatedImage: result.generatedImage,
      processedImage: result.processedImage,
      meshObjUrl: result.meshObjUrl,
      meshGlbUrl: result.meshGlbUrl,
      processingTime: result.processingTime,
      segmentedMeshUrl: null,  // Clear segmented mesh when new generation
      partCount: null,
      history: [...state.history.slice(-9), result], // Keep last 10
    })),

  setIsSegmenting: (isSegmenting) => set({ isSegmenting }),

  setSegmentedMesh: (segmentedMeshUrl, partCount) => set({ segmentedMeshUrl, partCount }),

  setMultiviewImages: (multiviewFront, multiviewLeft, multiviewRight, multiviewBack) =>
    set({ multiviewFront, multiviewLeft, multiviewRight, multiviewBack }),

  clearResult: () =>
    set({
      generatedImage: null,
      processedImage: null,
      meshObjUrl: null,
      meshGlbUrl: null,
      segmentedMeshUrl: null,
      partCount: null,
      processingTime: null,
      multiviewFront: null,
      multiviewLeft: null,
      multiviewRight: null,
      multiviewBack: null,
      error: null,
    }),

  reset: () =>
    set({
      prompt: '',
      checkpoint: '4-Step',
      removeBackground: true,
      foregroundRatio: 0.90,
      mcResolution: 256,
      engine3d: 'hunyuan3d',
      imageEngine: 'sdxl',
      meshQuality: 'balanced',
      inputMode: 'text',
      isLoading: false,
      generationStage: 'idle',
      isSegmenting: false,
      error: null,
      generatedImage: null,
      processedImage: null,
      meshObjUrl: null,
      meshGlbUrl: null,
      segmentedMeshUrl: null,
      partCount: null,
      processingTime: null,
      multiviewFront: null,
      multiviewLeft: null,
      multiviewRight: null,
      multiviewBack: null,
    }),
}));
