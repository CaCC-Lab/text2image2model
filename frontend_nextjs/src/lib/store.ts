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
  isLoading: boolean;
  generationStage: 'idle' | 'image' | '3d';  // Progressive generation stage
  error: string | null;
  backendConnected: boolean;

  // Output state
  generatedImage: string | null;
  processedImage: string | null;
  meshObjUrl: string | null;
  meshGlbUrl: string | null;
  processingTime: number | null;

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
  setIsLoading: (value: boolean) => void;
  setGenerationStage: (stage: 'idle' | 'image' | '3d') => void;
  setError: (error: string | null) => void;
  setGeneratedImage: (image: string | null) => void;
  setBackendConnected: (value: boolean) => void;
  setGenerationResult: (result: GenerationResult) => void;
  clearResult: () => void;
  reset: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  // Initial state
  prompt: '',
  checkpoint: '4-Step',
  removeBackground: true,
  foregroundRatio: 0.85,
  mcResolution: 128,
  engine3d: 'hunyuan3d',
  imageEngine: 'sdxl',
  meshQuality: 'balanced',
  isLoading: false,
  generationStage: 'idle',
  error: null,
  backendConnected: false,
  generatedImage: null,
  processedImage: null,
  meshObjUrl: null,
  meshGlbUrl: null,
  processingTime: null,
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
      history: [...state.history.slice(-9), result], // Keep last 10
    })),

  clearResult: () =>
    set({
      generatedImage: null,
      processedImage: null,
      meshObjUrl: null,
      meshGlbUrl: null,
      processingTime: null,
      error: null,
    }),

  reset: () =>
    set({
      prompt: '',
      checkpoint: '4-Step',
      removeBackground: true,
      foregroundRatio: 0.85,
      mcResolution: 128,
      engine3d: 'hunyuan3d',
      imageEngine: 'sdxl',
      meshQuality: 'balanced',
      isLoading: false,
      generationStage: 'idle',
      error: null,
      generatedImage: null,
      processedImage: null,
      meshObjUrl: null,
      meshGlbUrl: null,
      processingTime: null,
    }),
}));
