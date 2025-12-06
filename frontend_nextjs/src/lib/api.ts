/**
 * API client for Text-to-Image-to-3D backend
 */
import axios, { AxiosError } from 'axios';
import type { GenerationRequest, GenerationResponse, HealthResponse, ErrorResponse } from '@/types/api';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Create axios instance with defaults
const api = axios.create({
  baseURL: API_URL,
  timeout: 660000, // 11 minutes for Hunyuan3D high quality texture generation
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Check backend health
 */
export async function checkHealth(): Promise<boolean> {
  try {
    const response = await api.get<HealthResponse>('/health', { timeout: 5000 });
    return response.status === 200 && response.data.status === 'healthy';
  } catch {
    return false;
  }
}

/**
 * Generate image and 3D model from prompt (full pipeline)
 */
export async function generateContent(request: GenerationRequest): Promise<GenerationResponse> {
  try {
    const response = await api.post<GenerationResponse>('/api/generate', request);
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError<ErrorResponse>;
      if (axiosError.response?.data?.detail) {
        throw new Error(axiosError.response.data.detail);
      }
      if (error.code === 'ECONNREFUSED') {
        throw new Error('Cannot connect to backend. Please ensure the backend is running.');
      }
      if (error.code === 'ETIMEDOUT') {
        throw new Error('Request timed out. Please try again.');
      }
    }
    throw new Error('An unexpected error occurred');
  }
}

/**
 * Image-only response
 */
export interface ImageOnlyResponse {
  success: boolean;
  image: string;
  processing_time: number;
  error?: string;
}

/**
 * 3D-only response
 */
export interface ThreeDOnlyResponse {
  success: boolean;
  processed_image: string;
  mesh_obj_url: string;
  mesh_glb_url: string;
  processing_time: number;
  engine_3d?: string;
  error?: string;
}

/**
 * Generate image only from prompt (progressive step 1)
 */
export async function generateImageOnly(request: {
  prompt: string;
  checkpoint: string;
  image_engine?: string;
}): Promise<ImageOnlyResponse> {
  try {
    const response = await api.post<ImageOnlyResponse>('/api/generate/image-only', request);
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError<ErrorResponse>;
      if (axiosError.response?.data?.detail) {
        throw new Error(axiosError.response.data.detail);
      }
    }
    throw new Error('Image generation failed');
  }
}

/**
 * Generate 3D model from image (progressive step 2)
 */
export async function generate3DOnly(request: {
  image: string;
  remove_background: boolean;
  foreground_ratio: number;
  mc_resolution: number;
  engine_3d: string;
  mesh_quality: string;
}): Promise<ThreeDOnlyResponse> {
  try {
    const response = await api.post<ThreeDOnlyResponse>('/api/generate/3d-only', request);
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError<ErrorResponse>;
      if (axiosError.response?.data?.detail) {
        throw new Error(axiosError.response.data.detail);
      }
    }
    throw new Error('3D generation failed');
  }
}

/**
 * Get full URL for mesh files
 */
export function getMeshUrl(path: string): string {
  return `${API_URL}${path}`;
}
