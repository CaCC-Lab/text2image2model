/**
 * API types for Text-to-Image-to-3D Pipeline
 */

export type Engine3DType = 'triposr' | 'hunyuan3d' | 'hunyuan3d_mv' | 'hunyuan_api' | 'tripo_api' | 'gemini_mv' | 'auto_mv';
export type ImageEngineType = 'sdxl' | 'dalle' | 'gemini';
export type MeshQualityType = 'fast' | 'balanced' | 'high';

export interface MeshQualityInfo {
  name: string;
  description: string;
  quality: number;
  speed: number;
}

export const MESH_QUALITY_INFO: Record<MeshQualityType, MeshQualityInfo> = {
  'fast': {
    name: '高速 (10万面)',
    description: '1-2分、軽量なモデル',
    quality: 2,
    speed: 5,
  },
  'balanced': {
    name: 'バランス (20万面)',
    description: '2-4分、標準的な品質',
    quality: 3,
    speed: 3,
  },
  'high': {
    name: '高品質 (無制限)',
    description: '5-10分+、最高品質',
    quality: 5,
    speed: 1,
  },
};

export interface GenerationRequest {
  prompt: string;
  checkpoint: '1-Step' | '2-Step' | '4-Step' | '8-Step';
  remove_background: boolean;
  foreground_ratio: number;
  mc_resolution: number;
  engine_3d: Engine3DType;
}

// Frontend-friendly interface (camelCase)
export interface GenerationParams {
  prompt: string;
  checkpoint: CheckpointType;
  removeBackground: boolean;
  foregroundRatio: number;
  mcResolution: number;
  engine3d: Engine3DType;
}

// Convert frontend params to API request
export function toGenerationRequest(params: GenerationParams): GenerationRequest {
  return {
    prompt: params.prompt,
    checkpoint: params.checkpoint,
    remove_background: params.removeBackground,
    foreground_ratio: params.foregroundRatio,
    mc_resolution: params.mcResolution,
    engine_3d: params.engine3d,
  };
}

export interface GenerationResponse {
  generated_image: string;  // Base64 encoded
  processed_image: string;  // Base64 encoded
  mesh_obj_url: string;
  mesh_glb_url: string;
  processing_time: number;
  engine_3d?: string;
  // Multi-view images (for gemini_mv/auto_mv engines)
  multiview_front?: string;
  multiview_left?: string;
  multiview_right?: string;
  multiview_back?: string;
}

export interface HealthResponse {
  status: 'healthy' | 'unhealthy';
  worker_status: 'running' | 'stopped' | 'unknown';
  gpu_available: boolean;
  gpu_name?: string;
  version?: string;
}

export interface ConfigResponse {
  checkpoints: string[];
  default_checkpoint: string;
  default_remove_background: boolean;
  default_foreground_ratio: number;
  default_mc_resolution: number;
  mc_resolution_range: { min: number; max: number; step: number };
  foreground_ratio_range: { min: number; max: number; step: number };
  available_3d_engines: string[];
  default_3d_engine: string;
  available_image_engines: string[];
  default_image_engine: string;
}

export interface ErrorResponse {
  detail: string;
}

export type CheckpointType = '1-Step' | '2-Step' | '4-Step' | '8-Step';

export interface CheckpointInfo {
  name: string;
  description: string;
  speed: number;
  quality: number;
}

export const CHECKPOINT_INFO: Record<CheckpointType, CheckpointInfo> = {
  '1-Step': {
    name: '1ステップ (最速)',
    description: '超高速生成、品質は低め',
    speed: 5,
    quality: 1,
  },
  '2-Step': {
    name: '2ステップ (高速)',
    description: '高速生成、まずまずの品質',
    speed: 4,
    quality: 2,
  },
  '4-Step': {
    name: '4ステップ (バランス)',
    description: '速度と品質のバランス',
    speed: 3,
    quality: 3,
  },
  '8-Step': {
    name: '8ステップ (高品質)',
    description: '最高品質、生成は遅め',
    speed: 1,
    quality: 4,
  },
};

export const SAMPLE_PROMPTS = [
  '光沢のある赤いスポーツカー、スタジオ照明',
  'かわいいロボットキャラクター、白と青、フレンドリーなデザイン',
  'モダンな椅子、ミニマリストデザイン、木製',
  '未来的なスニーカー、ホログラフィック素材、コンセプトデザイン',
  'クリスタルの花瓶、エレガントな形状、反射する表面',
  'セラミックのコーヒーマグ、ミニマリスト、マットホワイト',
];

export interface Engine3DInfo {
  name: string;
  description: string;
  quality: number;
  speed: number;
}

export const ENGINE_3D_INFO: Record<Engine3DType, Engine3DInfo> = {
  'hunyuan3d': {
    name: 'Hunyuan3D-2 (推奨)',
    description: '高品質な3Dモデル生成、テクスチャ付き',
    quality: 5,
    speed: 3,
  },
  'hunyuan3d_mv': {
    name: 'Hunyuan3D-2 MV (マルチビュー)',
    description: '複数視点から高精度3D生成、左・後ろ画像対応',
    quality: 5,
    speed: 2,
  },
  'gemini_mv': {
    name: 'Gemini Auto MV',
    description: '画像からGeminiでマルチビュー自動生成→高精度3D',
    quality: 5,
    speed: 2,
  },
  'auto_mv': {
    name: 'フルオート MV (最高品質)',
    description: 'テキスト→画像→Gemini MV→高精度3D、全自動パイプライン',
    quality: 5,
    speed: 1,
  },
  'hunyuan_api': {
    name: 'Hunyuan API (クラウド)',
    description: 'Tencent Cloud API、APIキー必要',
    quality: 5,
    speed: 2,
  },
  'triposr': {
    name: 'TripoSR (高速)',
    description: '高速生成、シンプルなモデル向け',
    quality: 2,
    speed: 5,
  },
  'tripo_api': {
    name: 'Tripo API (クラウド)',
    description: 'クラウドベース高品質生成、APIキー必要',
    quality: 4,
    speed: 3,
  },
};

export interface ImageEngineInfo {
  name: string;
  description: string;
  quality: number;
  speed: number;
}

export const IMAGE_ENGINE_INFO: Record<ImageEngineType, ImageEngineInfo> = {
  'sdxl': {
    name: 'SDXL-Lightning (ローカル)',
    description: '超高速ローカル生成、GPU使用',
    quality: 4,
    speed: 5,
  },
  'dalle': {
    name: 'DALL-E 3 (クラウド)',
    description: 'OpenAI高品質画像生成、APIキー必要',
    quality: 5,
    speed: 3,
  },
  'gemini': {
    name: 'Gemini (クラウド)',
    description: 'Google AI画像生成、APIキー必要',
    quality: 4,
    speed: 3,
  },
};

// Part Segmentation (P3-SAM post-processing)
export interface PartSegmentationRequest {
  mesh_glb_url: string;
  post_process?: boolean;
  seed?: number;
}

export interface PartSegmentationResponse {
  success: boolean;
  segmented_mesh_url?: string;
  part_count?: number;
  processing_time?: number;
  error?: string;
}
