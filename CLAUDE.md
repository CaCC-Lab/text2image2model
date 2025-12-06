# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Text-to-Image-to-3D pipeline: テキストプロンプトから画像生成 → 3Dモデル生成。日本語プロンプト対応。

## Development Commands

```bash
# Backend (port 8080)
.\venv\Scripts\activate
python -m uvicorn backend.api:app --host 0.0.0.0 --port 8080

# Frontend (port 3000)
cd frontend_nextjs && npm run dev

# Unified launcher
python run.py --frontend nextjs   # Backend + Next.js
python run.py --backend-only      # Backend only

# Gradio standalone (legacy)
python app.py --port 7860 --share
```

## Architecture

```
テキスト → 画像生成(SDXL/DALL-E/Gemini) → 背景除去(rembg) → 3D生成(Hunyuan3D/TripoSR) → GLB/OBJ
```

### Backend (FastAPI)
- `backend/api.py` - FastAPI endpoints, CORS, request logging
- `backend/worker.py` - Multiprocess CUDA worker with `task_queue`/`result_queue`
- `backend/config.py` - Pydantic Settings, env vars with `T2I3D_` prefix
- `backend/models.py` - Request/Response Pydantic models
- `backend/external_apis.py` - DALL-E 3, Gemini, Tripo API clients
- `backend/translator.py` - Japanese→English translation (deep-translator + dictionary fallback)

### Frontend (Next.js 14)
- `frontend_nextjs/src/lib/store.ts` - Zustand state (prompt, engines, quality settings)
- `frontend_nextjs/src/lib/api.ts` - axios client (660s timeout for Hunyuan3D high quality)
- `frontend_nextjs/src/components/GenerateButton.tsx` - Two-stage generation (image → 3D)
- `frontend_nextjs/src/components/ModelViewer.tsx` - Three.js GLB viewer with OrbitControls
- `frontend_nextjs/src/types/api.ts` - TypeScript types and engine info constants

### 3D Engines
- **Hunyuan3D-2** (`hunyuan3d/`) - Default, high-quality with textures, Flow Matching DiT
- **TripoSR** (`tsr/`) - Fast, DINO→Transformer→TriplaneNeRF→Marching Cubes

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with worker status |
| `/api/generate/image-only` | POST | Text → Image generation |
| `/api/generate/3d-only` | POST | Image → 3D model generation |
| `/api/generate` | POST | Full pipeline (text → 3D) |
| `/api/models/{filename}` | GET | Download mesh files |

## Environment Variables

```bash
T2I3D_OPENAI_API_KEY=sk-xxx    # DALL-E 3
T2I3D_GEMINI_API_KEY=xxx       # Gemini
T2I3D_TRIPO_API_KEY=xxx        # Tripo API
```

## Key Types

```typescript
Engine3DType = 'hunyuan3d' | 'triposr' | 'tripo_api'
ImageEngineType = 'sdxl' | 'dalle' | 'gemini'
MeshQualityType = 'fast' | 'balanced' | 'high'  // Hunyuan3D only
CheckpointType = '1-Step' | '2-Step' | '4-Step' | '8-Step'  // SDXL steps
```

## GPU Configuration

- Dual GPU: RTX 4090 (GPU 0, Hunyuan3D) + RTX 3060 (GPU 1, SDXL)
- Single GPU: RTX 3080 10GB+ minimum
- Worker process isolates CUDA operations via multiprocessing
