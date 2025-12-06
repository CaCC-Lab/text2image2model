# Text-to-Image-to-3D Model Pipeline

テキストプロンプトまたは画像から高品質な3Dモデルを生成するパイプライン。SDXL-Lightningによる高速画像生成とHunyuan3D-2/TripoSRによる3D再構築を組み合わせています。

## Features

- **Text-to-Image**: SDXL-Lightning (1/2/4/8-step推論) + 日本語→英語自動翻訳
- **Image-to-3D**: Hunyuan3D-2 (高品質テクスチャ付き) / TripoSR (高速)
- **画像アップロード**: 既存の画像から直接3Dモデル生成
- **メッシュ品質選択**: 高速(10万面) / バランス(20万面) / 高品質(無制限)
- **Background Removal**: rembgによる自動背景除去
- **Export Formats**: OBJ, GLB (テクスチャ付き)
- **Multiple Frontends**: Gradio, Streamlit, Reflex, Next.js
- **Dual GPU Support**: RTX 4090 (3D) + RTX 3060 (SDXL) 構成対応

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Backend API (FastAPI :8080)                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────┐    ┌──────────────────────────────────────┐     │
│  │   API Layer   │───▶│         CUDA Worker Process          │     │
│  │  (api.py)     │    │           (worker.py)                │     │
│  └───────────────┘    │                                      │     │
│                       │  GPU 0 (RTX 4090): Hunyuan3D-2       │     │
│                       │  GPU 1 (RTX 3060): SDXL-Lightning    │     │
│                       └──────────────────────────────────────┘     │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                         Frontend Options                            │
├──────────┬──────────┬──────────┬─────────────────────────────────────┤
│  Gradio  │Streamlit │  Reflex  │ Next.js (推奨)                      │
│  :7860   │  :8501   │  :3000   │ :3000                               │
└──────────┴──────────┴──────────┴─────────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Required for mesh simplification (highly recommended)
pip install fast-simplification
```

### 2. Run Application

```bash
# Start backend + Next.js frontend (recommended)
python run.py --frontend nextjs

# Start backend + Gradio frontend
python run.py --frontend gradio

# Start backend only
python run.py --backend-only

# Or run separately:
# Terminal 1: Backend
python run_backend.py --port 8080

# Terminal 2: Next.js Frontend
cd frontend_nextjs && npm run dev
```

### 3. Access the Application

- **Next.js Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8080
- **API Docs**: http://localhost:8080/docs

## 3D Engine Comparison

| Feature | Hunyuan3D-2 (推奨) | TripoSR |
|---------|-------------------|---------|
| 品質 | ★★★★★ | ★★☆☆☆ |
| 速度 | ★★★☆☆ | ★★★★★ |
| テクスチャ | 高品質UV付き | なし |
| メッシュ品質選択 | 可能 | 固定 |
| 推奨用途 | 最終成果物 | プレビュー |

## Mesh Quality Options (Hunyuan3D-2)

| 設定 | 最大面数 | 処理時間 | 用途 |
|------|---------|---------|------|
| 高速 | 100,000 | 1-2分 | プレビュー、軽量モデル |
| バランス | 200,000 | 2-4分 | 一般的な用途 |
| 高品質 | 無制限 | 5-10分+ | 最高品質、大規模モデル |

## Project Structure

```
text2image2model/
├── backend/
│   ├── api.py              # FastAPI endpoints
│   ├── worker.py           # CUDA worker (dual GPU support)
│   ├── models.py           # Pydantic models
│   └── config.py           # Settings management
├── frontend_nextjs/        # 推奨フロントエンド
│   ├── src/
│   │   ├── app/            # App Router
│   │   ├── components/     # React components
│   │   │   ├── GenerateButton.tsx
│   │   │   ├── ImageUpload.tsx    # 画像アップロード機能
│   │   │   ├── SettingsPanel.tsx  # メッシュ品質選択
│   │   │   ├── ModelViewer.tsx    # 3Dビューア
│   │   │   └── ...
│   │   ├── lib/            # API client, Store
│   │   └── types/          # TypeScript types
│   └── package.json
├── hunyuan3d/              # Hunyuan3D-2 integration
├── tsr/                    # TripoSR module
├── run.py                  # Unified launcher
└── requirements.txt
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | ヘルスチェック |
| GET | `/api/config` | 設定取得 |
| POST | `/api/generate` | テキストから画像+3D生成 |
| POST | `/api/generate/image-only` | 画像のみ生成 |
| POST | `/api/generate/3d-only` | 画像から3D生成 |
| GET | `/api/models/{filename}` | メッシュファイル取得 |

### Example Request

```bash
# Full generation (text to 3D)
curl -X POST http://localhost:8080/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "かわいいロボットキャラクター、白と青",
    "checkpoint": "4-Step",
    "remove_background": true,
    "foreground_ratio": 0.85,
    "mc_resolution": 128,
    "engine_3d": "hunyuan3d",
    "mesh_quality": "balanced"
  }'

# 3D only (from image)
curl -X POST http://localhost:8080/api/generate/3d-only \
  -H "Content-Type: application/json" \
  -d '{
    "image": "<base64_encoded_image>",
    "remove_background": true,
    "engine_3d": "hunyuan3d",
    "mesh_quality": "fast"
  }'
```

## Technology Stack

### Backend
- Python 3.10+
- FastAPI + Uvicorn
- PyTorch + CUDA
- SDXL-Lightning (ByteDance)
- Hunyuan3D-2 (Tencent) - 高品質3D生成
- TripoSR (Stability AI) - 高速3D生成
- fast-simplification - メッシュ最適化
- rembg - 背景除去

### Frontend (Next.js)
- Next.js 14 (App Router)
- React 18 + TypeScript
- Three.js / React Three Fiber
- Zustand (状態管理)
- Tailwind CSS
- Framer Motion

## Hardware Requirements

### Recommended (Dual GPU)
- GPU 0: RTX 4090 24GB (Hunyuan3D-2用)
- GPU 1: RTX 3060 12GB (SDXL用)
- RAM: 32GB+

### Minimum
- GPU: RTX 3080 10GB+ (単一GPU)
- RAM: 16GB+

## Development

```bash
# Run tests
pytest tests/

# Type checking
mypy backend/

# Linting
ruff check .

# Frontend development
cd frontend_nextjs
npm run dev
```

## Troubleshooting

### タイムアウトエラー
高品質モードでは処理に5-10分かかることがあります。タイムアウト設定は15分に設定されています。

### メッシュ簡略化が遅い
`fast-simplification`がインストールされていることを確認してください：
```bash
pip install fast-simplification
```

### GPU メモリ不足
- メッシュ品質を「高速」に変更
- TripoSRエンジンを使用

## License

MIT License

## Acknowledgments

- [SDXL-Lightning](https://huggingface.co/ByteDance/SDXL-Lightning) by ByteDance
- [Hunyuan3D-2](https://github.com/tencent/Hunyuan3D-2) by Tencent
- [TripoSR](https://huggingface.co/stabilityai/TripoSR) by Stability AI
- [DINO](https://github.com/facebookresearch/dino) by Meta AI
