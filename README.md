# Text-to-Image-to-3D Model Pipeline

テキストプロンプトまたは画像から高品質な3Dモデルを生成するパイプライン。SDXL-Lightningによる高速画像生成とHunyuan3D-2/TripoSRによる3D再構築を組み合わせています。

## Features

- **Text-to-Image**: SDXL-Lightning / DALL-E 3 / Gemini + 日本語→英語自動翻訳
- **Image-to-3D**: 複数エンジン対応
  - Hunyuan3D-2 (高品質テクスチャ付き)
  - Hunyuan3D-2 MV (マルチビュー対応)
  - Hunyuan API (クラウド)
  - TripoSR (高速ローカル)
  - Tripo API (クラウド)
- **Part Segmentation**: P3-SAMによるメッシュパーツ分割 (後処理)
- **Multi-View Upload**: 正面/左側/背面の3視点からの3D生成
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

# For P3-SAM part segmentation (optional)
git submodule update --init hunyuan3d_part
```

### 2. Environment Variables (for Cloud APIs)

```bash
# .env file
OPENAI_API_KEY=sk-xxx           # For DALL-E 3
GOOGLE_AI_API_KEY=xxx           # For Gemini
TENCENT_SECRET_ID=xxx           # For Hunyuan API
TENCENT_SECRET_KEY=xxx          # For Hunyuan API
TRIPO_API_KEY=xxx               # For Tripo API
```

### 3. Run Application

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

### 4. Access the Application

- **Next.js Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8080
- **API Docs**: http://localhost:8080/docs

## 3D Engine Comparison

| Engine | 品質 | 速度 | テクスチャ | マルチビュー | API Key |
|--------|------|------|-----------|-------------|---------|
| Hunyuan3D-2 (推奨) | ★★★★★ | ★★★☆☆ | 高品質UV | ✗ | 不要 |
| Hunyuan3D-2 MV | ★★★★★ | ★★☆☆☆ | 高品質UV | ✓ | 不要 |
| Hunyuan API | ★★★★★ | ★★☆☆☆ | 高品質UV | ✗ | 必要 |
| TripoSR | ★★☆☆☆ | ★★★★★ | なし | ✗ | 不要 |
| Tripo API | ★★★★☆ | ★★★☆☆ | あり | ✗ | 必要 |

## Image Engine Comparison

| Engine | 品質 | 速度 | API Key | 備考 |
|--------|------|------|---------|------|
| SDXL-Lightning | ★★★★☆ | ★★★★★ | 不要 | ローカルGPU使用 |
| DALL-E 3 | ★★★★★ | ★★★☆☆ | 必要 | OpenAI API |
| Gemini | ★★★★☆ | ★★★☆☆ | 必要 | Google AI API |

## Mesh Quality Options (Hunyuan3D-2)

| 設定 | 最大面数 | 処理時間 | 用途 |
|------|---------|---------|------|
| 高速 | 100,000 | 1-2分 | プレビュー、軽量モデル |
| バランス | 200,000 | 2-4分 | 一般的な用途 |
| 高品質 | 無制限 | 5-10分+ | 最高品質、大規模モデル |

## Part Segmentation (P3-SAM)

生成した3Dモデルをパーツに分割する後処理機能です。

### 使用方法
1. 3Dモデルを生成
2. 「パーツ分割 (P3-SAM)」ボタンをクリック
3. 分割完了後、「オリジナル」/「分割済み」で切り替え
4. 分割済みGLBをダウンロード

### セットアップ
```bash
# Hunyuan3D-Part サブモジュールをクローン
git submodule update --init hunyuan3d_part

# モデルは初回実行時にHuggingFaceから自動ダウンロード
# tencent/Hunyuan3D-Part (約450MB)
```

## Multi-View 3D Generation

Hunyuan3D-2 MVエンジンを使用すると、複数の視点画像から高精度な3Dモデルを生成できます。

### 使用方法
1. 3Dエンジンで「Hunyuan3D-2 MV」を選択
2. 正面画像をアップロード (必須)
3. 左側・背面画像を追加 (オプション)
4. 「3Dモデルを生成」をクリック

## Project Structure

```
text2image2model/
├── backend/
│   ├── api.py              # FastAPI endpoints
│   ├── worker.py           # CUDA worker (dual GPU support)
│   ├── models.py           # Pydantic models
│   ├── config.py           # Settings management
│   ├── external_apis.py    # Cloud API clients (DALL-E, Gemini, Hunyuan, Tripo)
│   └── part_segmentation.py # P3-SAM wrapper
├── frontend_nextjs/        # 推奨フロントエンド
│   ├── src/
│   │   ├── app/            # App Router
│   │   ├── components/     # React components
│   │   │   ├── GenerateButton.tsx
│   │   │   ├── ImageUpload.tsx    # 画像アップロード + マルチビュー
│   │   │   ├── SettingsPanel.tsx  # エンジン・品質選択
│   │   │   ├── ModelViewer.tsx    # 3Dビューア + パーツ分割
│   │   │   └── ...
│   │   ├── lib/            # API client, Store
│   │   └── types/          # TypeScript types
│   └── package.json
├── hunyuan3d/              # Hunyuan3D-2 integration
├── hunyuan3d_part/         # P3-SAM part segmentation (git submodule)
│   ├── P3-SAM/             # Part segmentation model
│   └── XPart/              # Part generation (future)
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
| POST | `/api/generate/3d-only` | 画像から3D生成 (マルチビュー対応) |
| POST | `/api/segment-parts` | メッシュパーツ分割 (P3-SAM) |
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
    "engine_3d": "hunyuan3d"
  }'

# 3D only with multi-view images
curl -X POST http://localhost:8080/api/generate/3d-only \
  -H "Content-Type: application/json" \
  -d '{
    "image": "<base64_front_image>",
    "image_left": "<base64_left_image>",
    "image_back": "<base64_back_image>",
    "remove_background": true,
    "engine_3d": "hunyuan3d_mv",
    "mesh_quality": "balanced"
  }'

# Part segmentation
curl -X POST http://localhost:8080/api/segment-parts \
  -H "Content-Type: application/json" \
  -d '{
    "mesh_glb_url": "/api/models/xxx.glb",
    "post_process": true,
    "seed": 42
  }'
```

## Technology Stack

### Backend
- Python 3.10+
- FastAPI + Uvicorn
- PyTorch + CUDA
- SDXL-Lightning (ByteDance)
- Hunyuan3D-2 / Hunyuan3D-2 MV (Tencent) - 高品質3D生成
- P3-SAM (Tencent) - パーツ分割
- TripoSR (Stability AI) - 高速3D生成
- fast-simplification - メッシュ最適化
- rembg - 背景除去

### Cloud APIs (Optional)
- OpenAI DALL-E 3 - 画像生成
- Google Gemini - 画像生成
- Tencent Hunyuan API - 3D生成
- Tripo API - 3D生成

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

# Frontend build
cd frontend_nextjs
npm run build
```

## Troubleshooting

### タイムアウトエラー
高品質モードでは処理に5-10分かかることがあります。タイムアウト設定は30分に設定されています。

### メッシュ簡略化が遅い
`fast-simplification`がインストールされていることを確認してください：
```bash
pip install fast-simplification
```

### GPU メモリ不足
- メッシュ品質を「高速」に変更
- TripoSRエンジンを使用
- Cloud APIを使用（Hunyuan API / Tripo API）

### P3-SAMが利用できない
```bash
# サブモジュールを初期化
git submodule update --init hunyuan3d_part

# Sonata依存関係の確認
pip install torch>=2.4.0
```

### Cloud APIエラー
- 環境変数でAPIキーが正しく設定されているか確認
- API利用制限に達していないか確認

## License

MIT License

## Acknowledgments

- [SDXL-Lightning](https://huggingface.co/ByteDance/SDXL-Lightning) by ByteDance
- [Hunyuan3D-2](https://github.com/tencent/Hunyuan3D-2) by Tencent
- [Hunyuan3D-Part / P3-SAM](https://github.com/Tencent-Hunyuan/Hunyuan3D-Part) by Tencent
- [TripoSR](https://huggingface.co/stabilityai/TripoSR) by Stability AI
- [Sonata](https://github.com/facebookresearch/sonata) by Meta AI
- [DINO](https://github.com/facebookresearch/dino) by Meta AI
- [OpenAI DALL-E 3](https://openai.com/dall-e-3)
- [Google Gemini](https://ai.google.dev/)
- [Tripo AI](https://www.tripo3d.ai/)
