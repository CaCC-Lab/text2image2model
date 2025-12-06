# Text-to-Image-to-3D プロジェクト引き継ぎ

## 最終更新: 2024-12-06

## プロジェクト概要
テキストから画像を生成し、その画像から3Dモデルを生成するパイプライン。

## 現在の構成

### 画像生成エンジン (Image Generation)
| エンジン | 種別 | 説明 | 環境変数 |
|---------|------|------|----------|
| SDXL-Lightning | ローカル | デフォルト、超高速GPU生成 | 不要 |
| DALL-E 3 | クラウド | OpenAI高品質画像生成 | `T2I3D_OPENAI_API_KEY` |
| Gemini | クラウド | Google AI画像生成 | `T2I3D_GEMINI_API_KEY` |

### 3D生成エンジン (3D Generation)
| エンジン | 種別 | 説明 | 環境変数 |
|---------|------|------|----------|
| Hunyuan3D-2 | ローカル | デフォルト、高品質テクスチャ | 不要 |
| TripoSR | ローカル | 高速変換 | 不要 |
| Tripo API | クラウド | クラウドベース3D生成 | `T2I3D_TRIPO_API_KEY` |

## 直近の変更履歴

### 2024-12-06: 外部API統合
- **画像生成エンジン選択機能追加**
  - SDXL-Lightning (ローカル)
  - DALL-E 3 (OpenAI API)
  - Gemini (Google API)
- **3DエンジンからRodin/DALL-E削除** (ユーザー指示)
- **フロントエンドに画像エンジン選択UI追加**

### 変更ファイル一覧
**バックエンド:**
- `backend/config.py` - API設定・キー追加
- `backend/models.py` - Pydanticモデル更新
- `backend/api.py` - エンドポイント更新
- `backend/external_apis.py` - 外部APIクライアント (DALL-E, Gemini, Tripo)
- `backend/worker.py` - 画像/3D生成処理更新

**フロントエンド:**
- `frontend_nextjs/src/types/api.ts` - 型定義 (ImageEngineType追加)
- `frontend_nextjs/src/lib/store.ts` - 状態管理 (imageEngine追加)
- `frontend_nextjs/src/lib/api.ts` - APIクライアント更新
- `frontend_nextjs/src/components/SettingsPanel.tsx` - 画像エンジン選択UI
- `frontend_nextjs/src/components/GenerateButton.tsx` - imageEngine送信

## サーバー起動方法

### バックエンド (ポート8080)
```bash
cd W:\dev\text2image2model
.\venv\Scripts\activate
python -m uvicorn backend.api:app --host 0.0.0.0 --port 8080
```

### フロントエンド (ポート3000)
```bash
cd W:\dev\text2image2model\frontend_nextjs
npm run dev
```

## 環境変数設定 (.env)
```env
T2I3D_OPENAI_API_KEY=sk-xxx      # DALL-E 3用
T2I3D_GEMINI_API_KEY=xxx         # Gemini用
T2I3D_TRIPO_API_KEY=xxx          # Tripo3D用
```

## アクセスURL
- フロントエンド: http://localhost:3000
- バックエンドAPI: http://localhost:8080
- API docs: http://localhost:8080/docs

## 注意事項
- **Z:ドライブとW:ドライブ**: Z:はgitリポジトリ、W:はメイン実行環境
- **メモリ問題**: Z:ドライブでは実行エラーが発生するため、W:で実行
- 今回の変更はZ:で編集後、W:にコピー済み

## 未実装/TODO
- クラウドAPIのエラーハンドリング強化
- APIキー未設定時の警告表示
- 画像生成エンジンの生成結果プレビュー比較

## 技術スタック
- **バックエンド**: FastAPI, Python, PyTorch
- **フロントエンド**: Next.js 14, TypeScript, Tailwind CSS, Zustand
- **3Dモデル**: Hunyuan3D-2, TripoSR
- **画像生成**: SDXL-Lightning, DALL-E 3, Gemini
