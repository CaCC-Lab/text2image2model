# Text-to-Image-to-3D 実装計画書

## プロジェクト概要

ソニー「次世代型AIコンテンツ生成サービス」案件に向けて、4種類のUIフレームワークでフロントエンドを実装し、Pythonプロダクト開発スキルをアピールする。

## ディレクトリ構造

```
text2image2model/
├── backend/
│   ├── __init__.py
│   ├── api.py                 # FastAPI エンドポイント
│   ├── worker.py              # CUDA ワーカープロセス
│   ├── models.py              # Pydantic モデル定義
│   └── config.py              # 設定管理
├── frontend_gradio/
│   ├── app.py                 # メインアプリ
│   ├── theme.py               # カスタムテーマ
│   └── components.py          # カスタムコンポーネント
├── frontend_streamlit/
│   ├── app.py                 # メインアプリ
│   ├── components/            # UIコンポーネント
│   └── utils.py               # ユーティリティ
├── frontend_reflex/
│   ├── frontend_reflex/       # Reflexプロジェクト
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── state.py           # 状態管理
│   │   └── components/        # UIコンポーネント
│   └── rxconfig.py
├── frontend_nextjs/
│   ├── src/
│   │   ├── app/               # App Router
│   │   ├── components/        # Reactコンポーネント
│   │   └── lib/               # API クライアント
│   ├── package.json
│   └── tailwind.config.js
├── static/                    # 共通静的ファイル
│   └── demo/                  # デモ用サンプル
├── tests/                     # テストコード
├── app.py                     # 現行版（バックアップ）
├── requirements.txt           # Python依存関係
└── IMPLEMENTATION_PLAN.md     # この計画書
```

---

## Phase 1: バックエンドAPI共通化

### 1.1 プロジェクト構造整備
- [x] `backend/` ディレクトリ作成
- [x] `__init__.py` 作成
- [x] 既存 `app.py` を `app_legacy.py` としてバックアップ

### 1.2 設定管理 (`backend/config.py`)
- [x] 環境変数による設定管理クラス作成
- [x] モデルパス設定
- [x] デバイス設定（cuda/cpu）
- [x] チェックポイント設定
- [x] Marching Cubes デフォルト解像度設定

### 1.3 Pydanticモデル定義 (`backend/models.py`)
- [x] `GenerationRequest` - 生成リクエスト
  - [x] `prompt: str`
  - [x] `checkpoint: Literal["1-Step", "2-Step", "4-Step", "8-Step"]`
  - [x] `remove_background: bool`
  - [x] `foreground_ratio: float`
  - [x] `mc_resolution: int`
- [x] `GenerationResponse` - 生成レスポンス
  - [x] `generated_image: str` (Base64)
  - [x] `processed_image: str` (Base64)
  - [x] `mesh_obj_url: str`
  - [x] `mesh_glb_url: str`
  - [x] `processing_time: float`
- [x] `HealthResponse` - ヘルスチェック
- [x] `ErrorResponse` - エラーレスポンス

### 1.4 CUDAワーカー (`backend/worker.py`)
- [x] 既存の `cuda_worker_process` をモジュール化
- [x] ワーカー起動関数 `start_worker()`
- [x] ワーカー停止関数 `stop_worker()`
- [x] タスクキュー管理
- [x] 結果キュー管理
- [x] エラーハンドリング強化
- [x] ロギング追加

### 1.5 FastAPI エンドポイント (`backend/api.py`)
- [x] FastAPI アプリケーション初期化
- [x] CORS 設定
- [x] 静的ファイル配信設定

#### エンドポイント実装
- [x] `GET /health` - ヘルスチェック
- [x] `GET /api/config` - 設定情報取得
- [x] `POST /api/generate` - 画像・3D生成
  - [x] リクエストバリデーション
  - [x] ワーカーへのタスク送信
  - [x] 結果の受信と整形
  - [x] Base64エンコード処理
- [x] `GET /api/models/{filename}` - 3Dモデルファイル配信
- [x] `POST /api/generate/image-only` - 画像のみ生成
- [x] `POST /api/generate/3d-only` - 3Dのみ生成（画像アップロード）

#### ミドルウェア
- [x] リクエストロギング
- [x] エラーハンドリング
- [x] レスポンス時間計測

### 1.6 バックエンドテスト
- [x] `tests/test_api.py` 作成
- [x] ヘルスチェックテスト
- [x] 生成エンドポイントテスト
- [x] エラーケーステスト

### 1.7 起動スクリプト
- [x] `run_backend.py` 作成
- [x] コマンドライン引数対応
  - [x] `--host`
  - [x] `--port`
  - [x] `--reload` (開発用)

---

## Phase 2: Gradio カスタム版

### 2.1 プロジェクト構造
- [x] `frontend_gradio/` ディレクトリ作成
- [x] `__init__.py` 作成

### 2.2 カスタムテーマ (`frontend_gradio/theme.py`)
- [x] ダークテーマベース作成
- [x] ソニーブランドカラー適用
  - [x] プライマリ: #000000 (ソニーブラック)
  - [x] アクセント: #0066CC (ソニーブルー)
- [x] フォント設定（Inter/JetBrains Mono）
- [x] ボーダー半径設定
- [x] シャドウ設定
- [x] ボタンスタイル
- [x] 入力フィールドスタイル

### 2.3 カスタムCSS
- [x] グラデーション背景
- [x] ホバーアニメーション
- [x] ローディングアニメーション
- [x] 3Dビューアスタイリング
- [x] レスポンシブ対応

### 2.4 UIコンポーネント (`frontend_gradio/components.py`)
- [x] ヘッダーコンポーネント
  - [x] ロゴ
  - [x] タイトル
  - [x] サブタイトル
- [x] プロンプト入力セクション
  - [x] テキストエリア（大きめ）
  - [x] サンプルプロンプトボタン
- [x] 設定パネル
  - [x] アコーディオン形式
  - [x] ステップ数選択（ドロップダウン形式）
  - [x] 背景除去トグル
  - [x] 詳細設定（折りたたみ）
- [x] 生成ボタン
  - [x] プライマリスタイル
  - [x] ローディング状態
- [x] 結果表示エリア
  - [x] 画像ギャラリー（生成前/後）
  - [x] 3Dモデルビューア
  - [x] ダウンロードボタン
- [x] 処理時間表示
- [x] フッター

### 2.5 メインアプリ (`frontend_gradio/app.py`)
- [x] バックエンドAPI接続
- [x] UIレイアウト構築
- [x] イベントハンドラ設定
- [x] エラー表示処理
- [x] プログレス表示

### 2.6 追加機能
- [x] ギャラリーモード（サンプルギャラリータブ）
- [x] プロンプト履歴（履歴タブ、JSONファイル保存）
- [ ] お気に入り保存
- [ ] 共有機能（URL生成）

### 2.7 テスト
- [x] UIコンポーネントテスト
- [ ] E2Eテスト

### 2.8 起動スクリプト
- [x] `run_gradio.py` 作成

---

## Phase 3: Streamlit版

### 3.1 プロジェクト構造
- [x] `frontend_streamlit/` ディレクトリ作成
- [x] `.streamlit/config.toml` 作成（テーマ設定）

### 3.2 テーマ設定 (`.streamlit/config.toml`)
- [x] ダークテーマ設定
- [x] プライマリカラー設定
- [x] 背景色設定
- [x] フォント設定

### 3.3 カスタムCSS
- [x] `style.css` 作成
- [x] モダンなカード UI
- [x] ボタンスタイル
- [x] アニメーション
- [x] レスポンシブ対応

### 3.4 コンポーネント (`frontend_streamlit/components/`)
- [x] `header.py` - ヘッダー
- [x] `sidebar.py` - サイドバー設定
- [x] `prompt_input.py` - プロンプト入力
- [x] `image_display.py` - 画像表示
- [x] `model_viewer.py` - 3Dモデル表示（Google model-viewer）
- [ ] `progress.py` - プログレス表示
- [x] `footer.py` - フッター

### 3.5 ユーティリティ (`frontend_streamlit/utils.py`)
- [x] API クライアント関数
- [x] Base64 変換ユーティリティ
- [x] ファイルダウンロードヘルパー
- [x] セッション状態管理

### 3.6 メインアプリ (`frontend_streamlit/app.py`)
- [x] ページ設定
- [x] レイアウト構築
  - [x] 2カラムレイアウト
  - [x] サイドバー設定パネル
  - [x] メインエリア（結果表示）
- [x] バックエンドAPI連携
- [x] 状態管理（st.session_state）
- [x] エラーハンドリング

### 3.7 追加機能
- [x] 履歴表示（セッション内）
- [ ] 画像比較スライダー
- [ ] 設定プリセット

### 3.8 起動スクリプト
- [x] `run_streamlit.py` 作成

### 3.9 テスト
- [ ] コンポーネントテスト
- [ ] 統合テスト

---

## Phase 4: Reflex版

### 4.1 プロジェクト初期化
- [x] `reflex init` でプロジェクト作成
- [x] `rxconfig.py` 設定

### 4.2 状態管理 (`frontend_reflex/state.py`)
- [x] `AppState` クラス作成
  - [x] `prompt: str`
  - [x] `checkpoint: str`
  - [x] `remove_background: bool`
  - [x] `foreground_ratio: float`
  - [x] `mc_resolution: int`
  - [x] `generated_image: str`
  - [x] `processed_image: str`
  - [x] `mesh_obj_url: str`
  - [x] `mesh_glb_url: str`
  - [x] `is_loading: bool`
  - [x] `error: str`
  - [x] `processing_time: float`
- [x] `generate()` メソッド
- [x] `reset()` メソッド
- [x] バリデーション

### 4.3 コンポーネント (`frontend_reflex/components/`)
- [x] `navbar.py` - ナビゲーションバー
- [x] `prompt_section.py` - プロンプト入力セクション
- [x] `settings_panel.py` - 設定パネル
- [x] `image_gallery.py` - 画像ギャラリー
- [x] `model_viewer.py` - 3Dモデルビューア（Google model-viewer）
- [ ] `loading_overlay.py` - ローディング表示
- [ ] `result_section.py` - 結果表示セクション
- [x] `footer.py` - フッター

### 4.4 スタイリング
- [x] Tailwind CSS 設定
- [x] カスタムスタイル定義
- [x] レスポンシブデザイン
- [ ] アニメーション

### 4.5 メインアプリ (`frontend_reflex/app.py`)
- [x] ルーティング設定
- [x] レイアウト構築
- [x] コンポーネント配置
- [x] 状態バインディング

### 4.6 3Dビューア統合
- [x] Google model-viewer 統合
- [x] GLB/OBJ ローダー実装
- [x] 回転・ズーム操作
- [ ] ライティング設定

### 4.7 追加機能
- [ ] ダークモード切り替え
- [ ] 言語切り替え（日/英）
- [ ] キーボードショートカット

### 4.8 起動スクリプト
- [x] `run_reflex.py` 作成

### 4.9 テスト
- [ ] 状態管理テスト
- [ ] コンポーネントテスト
- [ ] E2Eテスト

---

## Phase 5: Next.js + FastAPI版

### 5.1 Next.js プロジェクト初期化
- [x] `npx create-next-app@latest` でプロジェクト作成
- [x] TypeScript 設定
- [x] Tailwind CSS 設定
- [x] ESLint 設定

### 5.2 依存関係インストール
- [x] `@react-three/fiber` - Three.js React バインディング
- [x] `@react-three/drei` - Three.js ヘルパー
- [x] `framer-motion` - アニメーション
- [x] `lucide-react` - アイコン
- [ ] `react-dropzone` - ファイルドロップ
- [x] `zustand` - 状態管理
- [x] `axios` - API クライアント
- [x] `react-hot-toast` - 通知

### 5.3 型定義 (`src/types/`)
- [x] `api.ts` - API リクエスト/レスポンス型
- [x] `state.ts` - 状態型（store.tsに統合）

### 5.4 API クライアント (`src/lib/api.ts`)
- [x] Axios インスタンス設定
- [x] `generateContent()` 関数
- [x] `getHealth()` 関数
- [ ] `getConfig()` 関数
- [x] エラーハンドリング

### 5.5 状態管理 (`src/lib/store.ts`)
- [x] Zustand ストア作成
- [x] 生成パラメータ状態
- [x] 結果状態
- [x] UI状態（ローディング等）

### 5.6 UIコンポーネント (`src/components/`)

#### レイアウト
- [x] `Layout.tsx` - ベースレイアウト（layout.tsxに統合）
- [x] `Header.tsx` - ヘッダー
- [x] `Footer.tsx` - フッター
- [ ] `Sidebar.tsx` - サイドバー

#### 入力
- [x] `PromptInput.tsx` - プロンプト入力
- [x] `CheckpointSelector.tsx` - ステップ選択（SettingsPanelに統合）
- [x] `SettingsPanel.tsx` - 設定パネル
- [x] `Slider.tsx` - カスタムスライダー（SettingsPanelに統合）
- [x] `Toggle.tsx` - トグルスイッチ（SettingsPanelに統合）

#### 表示
- [x] `ImagePreview.tsx` - 画像プレビュー（ImageDisplayに統合）
- [ ] `ImageComparison.tsx` - 画像比較
- [x] `ModelViewer.tsx` - 3Dモデルビューア
- [ ] `LoadingOverlay.tsx` - ローディング
- [ ] `ProgressBar.tsx` - プログレスバー
- [ ] `ResultCard.tsx` - 結果カード

#### 3Dビューア詳細 (`src/components/ModelViewer/`)
- [x] `Scene.tsx` - Three.js シーン（ModelViewerに統合）
- [x] `Model.tsx` - モデル表示（ModelViewerに統合）
- [x] `Controls.tsx` - カメラコントロール（OrbitControlsで実装）
- [x] `Lighting.tsx` - ライティング（ModelViewerに統合）
- [x] `Environment.tsx` - 環境マップ（drei Environmentで実装）
- [ ] `ExportButton.tsx` - エクスポート

### 5.7 ページ (`src/app/`)
- [x] `page.tsx` - メインページ
- [x] `layout.tsx` - ルートレイアウト
- [ ] `loading.tsx` - ローディング状態
- [ ] `error.tsx` - エラー状態
- [x] `globals.css` - グローバルスタイル

### 5.8 アニメーション
- [x] ページ遷移アニメーション（framer-motion）
- [x] コンポーネント出現アニメーション
- [x] ホバーエフェクト
- [x] ローディングアニメーション
- [x] 3Dモデル回転アニメーション（autoRotate）

### 5.9 レスポンシブ対応
- [x] モバイルレイアウト
- [x] タブレットレイアウト
- [x] デスクトップレイアウト

### 5.10 最適化
- [x] 画像最適化（next/image）
- [ ] コード分割
- [ ] 動的インポート（3Dビューア）
- [x] メタデータ設定

### 5.11 起動スクリプト
- [x] `run_nextjs.py` 作成

### 5.12 テスト
- [ ] Jest 設定
- [ ] コンポーネントテスト
- [ ] E2Eテスト（Playwright）

---

## Phase 6: 統合・ドキュメント

### 6.1 起動スクリプト統合
- [x] `run.py` - 統合起動スクリプト
  - [x] `--frontend` オプション（gradio/streamlit/reflex/nextjs）
  - [x] `--backend-port` オプション
  - [x] `--backend-only` オプション
  - [x] `--frontend-only` オプション
  - [ ] `--dev` オプション（開発モード）

### 6.2 Docker対応
- [ ] `Dockerfile.backend` - バックエンド用
- [ ] `Dockerfile.gradio` - Gradio版
- [ ] `Dockerfile.streamlit` - Streamlit版
- [ ] `Dockerfile.reflex` - Reflex版
- [ ] `Dockerfile.nextjs` - Next.js版
- [ ] `docker-compose.yml` - 統合構成

### 6.3 ドキュメント
- [x] `README.md` 更新
  - [x] プロジェクト概要
  - [x] 各フロントエンドの説明
  - [x] インストール手順
  - [x] 起動方法
  - [ ] スクリーンショット
- [ ] `docs/API.md` - API ドキュメント
- [ ] `docs/ARCHITECTURE.md` - アーキテクチャ説明
- [ ] `docs/DEVELOPMENT.md` - 開発ガイド

### 6.4 デモ準備
- [x] サンプルプロンプト集（各フロントエンドに組み込み済み）
- [ ] デモ用画像/モデル
- [ ] プレゼン資料

---

## タイムライン目安

| Phase | 内容 | 目安時間 |
|-------|------|----------|
| 1 | バックエンドAPI共通化 | 4-6時間 |
| 2 | Gradioカスタム版 | 3-4時間 |
| 3 | Streamlit版 | 4-5時間 |
| 4 | Reflex版 | 5-6時間 |
| 5 | Next.js版 | 8-10時間 |
| 6 | 統合・ドキュメント | 3-4時間 |
| **合計** | | **27-35時間** |

---

## 優先順位

1. **Phase 1** (必須) - 全てのベース
2. **Phase 2** (高) - 既存コードからの改良で最速
3. **Phase 3** (高) - Pythonスキルアピール
4. **Phase 5** (中) - 本番品質アピール
5. **Phase 4** (中) - 技術幅アピール
6. **Phase 6** (高) - 面談用資料

---

## 技術スタック詳細

### バックエンド
- Python 3.11+
- FastAPI
- PyTorch + CUDA
- SDXL-Lightning
- TripoSR
- Pydantic v2
- Uvicorn

### Gradio版
- Gradio 6.x
- カスタムCSS/JS

### Streamlit版
- Streamlit 1.x
- streamlit-3d-viewer

### Reflex版
- Reflex 0.4+
- カスタムThree.jsコンポーネント

### Next.js版
- Next.js 14 (App Router)
- React 18
- TypeScript
- Tailwind CSS
- Three.js / React Three Fiber
- Framer Motion
- Zustand

---

## 注意事項

1. **CUDA処理は必ず別プロセスで実行** - Gradio 6との互換性問題を回避
2. **Base64エンコードのサイズ制限に注意** - 大きな画像はURL参照に切り替え
3. **3Dモデルファイルはtempfileを使用** - 適切なクリーンアップが必要
4. **CORS設定を適切に** - Next.js版では必須
5. **エラーハンドリングを丁寧に** - ユーザー体験向上

---

*Last Updated: 2024-12-05*
