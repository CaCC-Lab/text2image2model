"""
P3-SAM Part Segmentation Module.

Wraps the Hunyuan3D-Part P3-SAM model for mesh part segmentation.
This is a POST-PROCESSING step that segments an existing mesh into parts.

Usage:
    from backend.part_segmentation import segment_mesh_parts

    # Segment an existing mesh
    segmented_mesh, aabb, face_ids = segment_mesh_parts(mesh_path)
"""
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Chinese to Japanese translation map for P3-SAM logs
_CN_TO_JP_MAP = {
    # Loading messages
    "加载模型": "モデル読み込み中",
    "模型加载完成": "モデル読み込み完了",
    "加载数据列表": "データリスト読み込み中",
    "加载数据": "データ読み込み中",
    "加载数据完成": "データ読み込み完了",
    "加载mesh": "メッシュ読み込み",
    # Processing messages
    "预处理特征": "特徴量の前処理中",
    "PCA获取特征颜色": "PCAで特徴色を取得中",
    "采样完成": "サンプリング完了",
    "采样点云": "点群サンプリング",
    "prmopt 推理完成": "プロンプト推論完了",
    "prompt 推理完成": "プロンプト推論完了",
    "获取邻接面片": "隣接面の取得",
    "处理邻接面片": "隣接面の処理",
    # Results messages
    "NMS完成，mask数量": "NMS完了、マスク数",
    "再次合并，合并数量": "再統合完了、統合数",
    "修补遗漏mask": "欠損マスク修復",
    "最终mask数量": "最終マスク数",
    "保存点云完成": "点群保存完了",
    "生成face_ids完成": "face_ids生成完了",
    "修复face_ids完成": "face_ids修復完了",
    "保存mesh完成": "メッシュ保存完了",
    "保存mesh结果完成": "メッシュ結果保存完了",
    # Stats messages
    "点数": "頂点数",
    "面片数": "面数",
    "连通区域数量": "連結領域数",
    "ID数量": "ID数",
    "总面积": "総面積",
    "合并mesh": "メッシュ統合",
    "共": "合計",
    "个mesh": "個のメッシュ",
    "个数据": "個のデータ",
    "保留": "保持",
    "其他": "その他",
    "新增part": "新規パーツ",
    "选择点": "選択点",
    "获取mask成功": "マスク取得成功",
    "获取iou成功": "IoU取得成功",
    "最佳mask": "最適マスク",
    "运行时间": "実行時間",
    "秒": "秒",
    "代码": "処理",
}


class ChineseToJapaneseFilter:
    """Filter that translates Chinese print output to Japanese."""

    def __init__(self, original_write):
        self.original_write = original_write

    def write(self, text):
        translated = text
        for cn, jp in _CN_TO_JP_MAP.items():
            translated = translated.replace(cn, jp)
        self.original_write(translated)

    def flush(self):
        pass


def _install_translation_filter():
    """Install Chinese-to-Japanese translation filter on stdout."""
    import builtins
    original_print = builtins.print

    def translated_print(*args, **kwargs):
        # Convert args to strings and translate
        translated_args = []
        for arg in args:
            text = str(arg)
            for cn, jp in _CN_TO_JP_MAP.items():
                text = text.replace(cn, jp)
            translated_args.append(text)
        original_print(*translated_args, **kwargs)

    builtins.print = translated_print
    return original_print


def _uninstall_translation_filter(original_print):
    """Restore original print function."""
    import builtins
    builtins.print = original_print

# Add P3-SAM to path
P3SAM_PATH = Path(__file__).parent.parent / "hunyuan3d_part" / "P3-SAM"
XPART_PATH = Path(__file__).parent.parent / "hunyuan3d_part" / "XPart"

# Global model instance (lazy loaded)
_auto_mask_model = None
_auto_mask_post_process = None
_sonata_patched = False


def _patch_sonata_flash_attention():
    """
    Patch the sonata model loader to disable flash attention on Windows.
    Flash attention requires CUDA compilation which is difficult on Windows.
    """
    global _sonata_patched
    if _sonata_patched:
        return

    try:
        # Check if flash_attn is available
        import flash_attn
        logger.info("[P3-SAM] flash_attn available, no patch needed")
        return
    except ImportError:
        logger.info("[P3-SAM] flash_attn not available, patching sonata to disable it")

    try:
        from models import sonata
        original_load = sonata.load

        def patched_load(name="sonata", repo_id="facebook/sonata", download_root=None, custom_config=None, ckpt_only=False):
            """Patched sonata.load that disables flash attention."""
            if custom_config is None:
                custom_config = {}
            # Disable flash attention
            custom_config['enable_flash'] = False
            logger.info("[P3-SAM] Disabled flash attention in sonata model")
            return original_load(name, repo_id=repo_id, download_root=download_root,
                               custom_config=custom_config, ckpt_only=ckpt_only)

        sonata.load = patched_load
        _sonata_patched = True
        logger.info("[P3-SAM] Sonata patched successfully")

    except Exception as e:
        logger.warning(f"[P3-SAM] Failed to patch sonata: {e}")


def _setup_paths():
    """Setup Python paths for P3-SAM imports."""
    p3sam_demo = str(P3SAM_PATH / "demo")
    p3sam_root = str(P3SAM_PATH)
    xpart_partgen = str(XPART_PATH / "partgen")

    for path in [p3sam_demo, p3sam_root, xpart_partgen]:
        if path not in sys.path:
            sys.path.insert(0, path)

    # Inject fpsample fallback for Windows compatibility
    try:
        import fpsample
    except ImportError:
        logger.info("[P3-SAM] fpsample not found, using numpy fallback")
        from . import fpsample_fallback
        sys.modules['fpsample'] = fpsample_fallback

    # Patch sonata to disable flash attention on Windows (flash_attn not available)
    # Must be called AFTER paths are set up
    _patch_sonata_flash_attention()


def _ensure_setup():
    """Ensure paths and patches are applied."""
    _setup_paths()


def is_p3sam_available() -> bool:
    """Check if P3-SAM is available and can be loaded."""
    if not P3SAM_PATH.exists():
        return False
    if not (P3SAM_PATH / "demo" / "auto_mask.py").exists():
        return False
    if not (P3SAM_PATH / "model.py").exists():
        return False
    return True


def get_auto_mask_model(post_process: bool = True):
    """
    Get or create the AutoMask model instance.

    Args:
        post_process: Whether to enable post-processing for cleaner segmentation

    Returns:
        AutoMask model instance
    """
    global _auto_mask_model, _auto_mask_post_process

    if _auto_mask_model is not None and _auto_mask_post_process == post_process:
        return _auto_mask_model

    if not is_p3sam_available():
        raise RuntimeError(
            "P3-SAM is not available. Please ensure hunyuan3d_part submodule is cloned."
        )

    _setup_paths()

    logger.info("[P3-SAM] Loading AutoMask model...")

    try:
        from auto_mask import AutoMask

        # Check if flash_attn is available - use fewer points if not (much slower)
        try:
            import flash_attn
            point_num = 100000  # Full resolution with flash attention
            prompt_num = 400
        except ImportError:
            point_num = 10000  # Heavily reduced for standard attention (much faster)
            prompt_num = 100  # Reduced prompts
            logger.warning("[P3-SAM] Using reduced settings (10K points, 100 prompts) due to missing flash_attn")

        # Apply Chinese-to-Japanese translation filter during model loading
        original_print = _install_translation_filter()
        try:
            # Model will auto-download from HuggingFace if not present
            _auto_mask_model = AutoMask(
                ckpt_path=None,  # Auto-download from HuggingFace
                point_num=point_num,
                prompt_num=prompt_num,
                threshold=0.95,
                post_process=post_process,
            )
        finally:
            _uninstall_translation_filter(original_print)

        _auto_mask_post_process = post_process

        logger.info("[P3-SAM] AutoMask model loaded!")
        return _auto_mask_model

    except Exception as e:
        logger.error(f"[P3-SAM] Failed to load model: {e}")
        raise


def unload_p3sam_model():
    """Unload P3-SAM model to free GPU memory."""
    global _auto_mask_model, _auto_mask_post_process

    if _auto_mask_model is not None:
        logger.info("[P3-SAM] Unloading model...")
        del _auto_mask_model
        _auto_mask_model = None
        _auto_mask_post_process = None

        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def segment_mesh_parts(
    mesh_path: str,
    output_dir: Optional[str] = None,
    post_process: bool = True,
    save_intermediate: bool = False,
    seed: int = 42,
) -> Tuple[str, np.ndarray, np.ndarray]:
    """
    Segment a 3D mesh into parts using P3-SAM.

    Args:
        mesh_path: Path to input mesh file (GLB, OBJ, or PLY)
        output_dir: Directory to save output files (default: temp directory)
        post_process: Whether to enable post-processing
        save_intermediate: Whether to save intermediate results
        seed: Random seed for reproducibility

    Returns:
        Tuple of (segmented_mesh_path, aabb_array, face_ids_array)
        - segmented_mesh_path: Path to the colored segmented mesh
        - aabb_array: Bounding boxes for each part [N, 2, 3]
        - face_ids_array: Part ID for each face
    """
    import trimesh

    _setup_paths()
    from auto_mask import set_seed

    logger.info(f"[P3-SAM] Segmenting mesh: {mesh_path}")

    # Create output directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="p3sam_")
    os.makedirs(output_dir, exist_ok=True)

    # Load mesh
    mesh = trimesh.load(mesh_path, force='mesh')
    logger.info(f"[P3-SAM] Loaded mesh: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")

    # Get model
    model = get_auto_mask_model(post_process=post_process)

    # Set seed for reproducibility
    set_seed(seed)

    # Run segmentation with Chinese-to-Japanese log translation
    logger.info("[P3-SAM] Running part segmentation...")
    original_print = _install_translation_filter()
    try:
        aabb, face_ids, processed_mesh = model.predict_aabb(
            mesh,
            save_path=output_dir,
            save_mid_res=save_intermediate,
            show_info=True,
            clean_mesh_flag=True,
            seed=seed,
            is_parallel=True,
            prompt_bs=32,
        )
    finally:
        _uninstall_translation_filter(original_print)

    logger.info(f"[P3-SAM] Found {len(np.unique(face_ids))} parts")

    # Create colored mesh
    color_map = {}
    unique_ids = np.unique(face_ids)
    np.random.seed(seed)
    for i in unique_ids:
        if i < 0:
            continue
        color_map[i] = (np.random.rand(3) * 255).astype(np.uint8)

    face_colors = np.zeros((len(processed_mesh.faces), 4), dtype=np.uint8)
    for i, fid in enumerate(face_ids):
        if fid >= 0 and fid in color_map:
            face_colors[i, :3] = color_map[fid]
            face_colors[i, 3] = 255
        else:
            face_colors[i] = [128, 128, 128, 255]  # Gray for unassigned

    # Create output mesh with colors
    segmented_mesh = trimesh.Trimesh(
        vertices=processed_mesh.vertices,
        faces=processed_mesh.faces,
    )
    segmented_mesh.visual.face_colors = face_colors

    # Save output files
    output_glb = os.path.join(output_dir, "segmented_mesh.glb")
    segmented_mesh.export(output_glb)

    # Also save face_ids and aabb
    np.save(os.path.join(output_dir, "face_ids.npy"), face_ids)
    np.save(os.path.join(output_dir, "aabb.npy"), aabb)

    logger.info(f"[P3-SAM] Saved segmented mesh to: {output_glb}")

    return output_glb, aabb, face_ids


def segment_mesh_from_bytes(
    mesh_data: bytes,
    mesh_format: str = "glb",
    post_process: bool = True,
    seed: int = 42,
) -> Tuple[bytes, np.ndarray, np.ndarray]:
    """
    Segment a mesh from bytes data.

    Args:
        mesh_data: Raw mesh data bytes
        mesh_format: File format (glb, obj, ply)
        post_process: Whether to enable post-processing
        seed: Random seed

    Returns:
        Tuple of (segmented_mesh_bytes, aabb_array, face_ids_array)
    """
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=f".{mesh_format}", delete=False) as f:
        f.write(mesh_data)
        temp_input = f.name

    try:
        # Run segmentation
        output_dir = tempfile.mkdtemp(prefix="p3sam_")
        output_path, aabb, face_ids = segment_mesh_parts(
            temp_input,
            output_dir=output_dir,
            post_process=post_process,
            seed=seed,
        )

        # Read output as bytes
        with open(output_path, "rb") as f:
            output_bytes = f.read()

        return output_bytes, aabb, face_ids

    finally:
        # Cleanup
        os.unlink(temp_input)


# Check availability on import
if not is_p3sam_available():
    logger.warning(
        "P3-SAM is not available. Run: git submodule update --init hunyuan3d_part"
    )
