"""診断スクリプト2: app.pyと同じ条件で問題を特定"""
import torch
import time
import gc

def test_with_triposr():
    """TripoSRとSDXLを同時にロードした状態でテスト"""
    print("=" * 60)
    print("TEST: SDXL + TripoSR simultaneous loading")
    print("=" * 60)

    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from tsr.system import TSR
    import rembg

    device = "cuda:0"

    # 初期VRAM
    torch.cuda.empty_cache()
    print(f"\n[Initial] VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # SDXL Pipeline
    print("\n[1] Loading SDXL Pipeline...")
    t0 = time.time()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to(device)
    print(f"    Loaded in {time.time() - t0:.2f}s")
    print(f"    VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # TripoSR
    print("\n[2] Loading TripoSR...")
    t0 = time.time()
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(8192)
    model.to(device)
    print(f"    Loaded in {time.time() - t0:.2f}s")
    print(f"    VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # rembg
    print("\n[3] Loading rembg...")
    rembg_session = rembg.new_session()
    print(f"    VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # Lightning checkpoint
    print("\n[4] Loading Lightning checkpoint...")
    t0 = time.time()
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
        prediction_type="epsilon"
    )
    ckpt_path = hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_4step_unet.safetensors")
    state_dict = load_file(ckpt_path, device="cuda:0")
    pipe.unet.load_state_dict(state_dict)
    del state_dict
    torch.cuda.empty_cache()
    print(f"    Loaded in {time.time() - t0:.2f}s")
    print(f"    VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"    VRAM reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")

    # 推論テスト
    print("\n[5] Warmup inference...")
    t0 = time.time()
    with torch.no_grad():
        _ = pipe("warmup", num_inference_steps=1, guidance_scale=0)
    torch.cuda.synchronize()
    print(f"    Warmup: {time.time() - t0:.2f}s")

    print("\n[6] Actual inference (4 steps)...")
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        result = pipe("a beautiful cat, high quality", num_inference_steps=4, guidance_scale=0)
    torch.cuda.synchronize()
    inference_time = time.time() - t0
    print(f"    Inference: {inference_time:.2f}s")
    print(f"    VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # 2回目
    print("\n[7] Second inference (4 steps)...")
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        result = pipe("a red sports car", num_inference_steps=4, guidance_scale=0)
    torch.cuda.synchronize()
    inference_time2 = time.time() - t0
    print(f"    Inference: {inference_time2:.2f}s")

    return inference_time, inference_time2

def check_cuda_stream_sync():
    """CUDAストリームの同期問題を確認"""
    print("\n" + "=" * 60)
    print("TEST: CUDA Stream Synchronization")
    print("=" * 60)

    from diffusers import StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda:0")

    # 同期なし
    print("\n[Without explicit sync]")
    t0 = time.time()
    result = pipe("test", num_inference_steps=4, guidance_scale=0)
    t1 = time.time()
    print(f"    Time: {t1-t0:.2f}s")

    # 同期あり
    print("\n[With explicit sync before and after]")
    torch.cuda.synchronize()
    t0 = time.time()
    result = pipe("test", num_inference_steps=4, guidance_scale=0)
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"    Time: {t1-t0:.2f}s")

    del pipe
    torch.cuda.empty_cache()

if __name__ == "__main__":
    print("Diagnosis 2: Simulating app.py conditions")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    time1, time2 = test_with_triposr()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"First inference: {time1:.2f}s")
    print(f"Second inference: {time2:.2f}s")

    if time1 < 5 and time2 < 5:
        print("\n✓ Both fast - problem is specific to Gradio event handling")
    elif time1 > 10 or time2 > 10:
        print("\n✗ Slow - VRAM contention between SDXL and TripoSR")
