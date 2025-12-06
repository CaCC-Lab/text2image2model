"""診断スクリプト: 遅延の原因を特定"""
import torch
import time
import subprocess
import threading

def monitor_gpu(stop_event, results):
    """GPU使用率を監視"""
    usage_samples = []
    while not stop_event.is_set():
        try:
            output = subprocess.check_output([
                'nvidia-smi',
                '--query-gpu=utilization.gpu,memory.used,memory.total',
                '--format=csv,noheader,nounits'
            ], encoding='utf-8')
            parts = output.strip().split(', ')
            usage_samples.append({
                'gpu_util': int(parts[0]),
                'mem_used': int(parts[1]),
                'mem_total': int(parts[2])
            })
        except:
            pass
        time.sleep(0.5)
    results['gpu_samples'] = usage_samples

def test_direct_inference():
    """Gradioなしで直接推論テスト"""
    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    print("=" * 60)
    print("TEST 1: Direct inference (no Gradio)")
    print("=" * 60)

    # GPU監視スレッド開始
    stop_event = threading.Event()
    results = {}
    monitor_thread = threading.Thread(target=monitor_gpu, args=(stop_event, results))
    monitor_thread.start()

    # パイプライン初期化
    print("\n[1] Loading base pipeline...")
    t0 = time.time()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda:0")
    print(f"    Base pipeline loaded in {time.time() - t0:.2f}s")

    # チェックポイント読み込み
    print("\n[2] Loading Lightning checkpoint...")
    t0 = time.time()
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
        prediction_type="epsilon"
    )
    ckpt_path = hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_4step_unet.safetensors")
    print(f"    Checkpoint downloaded/cached in {time.time() - t0:.2f}s")

    t0 = time.time()
    state_dict = load_file(ckpt_path, device="cuda:0")
    print(f"    Safetensors loaded to GPU in {time.time() - t0:.2f}s")

    t0 = time.time()
    pipe.unet.load_state_dict(state_dict)
    print(f"    UNet state_dict loaded in {time.time() - t0:.2f}s")

    # 推論テスト (ウォームアップ)
    print("\n[3] Warmup inference...")
    t0 = time.time()
    with torch.no_grad():
        _ = pipe("warmup", num_inference_steps=1, guidance_scale=0)
    print(f"    Warmup completed in {time.time() - t0:.2f}s")

    # 推論テスト (本番)
    print("\n[4] Actual inference (4 steps)...")
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        result = pipe("a beautiful cat, high quality", num_inference_steps=4, guidance_scale=0)
    torch.cuda.synchronize()
    inference_time = time.time() - t0
    print(f"    Inference completed in {inference_time:.2f}s")

    # 監視停止
    stop_event.set()
    monitor_thread.join()

    # GPU使用状況サマリー
    if results.get('gpu_samples'):
        samples = results['gpu_samples']
        avg_util = sum(s['gpu_util'] for s in samples) / len(samples)
        max_util = max(s['gpu_util'] for s in samples)
        print(f"\n[GPU Stats] Avg utilization: {avg_util:.1f}%, Max: {max_util}%")

    return inference_time

def test_cuda_details():
    """CUDA環境の詳細確認"""
    print("\n" + "=" * 60)
    print("TEST 2: CUDA Environment Details")
    print("=" * 60)

    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")

    if torch.cuda.is_available():
        print(f"\nDevice: {torch.cuda.get_device_name(0)}")
        print(f"Device capability: {torch.cuda.get_device_capability(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"Total memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"Multi-processor count: {props.multi_processor_count}")

def test_memory_bandwidth():
    """メモリ帯域幅テスト"""
    print("\n" + "=" * 60)
    print("TEST 3: Memory Bandwidth Test")
    print("=" * 60)

    # 大きなテンソル転送テスト
    sizes = [100, 500, 1000]  # MB
    for size_mb in sizes:
        numel = size_mb * 1024 * 1024 // 4  # float32

        # CPU -> GPU
        cpu_tensor = torch.randn(numel, dtype=torch.float32)
        torch.cuda.synchronize()
        t0 = time.time()
        gpu_tensor = cpu_tensor.to("cuda:0")
        torch.cuda.synchronize()
        cpu_to_gpu = time.time() - t0

        # GPU computation
        torch.cuda.synchronize()
        t0 = time.time()
        result = gpu_tensor * 2 + 1
        torch.cuda.synchronize()
        compute = time.time() - t0

        print(f"\n{size_mb}MB tensor:")
        print(f"  CPU->GPU: {cpu_to_gpu*1000:.1f}ms ({size_mb/cpu_to_gpu:.0f} MB/s)")
        print(f"  GPU compute: {compute*1000:.1f}ms")

        del cpu_tensor, gpu_tensor, result
        torch.cuda.empty_cache()

def test_unet_forward():
    """UNet forward passの直接テスト"""
    print("\n" + "=" * 60)
    print("TEST 4: UNet Forward Pass Direct Test")
    print("=" * 60)

    from diffusers import StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda:0")

    # ダミー入力
    batch_size = 1
    latent = torch.randn(batch_size, 4, 128, 128, dtype=torch.float16, device="cuda:0")
    timestep = torch.tensor([500], device="cuda:0")
    encoder_hidden = torch.randn(batch_size, 77, 2048, dtype=torch.float16, device="cuda:0")
    added_cond = {
        "text_embeds": torch.randn(batch_size, 1280, dtype=torch.float16, device="cuda:0"),
        "time_ids": torch.randn(batch_size, 6, dtype=torch.float16, device="cuda:0")
    }

    # ウォームアップ
    with torch.no_grad():
        _ = pipe.unet(latent, timestep, encoder_hidden, added_cond_kwargs=added_cond)

    # 本番テスト
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = pipe.unet(latent, timestep, encoder_hidden, added_cond_kwargs=added_cond)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    print(f"UNet forward (10 iterations): {elapsed:.2f}s")
    print(f"Per iteration: {elapsed/10*1000:.1f}ms")

    del pipe
    torch.cuda.empty_cache()

if __name__ == "__main__":
    print("Starting comprehensive diagnosis...")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    test_cuda_details()
    test_memory_bandwidth()
    test_unet_forward()
    inference_time = test_direct_inference()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Direct inference time: {inference_time:.2f}s")
    if inference_time < 5:
        print("✓ Direct inference is FAST - problem is likely in Gradio/app integration")
    else:
        print("✗ Direct inference is SLOW - problem is in PyTorch/CUDA setup")
