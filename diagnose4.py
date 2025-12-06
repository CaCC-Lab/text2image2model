"""診断スクリプト4: CUDA stream/thread問題の特定"""
import torch
import time
import os

# 環境変数を設定（スレッド問題の可能性）
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

def test_thread_cuda():
    """スレッドからCUDA操作をテスト"""
    import threading
    import concurrent.futures
    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    print("=" * 60)
    print("TEST: CUDA operations from different thread contexts")
    print("=" * 60)

    # Load pipeline
    print("\n[Loading pipeline...]")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda:0")

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

    def run_inference(name):
        print(f"\n[{name}] Thread: {threading.current_thread().name}")
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            result = pipe("a cat", num_inference_steps=4, guidance_scale=0)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        print(f"[{name}] Time: {elapsed:.2f}s")
        return elapsed

    # Test 1: Main thread
    print("\n" + "-" * 40)
    print("Test 1: Main thread")
    print("-" * 40)
    t1 = run_inference("MainThread")

    # Test 2: Using threading.Thread
    print("\n" + "-" * 40)
    print("Test 2: threading.Thread")
    print("-" * 40)
    result = [None]
    def thread_func():
        result[0] = run_inference("threading.Thread")
    thread = threading.Thread(target=thread_func)
    thread.start()
    thread.join()
    t2 = result[0]

    # Test 3: Using ThreadPoolExecutor (similar to Gradio's AnyIO)
    print("\n" + "-" * 40)
    print("Test 3: ThreadPoolExecutor")
    print("-" * 40)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_inference, "ThreadPoolExecutor")
        t3 = future.result()

    # Test 4: asyncio context (like Gradio)
    print("\n" + "-" * 40)
    print("Test 4: asyncio + run_in_executor")
    print("-" * 40)
    import asyncio

    async def async_inference():
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: run_inference("asyncio"))

    t4 = asyncio.run(async_inference())

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Main thread:         {t1:.2f}s")
    print(f"threading.Thread:    {t2:.2f}s")
    print(f"ThreadPoolExecutor:  {t3:.2f}s")
    print(f"asyncio executor:    {t4:.2f}s")

    if t2 > t1 * 10 or t3 > t1 * 10 or t4 > t1 * 10:
        print("\n⚠ CONFIRMED: Thread context significantly affects CUDA performance!")
        print("This is the root cause of the slowdown in Gradio.")
    else:
        print("\n✓ All thread contexts perform similarly.")

if __name__ == "__main__":
    test_thread_cuda()
