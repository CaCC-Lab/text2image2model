"""診断スクリプト8: 解決策テスト - CUDAストリームを明示的に管理"""
import torch
import time
import threading
import queue

print("=" * 60)
print("TEST: Solution - Run CUDA in dedicated thread")
print("=" * 60)

from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import gradio as gr

# グローバル変数
pipe = None
task_queue = queue.Queue()
result_queue = queue.Queue()

def cuda_worker():
    """専用のCUDAワーカースレッド"""
    global pipe

    # このスレッドでパイプラインを初期化
    print("[Worker] Loading pipeline in dedicated CUDA thread...")
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
    print("[Worker] Pipeline loaded!")

    # タスクを待機
    while True:
        task = task_queue.get()
        if task is None:
            break

        prompt = task
        print(f"[Worker] Processing: {prompt}")

        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            result = pipe(prompt, num_inference_steps=4, guidance_scale=0)
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        print(f"[Worker] Done in {elapsed:.2f}s")
        result_queue.put((result.images[0], elapsed))

# ワーカースレッド開始
worker_thread = threading.Thread(target=cuda_worker, daemon=True)
worker_thread.start()

# パイプラインが準備できるまで待機
time.sleep(10)

def generate(prompt):
    """Gradioから呼ばれる関数 - タスクをワーカーに委譲"""
    print(f"[Gradio] Received request: {prompt}")
    task_queue.put(prompt)
    image, elapsed = result_queue.get()
    print(f"[Gradio] Returning result (took {elapsed:.2f}s)")
    return image

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Test with dedicated CUDA worker thread")
    prompt = gr.Textbox(label="Prompt", value="a cat")
    btn = gr.Button("Generate")
    img = gr.Image(label="Result")
    btn.click(fn=generate, inputs=prompt, outputs=img)

print("\nLaunching at http://127.0.0.1:7864")
demo.launch(server_name="127.0.0.1", server_port=7864)
