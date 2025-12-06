"""診断スクリプト3: Gradioの問題を特定"""
import torch
import time
import threading
import subprocess

def monitor_gpu_during_gradio():
    """Gradio実行中のGPU使用率を監視"""
    print("=" * 60)
    print("TEST: Gradio with GPU monitoring")
    print("=" * 60)
    print("\nThis will start Gradio and monitor GPU usage.")
    print("Please trigger a generation from the web interface.")
    print("Press Ctrl+C to stop.\n")

    import gradio as gr
    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    device = "cuda:0"

    # Load models
    print("[Loading SDXL...]")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to(device)

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
    print("[SDXL loaded]")

    # GPU monitor thread
    stop_monitor = threading.Event()
    def gpu_monitor():
        while not stop_monitor.is_set():
            try:
                output = subprocess.check_output([
                    'nvidia-smi',
                    '--query-gpu=utilization.gpu,memory.used',
                    '--format=csv,noheader,nounits'
                ], encoding='utf-8')
                parts = output.strip().split(', ')
                print(f"\r[GPU] Util: {parts[0]}%, Mem: {parts[1]} MB", end="", flush=True)
            except:
                pass
            time.sleep(0.5)

    monitor_thread = threading.Thread(target=gpu_monitor, daemon=True)
    monitor_thread.start()

    def generate(prompt):
        print(f"\n[Generate called with: {prompt}]")
        print(f"[Thread: {threading.current_thread().name}]")

        # Check if we're in main thread
        import sys
        print(f"[Recursion limit: {sys.getrecursionlimit()}]")

        torch.cuda.synchronize()
        t0 = time.time()

        # 各ステップの時間を計測
        step_times = []
        def callback(pipe, step, timestep, kwargs):
            step_times.append(time.time())
            if len(step_times) > 1:
                print(f"\n  Step {step}: {step_times[-1] - step_times[-2]:.2f}s")
            return kwargs

        with torch.no_grad():
            result = pipe(
                prompt,
                num_inference_steps=4,
                guidance_scale=0,
                callback_on_step_end=callback
            )

        torch.cuda.synchronize()
        total = time.time() - t0
        print(f"\n[Total time: {total:.2f}s]")

        return result.images[0]

    # Simple Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown("# Minimal SDXL Test")
        prompt = gr.Textbox(label="Prompt", value="a cat")
        btn = gr.Button("Generate")
        img = gr.Image(label="Result")
        btn.click(fn=generate, inputs=prompt, outputs=img)

    try:
        demo.launch(server_name="127.0.0.1", server_port=7861)
    finally:
        stop_monitor.set()

if __name__ == "__main__":
    monitor_gpu_during_gradio()
