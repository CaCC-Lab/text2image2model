"""診断スクリプト9: multiprocessingで完全に分離"""
import torch
import time
import multiprocessing as mp

print("=" * 60)
print("TEST: Solution - Run CUDA in separate PROCESS")
print("=" * 60)

def cuda_worker_process(task_queue, result_queue):
    """完全に別プロセスでCUDA処理"""
    import torch
    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    print("[Worker Process] Loading pipeline...")
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
    print("[Worker Process] Pipeline loaded!")

    # Ready signal
    result_queue.put("READY")

    while True:
        task = task_queue.get()
        if task is None:
            break

        prompt = task
        print(f"[Worker Process] Processing: {prompt}")

        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            result = pipe(prompt, num_inference_steps=4, guidance_scale=0)
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        print(f"[Worker Process] Done in {elapsed:.2f}s")

        # 画像をバイトに変換して送信
        import io
        buf = io.BytesIO()
        result.images[0].save(buf, format='PNG')
        result_queue.put((buf.getvalue(), elapsed))


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    task_queue = mp.Queue()
    result_queue = mp.Queue()

    # ワーカープロセス開始
    worker = mp.Process(target=cuda_worker_process, args=(task_queue, result_queue))
    worker.start()

    # 準備完了を待機
    status = result_queue.get()
    print(f"[Main] Worker status: {status}")

    # Gradio (メインプロセス)
    import gradio as gr
    from PIL import Image
    import io

    def generate(prompt):
        print(f"[Gradio] Sending request: {prompt}")
        task_queue.put(prompt)
        img_bytes, elapsed = result_queue.get()
        print(f"[Gradio] Received result (took {elapsed:.2f}s)")
        return Image.open(io.BytesIO(img_bytes))

    with gr.Blocks() as demo:
        gr.Markdown("# Test with separate CUDA process")
        prompt = gr.Textbox(label="Prompt", value="a cat")
        btn = gr.Button("Generate")
        img = gr.Image(label="Result")
        btn.click(fn=generate, inputs=prompt, outputs=img)

    print("\nLaunching at http://127.0.0.1:7870")
    demo.launch(server_name="127.0.0.1", server_port=7870)
