"""
Text to Image to 3D Model Pipeline
CUDA処理を別プロセスで実行してGradio 6との互換性問題を回避
"""
import logging
import os
import tempfile
import time
import argparse
import multiprocessing as mp
import io
import numpy as np
from PIL import Image

# Constants for SDXL-Lightning
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
LIGHTNING_REPO = "ByteDance/SDXL-Lightning"
CHECKPOINTS = {
    "1-Step": ["sdxl_lightning_1step_unet_x0.safetensors", 1],
    "2-Step": ["sdxl_lightning_2step_unet.safetensors", 2],
    "4-Step": ["sdxl_lightning_4step_unet.safetensors", 4],
    "8-Step": ["sdxl_lightning_8step_unet.safetensors", 8],
}


def cuda_worker_process(task_queue, result_queue):
    """
    別プロセスで実行されるCUDAワーカー
    Gradio 6のasyncioイベントループとの干渉を避けるため
    """
    import torch
    import rembg
    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from tsr.system import TSR
    from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    loaded = None

    print("[Worker] Initializing models...")

    # Initialize SDXL-Lightning
    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        variant="fp16"
    ).to(device)

    # Initialize TripoSR
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(8192)
    model.to(device)

    # Initialize rembg
    rembg_session = rembg.new_session()

    print("[Worker] Models initialized!")
    result_queue.put("READY")

    def fill_background(image):
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        return image

    while True:
        task = task_queue.get()
        if task is None:
            break

        try:
            task_type = task["type"]

            if task_type == "text_to_3d":
                prompt = task["prompt"]
                ckpt = task["ckpt"]
                do_remove_background = task["do_remove_background"]
                foreground_ratio = task["foreground_ratio"]
                mc_resolution = task["mc_resolution"]

                total_start = time.time()

                # Generate image
                checkpoint = CHECKPOINTS[ckpt][0]
                num_inference_steps = CHECKPOINTS[ckpt][1]

                if loaded != num_inference_steps:
                    print(f"[Worker] Loading checkpoint: {checkpoint}")
                    t0 = time.time()
                    pipe.scheduler = EulerDiscreteScheduler.from_config(
                        pipe.scheduler.config,
                        timestep_spacing="trailing",
                        prediction_type="sample" if num_inference_steps == 1 else "epsilon"
                    )
                    ckpt_path = hf_hub_download(LIGHTNING_REPO, checkpoint)
                    state_dict = load_file(ckpt_path, device=device)
                    pipe.unet.load_state_dict(state_dict)
                    loaded = num_inference_steps
                    print(f"[Worker] Checkpoint loaded in {time.time() - t0:.2f}s")

                print(f"[Worker] Generating image...")
                t0 = time.time()
                torch.cuda.synchronize()
                with torch.no_grad():
                    results = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=0)
                torch.cuda.synchronize()
                generated_image = results.images[0]
                print(f"[Worker] Image generation: {time.time() - t0:.2f}s")

                # Preprocess
                t0 = time.time()
                if do_remove_background:
                    image = generated_image.convert("RGB")
                    image = remove_background(image, rembg_session)
                    image = resize_foreground(image, foreground_ratio)
                    image = fill_background(image)
                else:
                    image = generated_image
                    if image.mode == "RGBA":
                        image = fill_background(image)
                processed_image = image
                print(f"[Worker] Preprocessing: {time.time() - t0:.2f}s")

                # Generate 3D
                print(f"[Worker] Generating 3D...")
                t0 = time.time()
                scene_codes = model(processed_image, device=device)
                print(f"[Worker] Scene encoding: {time.time() - t0:.2f}s")

                t0 = time.time()
                mesh = model.extract_mesh(scene_codes, True, resolution=mc_resolution)[0]
                print(f"[Worker] Mesh extraction: {time.time() - t0:.2f}s")

                mesh = to_gradio_3d_orientation(mesh)

                # Export meshes to temp files
                mesh_paths = []
                for fmt in ["obj", "glb"]:
                    mesh_path = tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False)
                    mesh.export(mesh_path.name)
                    mesh_paths.append(mesh_path.name)

                print(f"[Worker] === TOTAL: {time.time() - total_start:.2f}s ===")

                # Convert images to bytes
                gen_buf = io.BytesIO()
                generated_image.save(gen_buf, format='PNG')

                proc_buf = io.BytesIO()
                processed_image.save(proc_buf, format='PNG')

                result_queue.put({
                    "success": True,
                    "generated_image": gen_buf.getvalue(),
                    "processed_image": proc_buf.getvalue(),
                    "mesh_obj": mesh_paths[0],
                    "mesh_glb": mesh_paths[1],
                })

        except Exception as e:
            import traceback
            traceback.print_exc()
            result_queue.put({
                "success": False,
                "error": str(e)
            })


# Global queues for IPC
task_queue = None
result_queue = None
worker_process = None


def start_worker():
    """ワーカープロセスを開始"""
    global task_queue, result_queue, worker_process

    task_queue = mp.Queue()
    result_queue = mp.Queue()

    worker_process = mp.Process(
        target=cuda_worker_process,
        args=(task_queue, result_queue),
        daemon=True
    )
    worker_process.start()

    # Wait for worker to be ready
    status = result_queue.get(timeout=120)
    if status != "READY":
        raise RuntimeError("Worker failed to initialize")
    print("[Main] Worker is ready!")


def text_to_3d_pipeline(prompt, ckpt, do_remove_background, foreground_ratio, mc_resolution):
    """Gradioから呼ばれる関数 - タスクをワーカーに委譲"""
    import gradio as gr

    print(f"[Main] Sending task to worker: {prompt}")

    task_queue.put({
        "type": "text_to_3d",
        "prompt": prompt,
        "ckpt": ckpt,
        "do_remove_background": do_remove_background,
        "foreground_ratio": foreground_ratio,
        "mc_resolution": mc_resolution,
    })

    result = result_queue.get()

    if not result["success"]:
        raise gr.Error(f"Generation failed: {result['error']}")

    # Convert bytes back to images
    generated_image = Image.open(io.BytesIO(result["generated_image"]))
    processed_image = Image.open(io.BytesIO(result["processed_image"]))

    return generated_image, processed_image, result["mesh_obj"], result["mesh_glb"]


def create_interface():
    import gradio as gr

    with gr.Blocks(title="Text to Image to 3D Model") as interface:
        gr.Markdown(
            """
            # Text to Image to 3D Model Pipeline
            This application combines SDXL-Lightning for text-to-image generation and TripoSR for 3D model reconstruction.

            ## Process:
            1. Enter your text prompt to generate an image
            2. The image will be automatically processed
            3. A 3D model will be generated from the processed image
            """
        )

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Text Prompt")
                ckpt = gr.Dropdown(
                    label="SDXL-Lightning Steps",
                    choices=list(CHECKPOINTS.keys()),
                    value="4-Step"
                )
                do_remove_background = gr.Checkbox(label="Remove Background", value=True)
                foreground_ratio = gr.Slider(
                    label="Foreground Ratio",
                    minimum=0.5,
                    maximum=1.0,
                    value=0.85,
                    step=0.05,
                )
                mc_resolution = gr.Slider(
                    label="Marching Cubes Resolution",
                    minimum=32,
                    maximum=256,
                    value=128,
                    step=32
                )
                submit = gr.Button("Generate", variant="primary")

            with gr.Column():
                generated_image = gr.Image(label="Generated Image")
                processed_image = gr.Image(label="Processed Image")

        with gr.Row():
            with gr.Column():
                with gr.Tab("OBJ"):
                    output_model_obj = gr.Model3D(label="3D Model (OBJ)")
                    gr.Markdown("Note: The model shown here is flipped. Download to get correct results.")

            with gr.Column():
                with gr.Tab("GLB"):
                    output_model_glb = gr.Model3D(label="3D Model (GLB)")
                    gr.Markdown("Note: The model shown here has a darker appearance. Download to get correct results.")

        submit.click(
            fn=text_to_3d_pipeline,
            inputs=[
                prompt,
                ckpt,
                do_remove_background,
                foreground_ratio,
                mc_resolution
            ],
            outputs=[
                generated_image,
                processed_image,
                output_model_obj,
                output_model_glb
            ]
        )

        return interface


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--username", type=str, default=None)
    parser.add_argument("--password", type=str, default=None)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--listen", action="store_true")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--queuesize", type=int, default=1)

    args = parser.parse_args()

    # Start CUDA worker process
    start_worker()

    # Import gradio here (after worker started) to avoid interference
    import gradio as gr

    # Create and launch interface
    interface = create_interface()
    interface.queue(max_size=args.queuesize)
    interface.launch(
        auth=(args.username, args.password) if (args.username and args.password) else None,
        share=args.share,
        server_name="0.0.0.0" if args.listen else None,
        server_port=args.port
    )
