"""診断スクリプト5: Gradio queue無効化テスト"""
import torch
import time
import gradio as gr
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

print("=" * 60)
print("TEST: Gradio WITHOUT queue")
print("=" * 60)

device = "cuda:0"

# Load pipeline
print("\n[Loading SDXL...]")
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

def generate(prompt):
    print(f"\n[Generate: {prompt}]")
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        result = pipe(prompt, num_inference_steps=4, guidance_scale=0)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"[Time: {elapsed:.2f}s]")
    return result.images[0]

# Create interface WITHOUT queue
with gr.Blocks() as demo:
    gr.Markdown("# Test WITHOUT queue")
    prompt = gr.Textbox(label="Prompt", value="a cat")
    btn = gr.Button("Generate")
    img = gr.Image(label="Result")
    btn.click(fn=generate, inputs=prompt, outputs=img)

# Launch WITHOUT queue
print("\nLaunching Gradio WITHOUT queue...")
demo.launch(server_name="127.0.0.1", server_port=7862)
