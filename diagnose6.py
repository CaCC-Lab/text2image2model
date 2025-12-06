"""診断スクリプト6: Gradioインポートの影響を調査"""
import torch
import time

print("=" * 60)
print("TEST: Effect of Gradio import on CUDA performance")
print("=" * 60)

def run_inference(pipe):
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        result = pipe("a cat", num_inference_steps=4, guidance_scale=0)
    torch.cuda.synchronize()
    return time.time() - t0

# Test 1: Before Gradio import
print("\n[1] Loading pipeline BEFORE Gradio import...")
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

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

print("[2] Running inference BEFORE Gradio import...")
t1 = run_inference(pipe)
print(f"    Time: {t1:.2f}s")

t2 = run_inference(pipe)
print(f"    Time (2nd): {t2:.2f}s")

# Import Gradio
print("\n[3] Importing Gradio...")
import gradio as gr
print(f"    Gradio version: {gr.__version__}")

# Test 2: After Gradio import
print("\n[4] Running inference AFTER Gradio import...")
t3 = run_inference(pipe)
print(f"    Time: {t3:.2f}s")

t4 = run_inference(pipe)
print(f"    Time (2nd): {t4:.2f}s")

# Test 3: Create Blocks but don't launch
print("\n[5] Creating Gradio Blocks (not launching)...")
with gr.Blocks() as demo:
    gr.Markdown("Test")

print("\n[6] Running inference AFTER Blocks creation...")
t5 = run_inference(pipe)
print(f"    Time: {t5:.2f}s")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Before Gradio import:  {t1:.2f}s, {t2:.2f}s")
print(f"After Gradio import:   {t3:.2f}s, {t4:.2f}s")
print(f"After Blocks creation: {t5:.2f}s")

if t3 > t1 * 5:
    print("\n⚠ Gradio import significantly slows down CUDA!")
elif t5 > t3 * 5:
    print("\n⚠ Gradio Blocks creation significantly slows down CUDA!")
else:
    print("\n✓ Gradio import/Blocks don't affect performance directly")
    print("  Problem must be in launch() or event handling")
