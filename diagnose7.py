"""診断スクリプト7: launch()後の影響を調査"""
import torch
import time
import threading

print("=" * 60)
print("TEST: Effect of Gradio launch() on CUDA performance")
print("=" * 60)

# Load pipeline first
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import gradio as gr

def run_inference(pipe, name=""):
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        result = pipe("a cat", num_inference_steps=4, guidance_scale=0)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"    [{name}] Time: {elapsed:.2f}s")
    return elapsed

print("\n[1] Loading pipeline...")
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

print("\n[2] Test BEFORE launch...")
t_before = run_inference(pipe, "before launch")

# Create interface
def generate(prompt):
    return run_inference(pipe, "via Gradio")

with gr.Blocks() as demo:
    prompt = gr.Textbox(value="a cat")
    btn = gr.Button("Generate")
    img = gr.Image()
    btn.click(fn=generate, inputs=prompt, outputs=img)

# Launch in background thread
print("\n[3] Launching Gradio in background...")
server_thread = threading.Thread(
    target=lambda: demo.launch(
        server_name="127.0.0.1",
        server_port=7863,
        prevent_thread_lock=True,
        show_error=True
    ),
    daemon=True
)
server_thread.start()
time.sleep(3)  # Wait for server to start

print("\n[4] Test AFTER launch (from main thread)...")
t_after1 = run_inference(pipe, "after launch, main thread")
t_after2 = run_inference(pipe, "after launch, main thread 2nd")

print("\n[5] Now test from browser at http://127.0.0.1:7863")
print("    Press Enter after testing from browser...")
input()

print("\n[6] Final test from main thread...")
t_final = run_inference(pipe, "final main thread")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Before launch:              {t_before:.2f}s")
print(f"After launch (main thread): {t_after1:.2f}s, {t_after2:.2f}s")
print(f"Final (main thread):        {t_final:.2f}s")
