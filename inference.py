import torch
from diffusers import LTXPipeline
from peft import PeftModel
from diffusers.utils import export_to_video
from torchvision.transforms import ToPILImage
import os
import numpy as np
import gc

print(f"Available GPUs: {torch.cuda.device_count()}")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
base_model = LTXPipeline.from_pretrained(
    "diffusion-pipe/ltx_video_model/",#"Lightricks/LTX-Video",
    torch_dtype=torch.float16,#torch.bfloat16,
    use_safetensors=True,
    device_map="balanced"
)

 # Follow prompt guidelines:cite[4]
prompt = "A detailed cinematic scene:A woman with long brown hair and light skin smiles at another woman with long blonde hair. The camera angle is a close-up, focused on the woman's face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage."
generation_params = {
    "prompt": prompt,
    "negative_prompt": "low quality, blurry, distorted",
    "height": 512,       # Must be divisible by 32
    "width": 512,        # Must be divisible by 32
    "num_frames": 257,#1025,#257,   # 8*32 +1 = 257
    "num_inference_steps": 50,
    "guidance_scale": 7.5, # Higher guidance improves quality
    "decode_timestep": 0.03,
    "decode_noise_scale": 0.025
}
torch.cuda.empty_cache()
torch.cuda.synchronize()
gc.collect()

with torch.inference_mode():
    output = base_model(**generation_params)
    video_frames = output.frames[0]
#del output
torch.cuda.empty_cache()
torch.cuda.synchronize()
gc.collect()

converted_frames = [
    np.array(frame.convert('RGB')) / 255.0 for frame in video_frames
]
export_to_video(
    converted_frames,
    "generated_video_optimized.mp4",
    fps=24,
)
