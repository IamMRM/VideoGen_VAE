import torch
from diffusers import AutoPipelineForImageToVideo
from PIL import Image

# Load LTX-Video pipeline
pipe = AutoPipelineForImageToVideo.from_pretrained(
    "Lightricks/LTX-Video",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

# Prepare input image (resize to 256x256 if needed)
input_image = Image.open("your_image.jpg").resize((256, 256))

# Generate video (65 frames by default)
generated_frames = pipe(
    input_image,
    num_frames=65,  # LTX-Video works best with 8N+1 frames (65, 129, 257)
    num_inference_steps=30,
    guidance_scale=7.0,
).frames[0]

# Save frames as video (requires additional libraries like OpenCV)
# Or use LTX-Video's built-in utils:
from ltx_video.utils import save_video
save_video(generated_frames, "output_video.mp4", fps=10)