from diffusers import LTXVideoTransformer3DModel, AutoencoderKLLTXVideo, LTXImageToVideoPipeline, FlowMatchEulerDiscreteScheduler
from transformers import T5EncoderModel, T5Tokenizer
import torch


save_directory = "./ltx_video_model"

transformer = LTXVideoTransformer3DModel.from_pretrained(
    "Lightricks/LTX-Video", subfolder="transformer", torch_dtype=torch.bfloat16
)
transformer.save_pretrained(f"{save_directory}/transformer")

vae = AutoencoderKLLTXVideo.from_pretrained(
    "Lightricks/LTX-Video", subfolder="vae", torch_dtype=torch.bfloat16
)
vae.save_pretrained(f"{save_directory}/vae")

text_encoder = T5EncoderModel.from_pretrained(
    "Lightricks/LTX-Video", subfolder="text_encoder", torch_dtype=torch.bfloat16
)
text_encoder.save_pretrained(f"{save_directory}/text_encoder")

tokenizer = T5Tokenizer.from_pretrained(
    "Lightricks/LTX-Video", subfolder="tokenizer"
)
tokenizer.save_pretrained(f"{save_directory}/tokenizer")

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    "Lightricks/LTX-Video", subfolder="scheduler"
)
scheduler.save_pretrained(f"{save_directory}/scheduler")
