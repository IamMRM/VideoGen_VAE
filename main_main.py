import os
#import tensorflow as tf
import torch
import torch.optim as optim
from diffusers import LTXVideoTransformer3DModel, AutoencoderKLLTXVideo, LTXImageToVideoPipeline, FlowMatchEulerDiscreteScheduler
from video_diffusion_pytorch import Unet3D, GaussianDiffusion, Trainer
from transformers import T5EncoderModel, T5Tokenizer
from data_pipeline import VideoDataGenerator
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer, CLIPTextModel
from typing import List, Dict, Any
import numpy as np
from itertools import chain



def encode_text(text: str, batch_size: int):
    texts = [text] * batch_size
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = text_encoder(**inputs)
    return outputs.last_hidden_state[:, -1, :]

if __name__ == "__main__":
    device = "cpu"#torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    dataset_path = "/dataset/dataset_mp4_cleaned"
    output_dir = "output"
    batch_size = 1
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    # Freeze text encoder to save memory
    for param in text_encoder.parameters():
        param.requires_grad = False

    num_workers = min(8, os.cpu_count())
    num_epochs = 5
    image_size = 128
    os.makedirs(output_dir, exist_ok=True)

    dataset = VideoDataGenerator(dataset_path, height_or_width_pixels=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # adjust num_workers if needed

    model = Unet3D(dim=128, dim_mults=(1, 2, 4, 8)).to(device)
    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,
        num_frames=440,
        timesteps=1000,
        loss_type='l2'
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    use_amp = device == "cuda"
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            videos = batch['pixel_values'].to(device)
            text_embeds = encode_text("chatgpot is great", batch_size)
            if use_amp:
                with torch.cuda.amp.autocast():
                    loss = diffusion(videos, cond=text_embeds)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = diffusion(videos, cond=text_embeds)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch + 1}.pth")

        model.eval()
        with torch.no_grad():
            sample_text_embeds = encode_text("chatgpot is great", 1)
            generated_video = diffusion.sample(cond=sample_text_embeds, cond_scale=2.0)
            torch.save(generated_video, f"generated_epoch_{epoch + 1}.pt")