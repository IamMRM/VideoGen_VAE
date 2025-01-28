import os
import tensorflow as tf
import torch
from diffusers import LTXVideoTransformer3DModel, AutoencoderKLLTXVideo, LTXImageToVideoPipeline, FlowMatchEulerDiscreteScheduler
from transformers import T5EncoderModel, T5Tokenizer
from data_pipeline import TFRecordVideoDataset
from torch.utils.data import DataLoader
from typing import List, Dict, Any
import numpy as np
from itertools import chain

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = "dataset_cleaned"
    output_dir = "ltx_video_finetuned"
    batch_size=2
    num_workers=2
    learning_rate=3e-4
    num_epochs=50
    os.makedirs(output_dir, exist_ok=True)
    save_directory = "./ltx_video_model"

    transformer = LTXVideoTransformer3DModel.from_pretrained(
        f"{save_directory}/transformer", torch_dtype=torch.bfloat16)
    vae = AutoencoderKLLTXVideo.from_pretrained(
        f"{save_directory}/vae", torch_dtype=torch.bfloat16)
    transformer.to(device)
    vae.to(device)
    """text_encoder = T5EncoderModel.from_pretrained(
        f"{save_directory}/text_encoder", torch_dtype=torch.bfloat16)
    tokenizer = T5Tokenizer.from_pretrained(f"{save_directory}/tokenizer")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(f"{save_directory}/scheduler")

    pipeline = LTXImageToVideoPipeline(  # use LTXPipeline for just text to video generation
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler
    )
    model = pipeline.to(device)"""
    #from PIL import Image
    #width, height = 256, 256
    #random_image_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    #random_image = Image.fromarray(random_image_array)
    #prompt = "something cool happening"
    #result = pipeline(prompt=prompt, image=random_image, )
    #video_frames = result.frames
    #from diffusers.utils import export_to_video
    #export_to_video(video_frames, "output.mp4", fps=24)


    #LTX_VIDEO training
    dataset = TFRecordVideoDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    optimizer = torch.optim.AdamW(
        chain(transformer.parameters(), vae.parameters()), lr=learning_rate
    )
    transformer.train()
    vae.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)

            transformer_outputs = transformer(pixel_values=pixel_values)
            transformer_loss = transformer_outputs.loss

            vae_outputs = vae(pixel_values=pixel_values)
            vae_loss = vae_outputs.loss


            loss = transformer_loss + vae_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                chain(transformer.parameters(), vae.parameters()), max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item()

            # Print epoch statistics
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch + 1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
            # Save the optimizer state
            torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, 'optimizer.pt'))