# scripts/train_model.py

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, PNDMScheduler
import torch
import os
from prepare_data import prepare_dataloader

def fine_tune_model(data_dir, model_id="runwayml/stable-diffusion-v1-5", checkpoint_path=None, num_epochs=5, batch_size=16, lr=1e-5):
    device = "cuda"

    # Prepare data
    dataloader = prepare_dataloader(batch_size)

    # Load models and tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")

    text_encoder = text_encoder.to(device)
    unet = unet.to(device)
    vae = vae.to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        text_encoder = torch.nn.DataParallel(text_encoder)
        unet = torch.nn.DataParallel(unet)
        vae = torch.nn.DataParallel(vae)
        vae_module = vae.module
    else:
        vae_module = vae

    # Set the model parameters to the same data type as the input tensor
    unet = unet.to(torch.float16)
    vae = vae.to(torch.float16)

    if checkpoint_path:
        # Load weights from checkpoint
        state_dict = torch.load(checkpoint_path, map_location="cuda")
        unet.load_state_dict(state_dict, strict=False)

    # Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        for step, images in enumerate(dataloader):
            # Prepare inputs
            images = images.to(device, dtype=torch.float16)

            # Forward pass
            latents = vae_module.encode(images).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (images.size(0),), device=device).long()
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            # Prepare text input
            text_input = torch.zeros(images.size(0), dtype=torch.long, device=device)
            attention_mask = torch.ones(1, 1, 8, 8, device=device)  # Create a dummy attention mask

            # Predict noise
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_encoder(text_input, attention_mask=attention_mask).last_hidden_state).sample

            # Compute loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Step [{step}/{len(dataloader)}], Loss: {loss.item()}")

    # Save the fine-tuned model
    output_dir = "fine-tuned-stable-diffusion"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    unet.save_pretrained(output_dir)
    text_encoder.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model fine-tuned and saved to {output_dir}")

if __name__ == "__main__":
    data_dir = '../data/anky_genesis_collection'
    checkpoint_path = "../models/stable-diffusion-v1-5/v1-5-pruned.ckpt"  # Adjust if needed
    fine_tune_model(data_dir, checkpoint_path=checkpoint_path)