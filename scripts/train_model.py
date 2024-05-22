import os
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from torch.utils.data import DataLoader
from torchvision import transforms
from prepare_data import AnkyDataset

def fine_tune_model(data_dir, model_id="runwayml/stable-diffusion-v1-5", checkpoint_path=None, num_epochs=5, batch_size=8, lr=1e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = AnkyDataset(data_dir=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load models and tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")

    # Load the checkpoint
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        unet.load_state_dict(checkpoint["state_dict"], strict=False)

    vae = StableDiffusionPipeline.from_pretrained(model_id).vae
    text_encoder = torch.nn.DataParallel(text_encoder)
    unet = torch.nn.DataParallel(unet)
    vae = torch.nn.DataParallel(vae)

    text_encoder.to(device)
    unet.to(device)
    vae.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        for step, images in enumerate(dataloader):
            images = images.to(device, dtype=torch.float32)

            with torch.cuda.amp.autocast():
                latents = vae.module.encode(images).latent_dist.sample() * 0.18215
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, 1000, (images.size(0),), device=device).long()
                noisy_latents = latents + noise

                dummy_hidden_states = torch.zeros(images.size(0), 77, 768).to(device)
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=dummy_hidden_states)
                noise_pred = noise_pred.sample if hasattr(noise_pred, 'sample') else noise_pred

                if isinstance(noise_pred, (list, tuple)):
                    noise_pred = torch.stack(noise_pred)
                elif isinstance(noise_pred, map):
                    noise_pred = torch.stack(list(noise_pred))
                elif not isinstance(noise_pred, torch.Tensor):
                    noise_pred = torch.tensor(noise_pred, device=device)

                loss = torch.nn.functional.mse_loss(noise_pred, noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Free up GPU memory
            del images, latents, noise, noisy_latents, noise_pred
            torch.cuda.empty_cache()

            if step % 10 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Step [{step}/{len(dataloader)}], Loss: {loss.item()}")

    # Save the fine-tuned model
    output_dir = "fine-tuned-stable-diffusion"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    unet.module.save_pretrained(output_dir)
    text_encoder.module.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model fine-tuned and saved to {output_dir}")

if __name__ == "__main__":
    data_dir = '../data/anky_genesis_collection'
    checkpoint_path = '../models/stable-diffusion-v1-5/v1-5-pruned.ckpt'  # Adjust if needed
    fine_tune_model(data_dir, checkpoint_path=checkpoint_path)
