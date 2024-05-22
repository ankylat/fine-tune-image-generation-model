# scripts/generate_images.py

from diffusers import StableDiffusionPipeline
import torch

def generate_image(prompt, model_id="fine-tuned-stable-diffusion"):
    device = "cuda"
    pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipeline = pipeline.to(device)

    image = pipeline(prompt).images[0]

    # Save or display the generated image
    output_path = "generated_image.png"
    image.save(output_path)
    print(f"Generated image saved to {output_path}")

if __name__ == "__main__":
    prompt = "A character from Anky Genesis collection"
    generate_image(prompt)
