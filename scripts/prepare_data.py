# scripts/prepare_data.py
import json
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

DATA_DIR = "../data/anky_genesis_collection"

class AnkyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.data_dir, image_file)

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image

def prepare_dataloader(batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = AnkyDataset(data_dir=DATA_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

if __name__ == "__main__":
    dataloader = prepare_dataloader()
    for images in dataloader:
        print(images.shape)
        break