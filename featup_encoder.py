import torch
import torchvision.transforms as T
from PIL import Image

from featup.util import norm, unnorm
import os

import argparse
from tqdm import tqdm



input_size = 224
image_path = "test.jpg"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_norm = True


transform = T.Compose([
    T.Resize((224,288)),
    T.ToTensor(),
    norm
])

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', help="Colmap image files", type=str, required=True)
    parser.add_argument('--output_feature_folder', help="Output feature folder location, everything is saved as .pt", type=str, required=True)
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parser()

    image_folder = args.image_folder
    feature_folder = args.output_feature_folder
    allowed_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif")


    os.makedirs(feature_folder, exist_ok=True)
    images = [
        f for f in os.listdir(image_folder)
        if f.lower().endswith(allowed_extensions)
    ]
    upsampler = torch.hub.load("mhamilton723/FeatUp", 'maskclip', use_norm=False).to(device)

    for image in tqdm(images):
        path = os.path.join(image_folder, image)
        image_tensor = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)

        hr_feats = upsampler(image_tensor)
        lr_feats = upsampler.model(image_tensor)
        torch.save(hr_feats, os.path.join(feature_folder, image.split('.')[0]+'.pt'))
