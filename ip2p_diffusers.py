import os
import numpy as np
import random
from tqdm import tqdm
from argparse import ArgumentParser

from PIL import Image, ImageOps
import requests
import torch
import torchvision.transforms as transforms

from diffusers import StableDiffusionInstructPix2PixPipeline

def main():
    parser = ArgumentParser()
    parser.add_argument('--coco-image-paths', type=str, default=None,
                        help="""File with path of images to be edited.""")
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--cfg-text", default=7.5, type=float)
    parser.add_argument("--cfg-image", default=1.5, type=float)
    parser.add_argument("--edits-file", required=True, type=str,
                        help="""File with edit instructions to perform on coco images.""")
    parser.add_argument("--out-dir", required=True, type=str,
                        help="""Parent directory of folder to store edited images.""")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)
    print("Count gpus: ", torch.cuda.device_count())

    # Creating directory to save images
    save_dir = f"{args.out_dir}/ip2p-coco/steps{args.steps}_text{str(args.cfg_text).replace('.','')}_img{str(args.cfg_image).replace('.','')}"
    os.makedirs(save_dir, exist_ok=True)
    print("Edited images will be saved to", save_dir)
    
    # List of COCO images to be edited
    paths_file = open(f"{args.coco_image_paths}", "r")
    coco_img_paths = [path.rstrip() for path in paths_file]
    print("Number of COCO images to be edited:", len(coco_img_paths))

    # Selecting a random subset of COCO images
    sample_coco_img_paths = random.sample(coco_img_paths, 50)
    print(f"Only a subset of {len(sample_coco_img_paths)} images will be edited") 

    # List of image edits
    edits_file = open(f"{args.edits_file}", "r")
    edits = [line.rstrip() for line in edits_file]
    print("Number of image edit instructions:", len(edits))

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        safety_checker=None,
        requires_safety_checker=False,
        torch_dtype=torch.float16
    ).to(device)

    for j, img_path in enumerate(sample_coco_img_paths):

        print(f"Image {j+1}/{len(sample_coco_img_paths)}")
        img_id = os.path.splitext(os.path.split(img_path)[-1])[0]

        image = Image.open(img_path).convert("RGB").resize((512, 512))

        edit_images = pipe(
                prompt=edits,
                image=image,
                num_inference_steps=args.steps,
                guidance_scale=args.cfg_text,
                image_guidance_scale=args.cfg_image
                )
        for i, img in enumerate(edit_images.images):
            img.save(os.path.join(save_dir, img_id + f"_edit{i:02}.jpg"))

if __name__ == "__main__":
    main()


"""
image1 = Image.open("/home/mlfarinha/coco/images/train2017/000000393223.jpg").convert("RGB").resize((512, 512))
image2 = Image.open("/home/mlfarinha/coco/images/train2017/000000393224.jpg").convert("RGB").resize((512, 512))

transform = transforms.Compose([
    transforms.PILToTensor()
])

image1_tensor = transform(image1)
print("image1.shape:", image1.size)
image2_tensor = transform(image2)
print("image2.shape:", image2.size)

batch_images = torch.cat((torch.unsqueeze(image1_tensor, 0), torch.unsqueeze(image2_tensor, 0)), dim=0)
print("batch_images.shape:", batch_images.shape)

pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        safety_checker=None,
        requires_safety_checker=False,
        torch_dtype=torch.float16
        ).to(device)

prompts = ["make the person look more feminine", "make this person feminine", "increase the femininity of this person", "make the individual have a feminine appearance", "make the person look more masculine", "make the person masculine", "increase the masculinity of this person", "make the individual have a masculine appearance", "change the look of this individual", "modify the looks of this person", "give this person a different look"]
edit_images = pipe(prompt=prompts, image=image1, num_inference_steps=1000, guidance_scale=8.8)
print("edit_images.images.shape:", len(edit_images.images))
print("image1_edit1.shape:", edit_images.images[0].size)
print("image1_edit2.shape:", edit_images.images[1].size)
"""

