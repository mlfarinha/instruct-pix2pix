from __future__ import annotations

import math
import random
import sys
from argparse import ArgumentParser

import einops
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast

import os
import re
import csv
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import logging

from torch.utils.data import Dataset, DataLoader
import torchvision
from facenet_pytorch import MTCNN
from pycocotools.coco import COCO

sys.path.append("./stable_diffusion")

from stable_diffusion.ldm.util import instantiate_from_config


def COCODataset(coco_dataDir, coco_dataType, coco_face_detection_thres, device):

    instancesFile = '{}/coco/annotations/instances_{}.json'.format(coco_dataDir, coco_dataType)
    coco_instances = COCO(instancesFile)

    # Local folder containing COCO images
    cocoimgsDIR = '{}/coco/images/{}/'.format(coco_dataDir, coco_dataType)

    # Face detector to detect faces in COCO dataset
    mtcnn = MTCNN(device=device)

    # Fetch class IDs only corresponding to the filterClasses person
    catIds = coco_instances.getCatIds(catNms=["person"])
    imgIds = coco_instances.getImgIds(catIds=catIds)

    # List of image paths where face is detected
    images_path = list()

    for imgid in imgIds:
        img = coco_instances.loadImgs(imgid)[0]
        for root, dirs, files in os.walk(cocoimgsDIR):
            if img['file_name'] in files:

                img_path = os.path.join(root, img['file_name'])
                img_PIL = Image.open(img_path)
                
                try:
                    boxes, probs = mtcnn.detect(img_PIL) # try to detect face using face detector
                except:
                    print(f"Error occurred with mtcnn in image {img['file_name']}.")
                    pass

                if boxes is not None:
                    if len(boxes) == 1: # images with more than 2 faces may not work very well when swapping the face
                        if probs[0] > coco_face_detection_thres: # we only want images where we can clearly identify a face
                            images_path.append(img_path)
                        else:
                            print(f"Face of image {img['file_name']} detected with probability < {coco_face_detection_thres}.")
                    else:
                        print(f"Two or more faces detected in image {img['file_name']}.")
                else:
                    print(f"Face not detected in image {img['file_name']}.")
            else:
                print(f"Image {img['file_name']} not found.")

    return images_path

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)


def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model.") :]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model

def main():
    parser = ArgumentParser()
    parser.add_argument('--coco-dataDir', type=str, default=None,
                        help="""Parent directory where coco folder is located.""")
    parser.add_argument('--coco-dataType', type=str, default=None,
                        help="""Folder containing COCO images (found in ./coco/images/).""")
    parser.add_argument('--coco-face-detection-thres', type=float, default=0.98,
                        help="""MTCNN face detection threshold for COCO images.""")
    parser.add_argument('--img-paths-file', type=str, default=None,
                        help="""File with path to images to be edited.""")
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="configs/generate.yaml", type=str)
    parser.add_argument("--ckpt", default="checkpoints/instruct-pix2pix-00-22000.ckpt", type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)
    parser.add_argument("--out-dir", required=True, type=str,
                        help="""Parent directory of folder to store edited images.""")
    parser.add_argument("--edits-file", required=True, type=str,
                        help="""File with edit instructions to perform on coco images.""")
    parser.add_argument("--cfg-text", default=7.5, type=float)
    parser.add_argument("--cfg-image", default=1.5, type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)
    print("Count gpus: ", torch.cuda.device_count())

    print("Configuring model")
    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.eval().cuda()
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])

    # Creating directory to save images
    save_dir = f"{args.out_dir}/ip2p_coco_{args.resolution}res_{args.steps}steps_{str(args.cfg_text).replace('.','')}text_{str(args.cfg_image).replace('.','')}img"
    os.makedirs(save_dir, exist_ok=True)
    print("Edited images will be saved to ", save_dir)

    if args.coco_dataDir is not None and args.coco_dataType is not None and args.img_paths_file is None:
        print("Creating list of COCO image paths.")
        coco_img_paths = COCODataset(args.coco_dataDir, args.coco_dataType, args.coco_face_detection_thres, device)
        file = open(os.path.join(save_dir, "image_paths.txt"),"w")
        for path in coco_img_paths:
	        file.write(path+"\n")
        file.close()

    elif args.coco_dataDir is None and args.coco_dataType is None and args.img_paths_file is not None:
        file = open(f"{args.img_paths_file}", "r")
        coco_img_paths = [path.rstrip() for path in file]

    else:
        raise ValueError("please provide in the command line arguments either (coco_dataDir, coco_dataType) or img_paths_file.")

    # Length of dataset
    print("Number of COCO images to be edited: %i", len(coco_img_paths))

    for img_path in tqdm(coco_img_paths):

        img_id = os.path.splitext(os.path.split(img_path)[-1])[0]

        seed = random.randint(0, 100000) if args.seed is None else args.seed
        input_image = Image.open(img_path).convert("RGB")
        width, height = input_image.size
        factor = args.resolution / max(width, height)
        factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        width = int((width * factor) // 64) * 64
        height = int((height * factor) // 64) * 64
        input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

        if args.edits_file == "":
            input_image.save(args.output)
            return

        # Loop through the list of image edits
        edits_file = open(f"{args.edits_file}", "r")
        edits = [line.rstrip() for line in edits_file]
        for i, edit in enumerate(tqdm(edits)):

            with torch.no_grad(), autocast("cuda"), model.ema_scope():
                cond = {}
                cond["c_crossattn"] = [model.get_learned_conditioning([edit])]
                input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
                input_image = rearrange(input_image, "h w c -> 1 c h w").to(model.device)
                cond["c_concat"] = [model.encode_first_stage(input_image).mode()]

                uncond = {}
                uncond["c_crossattn"] = [null_token]
                uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

                sigmas = model_wrap.get_sigmas(args.steps)

                extra_args = {
                    "cond": cond,
                    "uncond": uncond,
                    "text_cfg_scale": args.cfg_text,
                    "image_cfg_scale": args.cfg_image,
                }
                torch.manual_seed(seed)
                z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
                z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
                x = model.decode_first_stage(z)
                x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
                x = 255.0 * rearrange(x, "1 c h w -> h w c")
                edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
            edited_image.save(os.path.join(save_dir, img_id + f"_edit{i}.jpg"))
            break


if __name__ == "__main__":
    main()
