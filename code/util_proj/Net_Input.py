import argparse
from argparse import ArgumentParser
import torch
import torchvision
from torch import nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import time
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from RgbYxy import *
from DataAugment_alpha import  * 
from Simulation import *

def Net_Input(img_amb, img_full, img_pattern, img_background, mask, args):
    '''
    Description: producing input pics for object detecting network
    Args:
        img_amb: one pic of stop sign in ambient light 
        img_full: one pic of stop sign in full illumination projection
        img_pattern: one pic of adversary example generated
        img_background: batchs of background image 
        args: must have 'nums_projection', 'sign_size_range' options
    Return:
        batchs of image 
    '''
    RgbYxyConvertor = RgbYxy()
    segmentation = SegImageGenerate()
    simulation = Image_Synthsis()

    ##############################################

    img_pattern = RgbYxyConvertor.rgb_to_Yxy(img_pattern.unsqueeze(0))
    img_inv_pattern = color_inv_Img(img_pattern, args.averaged_rgb)

    # generate segment projection pic 
    for i in range(args.nums_projection):
        if not 'img_simulation_input' in dir():
            img_simulation_input = segmentation.generate_image(img_pattern, img_inv_pattern, random.uniform(0.5, 6) * (1 if random.random() < 0.5 else -1), random.randint(0, 2))
        else:
            img_simulation_input = torch.cat((img_simulation_input, segmentation.generate_image(
                img_pattern, img_inv_pattern, random.uniform(0, 6), random.randint(1, 2))), dim=0)

    img_simulation_input = RgbYxyConvertor.Yxy_to_rgb(img_simulation_input)


    plt.figure()
    plt.imshow(img_simulation_input[0].detach(
    ).cpu().numpy().transpose(1, 2, 0))
    plt.show()


    # projection 
    # RANDOM CROP TO ORIGINAL SIZE
    image_size = img_amb.shape[1]
    i, j, h, w = transforms.RandomCrop.get_params(img_simulation_input, output_size=(image_size, image_size))
    img_simulation_input = TF.crop(img_simulation_input, i, j, h, w)
    img_background_input = simulation.imageProjected(img_amb, img_full, img_simulation_input)

    plt.figure()
    plt.imshow(img_background_input[0].detach(
    ).cpu().numpy().transpose(1, 2, 0))
    plt.show()

    # used for black stripe remove
    img_background_input = img_background_input + 0.1

    mask = mask.repeat(img_background_input.shape[0], 1, 1, 1)
    img_background_input = torch.cat((img_background_input, mask), dim= 1)

    # perspective change
    img_background_input = transforms.RandomPerspective(0.3, interpolation=InterpolationMode.NEAREST)(img_background_input)
 
    # paste to background with rotation
    img_net_input = img_bg_blend(img_background_input, img_background, args)

    # # Average Blur
    # conv_size = 5
    # layer = nn.Conv2d(3, 3, conv_size, padding=int((conv_size - 1)/2), padding_mode='replicate', dtype=torch.float, bias=False, groups=3)
    # average_weight = (torch.ones((conv_size, conv_size), requires_grad=False, dtype=torch.float, device=img_amb.device) * 1 / (conv_size * conv_size))\
    #     .expand(3, 1, conv_size, conv_size)
    # average_weight = nn.Parameter(data=average_weight, requires_grad=False)
    # layer.weight.data = average_weight
    # # p = 0.5
    # if random.randint(0, 1) == 0:
    #     img_net_input = layer(img_net_input)

    # gaussian blur (noise)
    blurred = transforms.GaussianBlur((3, 3), sigma=(0.1, 0.5))
    img_net_input = blurred(img_net_input)

    img_net_input = torch.clamp(img_net_input, 0, 1)
    return img_net_input
