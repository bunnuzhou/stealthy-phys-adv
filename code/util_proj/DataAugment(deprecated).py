import torch
from torch import nn
import torchvision.transforms as transforms
from PIL import Image
import time 
import numpy as np
import matplotlib.pyplot as plt
import random

from torchvision.transforms.functional import convert_image_dtype


class SegImageGenerate(nn.Module):
    '''
    usage :use func  generate_image to generate a segmented image 
    '''

    def __init__(self):
        super(SegImageGenerate, self).__init__()


    def generate_image(self, img_amb, img_full, slope=1, num_lines=1):
        '''
        img_amb: ambient light condition pics
        img_full: full white light projected pics
        input img form : (B, C, H, W); img size should be square and in hsv not rgb form
        slope: segment line slope
        num_lines: number of segment lines
        '''
        image_size = img_amb[0].shape[1]
        strip_width = int(image_size / num_lines)
        device = img_amb.device

        # generate image with segment lines
        start = random.randint(- image_size, 0)
        mask = torch.zeros((image_size, image_size), dtype=torch.bool).to(device)

        vague_halfwidth = random.randint(0, int(strip_width / 12))
        mask2 = torch.zeros((image_size, image_size), dtype=torch.bool).to(device)
        # torch image in C, H, W arrange
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                n = int(( j  + i / slope - start) / strip_width)
                if n % 2 == 0:
                    mask[i][j] = 1
                if  np.abs(i + slope * j - slope * (start + n * strip_width)) <= vague_halfwidth * np.power(1 + slope * slope, 0.5) or \
                        np.abs(i + slope * j - slope * (start + (n+1) * strip_width)) <= vague_halfwidth * np.power(1 + slope * slope, 0.5):
                    mask2[i][j] = 1
        mask = mask.repeat(len(img_amb), 3, 1, 1)
        img_seg = torch.where(mask, img_amb, img_full)

        # Gaussian filter for segment line 
        conv_size = 3 * 2 + 1
        layer = nn.Conv2d(3, 3, conv_size, padding=int((conv_size - 1)/2), padding_mode='replicate')
        average_weight = torch.full((conv_size, conv_size), 1 / (conv_size * conv_size), requires_grad=True, dtype=torch.float, device=device).unsqueeze(0)
        zero_weight = torch.zeros((conv_size, conv_size), requires_grad=True, dtype=torch.float, device=device).unsqueeze(0)
        layer.weight.data = torch.cat((torch.cat((average_weight, zero_weight, zero_weight), 0).unsqueeze(0), \
            torch.cat((zero_weight, average_weight, zero_weight), 0).unsqueeze(0), torch.cat((zero_weight, zero_weight, average_weight), 0).unsqueeze(0)), 0)
        layer.bias.data = torch.zeros(3, requires_grad=True, dtype=torch.float, device=device)
        img_vague = torch.where(mask2, layer(img_seg), img_seg)
        return img_vague