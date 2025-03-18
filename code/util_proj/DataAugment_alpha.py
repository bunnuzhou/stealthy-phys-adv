from tokenize import group
from unicodedata import decimal
import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import torchvision.transforms.functional as TF
from torch.nn import functional as F
from PIL import Image
import time 
import numpy as np
import matplotlib.pyplot as plt
import random
import math


from torchvision.transforms.functional import InterpolationMode, convert_image_dtype

# segmentation stripe
class SegImageGenerate(nn.Module):
    '''
    usage :use func  generate_image to generate a segmented image 
    '''

    def __init__(self):
        super(SegImageGenerate, self).__init__()
        self.eps = 1e-8


    def generate_image(self, img_pattern, img_inv_pattern , slope=0.001, num_lines=1):
        '''
        img_pattern: ambient light condition pics
        img_inv_pattern: full white light projected pics
        input img form : (1, C, H, W); 
        img size should be square and in Yxy not rgb form
        slope: segment line slope
        num_lines: number of segment lines
        '''
        image_size = img_pattern.shape[2]
        device = img_pattern.device

        # generate image with segment lines
        # start = random.randint(- int(image_size / (abs(slope) + 1e-6)), int(image_size))
        start = random.uniform(-1, 1)
        strip_width = int(image_size  / (num_lines)) + 1e-6
        vertical_len = int(strip_width / (np.power(1 + slope * slope, 0.5) + self.eps))
        vague_halfwidth = random.randint(int(vertical_len / 6), int(vertical_len/5))

        # theta = torch.tensor([
        #         [1/np.sqrt(slope**2+1), 1/np.sqrt(slope**2+1), start],
        #         [1/np.sqrt(slope**2+1), 0, 0]
        #     ], dtype = torch.float).to(device)
        theta = torch.tensor([
                [0, 1, start],
                [1, 0, 0]
            ], dtype = torch.float).to(device)

        mask = torch.ones((image_size, image_size), dtype=torch.float).to(device)
        grid = F.affine_grid(theta.unsqueeze(0), mask.unsqueeze(0).unsqueeze(0).size(), align_corners=True)
        mask = F.grid_sample(mask.unsqueeze(0).unsqueeze(0), grid, align_corners=True)[0][0]

        roll = random.randint(0, 1)
        if roll == 0:
            mask = torch.logical_xor(mask, torch.ones((image_size, image_size), dtype=torch.float).to(device)).float()


        
        # torch image in C, H, W arrange
        # for i in range(mask.shape[0]):
        #     for j in range(mask.shape[1]):
        #         n = math.floor(( j +i / slope - start) / strip_width)
        #         if n % 2 == 0:
        #             mask[i][j] = 1
        #         if  np.abs( i + slope * j - slope * (start + (n+1) * strip_width)) <= vague_halfwidth * np.power(1 + slope * slope, 0.5) :
        #             if (n % 2 == 1 and slope > 0) or (n % 2 == 0 and slope < 0):
        #                 mask[i][j] = (((i + slope * j - slope * (start + (n+1) * strip_width)
        #                                ) / (vague_halfwidth * np.power(1 + slope * slope, 0.5))) + 1) * 0.5 
        #             else:
        #                 mask[i][j] = ( -(i + slope * j - slope * (start + (n+1) * strip_width)
        #                                ) / ((vague_halfwidth * np.power(1 + slope * slope, 0.5))) + 1) * 0.5 
        #         elif np.abs( i + slope * j - slope * (start + (n) * strip_width)) <= vague_halfwidth * np.power(1 + slope * slope, 0.5):
        #             if (n % 2 == 1 and slope > 0) or (n % 2 == 0 and slope < 0):
        #                 mask[i][j] = (  -(i + slope * j - slope * (start + (n) * strip_width)
        #                                  ) / ((vague_halfwidth * np.power(1 + slope * slope, 0.5) )) + 1) * 0.5 
        #             else:
        #                 mask[i][j] = ((i + slope * j - slope * (start + (n) * strip_width)
        #                                ) / ((vague_halfwidth * np.power(1 + slope * slope, 0.5) )) + 1) * 0.5 

        # mask = transforms.RandomAffine(0, (0.5,0), None, 45)(mask.unsqueeze(0))
        mask = torch.nn.functional.conv2d((mask.unsqueeze(0)).unsqueeze(0), weight=(torch.ones(1,1, vague_halfwidth, vague_halfwidth).to(device))/(vague_halfwidth*vague_halfwidth),\
             stride=(1,1), groups=1, padding = 'same' )[0][0]
        
        img_vague = mask * img_inv_pattern + (1 - mask) * img_pattern
        return img_vague

    def rainbow_effect(self, img_input, intensity = 0.9):
        '''
        the rainbow effect pattern is GB(empty)GR(empty), 6 slots in total but only 4 of 6 are lit.
        the rainbow effect is the uneven distribution of RGB light, which is not scattered uniformly but somewhere is brighter 
                    while the left areas are darker.
        img: XYZ format image, [B, C, H, W]
        intensity: the intensity is the value of dark area, which is bound to be smaller than 1
        '''
        slot_num = 5
        intensity_extra_rb = slot_num * (1 - intensity)
        intensity_extra_g = intensity_extra_rb / 2 
        # band_type = torch.diag(torch.ones(3).to(img_input.device) * intensity_extra_rb)
        band_type =  torch.tensor([[intensity_extra_rb, 0, 0], 
                                     [0, intensity_extra_g, 0], 
                                     [0, 0, intensity_extra_rb]]).to(img_input.device)
        rand_idx = random.randint(0,3)
        if rand_idx == 3:
            rand_idx = 1
        weight_matrix_1 = torch.diag(torch.ones(3)).to(img_input.device) * intensity + torch.diag(band_type[rand_idx])
        weight_matrix_2 = torch.diag(torch.ones(3)).to(img_input.device) * intensity + torch.diag(band_type[(rand_idx + 1) % 3])
        random_band_1 = torch.nn.functional.conv2d(img_input, weight_matrix_1.reshape(3, 3, 1, 1), stride=1)
        random_band_2 = torch.nn.functional.conv2d(img_input, weight_matrix_2.reshape(3, 3, 1, 1), stride=1)
        random_band = self.generate_image(random_band_1, random_band_2)
        return random_band
        
    
        


#  augment and paste to Background
def img_bg_blend(img_sign, img_background, args):
    '''
    paste the sign images onto background images
    iuput image resized to image_size_range
    input form [B, C, H, W]
    '''

    img_sign = transforms.RandomRotation((-10, 10), InterpolationMode.BILINEAR)(img_sign)
    num_signs = img_sign.shape[0]
    num_background = img_background.shape[0]
    lower, upper = args.sign_size_range
    new_size = random.randint(lower, upper)
    Resize = transforms.Resize((new_size, new_size), interpolation=InterpolationMode.BILINEAR)
    img_sign = Resize(img_sign)
    x_pos = random.randint(0, img_background.shape[3]-new_size)
    y_pos = random.randint(0, img_background.shape[2]-new_size)
    padding = (x_pos, y_pos, img_background.shape[3] - x_pos - new_size, img_background.shape[2] - y_pos -new_size)
    padding_transform = transforms.Pad(padding)
    img_input = padding_transform(img_sign)
    img_input = img_input.repeat((num_background, 1, 1, 1))
    img_background = img_background.repeat((num_signs, 1, 1, 1))
    mask = img_input[:, 3, :, :].unsqueeze(1)
    img_input_rgb = img_input.narrow(1, 0, 3)
    image = mask * (img_input_rgb) + (1 - mask) * img_background
    return image, torch.tensor([x_pos, y_pos, x_pos + new_size, y_pos + new_size]).view(-1, 4).to(img_sign.device)
    
    
    
    
    
    
    
    if not 'img_ret' in dir():
        img_ret = __img_bg_blend(img_sign, img_background[j], args.sign_size_range)
    else:
        img_ret = torch.cat((img_ret, __img_bg_blend(img_sign, img_background[j], args.sign_size_range)), dim=0)
    
    return img_ret

def __img_bg_blend(img_input, img_background, image_size_range):
    '''
    called by fun img_bg_blend
    input form [C, H, W]
    '''
    device = img_input.device
    lower, upper = image_size_range

    # scale and paste
    new_size = random.randint(lower, upper)
    Resize = transforms.Resize((new_size, new_size), interpolation=InterpolationMode.NEAREST)
    img_input = Resize(img_input)
    x_pos = random.randint(0, img_background.shape[2]-new_size)
    y_pos = random.randint(0, img_background.shape[1]-new_size)

    padding = (x_pos, y_pos, img_background.shape[1] - x_pos - new_size, img_background.shape[2] - y_pos -new_size)
    padding_transform = transforms.Pad(padding)
    img_input = padding_transform(img_input)
    mask_form = img_input[3]
    img_input = img_input[0:3]
    image = mask_form * (img_input - 0.1) + (1 - mask_form) * img_background

    return image.unsqueeze(0)

