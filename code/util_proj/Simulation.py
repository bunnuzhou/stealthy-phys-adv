import os
from torch import nn
import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
from nose import tools

class Image_Synthsis(nn.Module):
    '''
    Author: bunnuzhou
    description: this class is used for image simulation
    usage: func imageProjected 
    '''
    def __init__(self):
        super(Image_Synthsis, self).__init__()

    def __constIntensity(self, img_amb, img_full, img_proj, amb_luv, full_luv, proj_luv):
        '''
        *** deprected *** unused version of simulation 
        description: systhesis the image while hypothesising camera asking for a const light intensity
        param {*} self
        param {*} img_amb   image without projection 
        param {*} img_full  image with full illumination white light projection
        param {*} img_proj  image to be projected 
        param {*} amb_luv   ambient luv
        param {*} full_luv  full_projectable_luv
        param {*} proj_luv  project_luv
        return {*}image          image illuminated in project_luv
        '''        
        first_part = amb_luv * torch.pow(img_amb, 2.2) / (amb_luv + proj_luv)
        second_part = (proj_luv / full_luv) * (1 / (amb_luv + proj_luv)) * \
            img_proj.mul(((amb_luv + full_luv) * torch.pow(img_full, 2.2) - amb_luv * torch.pow(img_amb, 2.2)))
        image = first_part + second_part
        image = pow(image, 1/ 2.2)
        image = torch.clamp(image, 0, 1)
        return image
    
    def __gammaAugment(self, img_amb, img_full, img_proj):
        '''
        description: sysythsis the image while hypothesising the camera is using a gamma augmentation enhancing light
        param {*} self
        param {*} img_amb   image without projection 
        param {*} img_full  image with full illumination white light projection
        param {*} img_proj  image to be projected 
        return {*}image
        '''        
        # img_proj = img_proj ** (1 / 2.2)
        # img_full = transforms.RandomRotation((-1, 1), interpolation=transforms.InterpolationMode.BILINEAR)(img_full)
        img_proj_extra = torch.clamp(img_proj.mul((torch.pow(img_full, 2.2)-torch.pow(img_amb, 2.2))), 0, 1)
        image = torch.pow(torch.clamp(1e-6 + torch.pow(img_amb, 2.2) + img_proj_extra, 0, 1), 1 / 2.2)
        image = torch.clamp(image, 0, 1)
        
        save_path = './tmp'
        from datetime import datetime
        pil_img_net_input = transforms.ToPILImage()(image[0])
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        img_name = f'result_{timestamp}.png'
        pil_img_net_input.save(os.path.join(save_path,img_name))
        return image

    def imageProjected(self, img_amb, img_full, img_proj):
        '''
        description: image systhesis simulation
        param {*} self
        param {*} img_amb   image without projection 
        param {*} img_full  image with full illumination white light projection
        param {*} img_proj  image to be projected 
        return {*}image
        '''            
        # if is_constIntensity:
        #     tools.assert_is_not_none(amb_luv)
        #     tools.assert_is_not_none(full_luv)
        #     tools.assert_is_not_none(proj_luv)
        #     return self.__constIntensity(img_amb, img_full, img_proj, amb_luv, full_luv, proj_luv)
        # else:
        return self.__gammaAugment(img_amb, img_full, img_proj)


