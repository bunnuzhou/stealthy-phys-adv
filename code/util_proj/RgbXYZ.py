from numpy import dtype
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class RgbXYZ(nn.Module):
    def __init__(self) :
        super(RgbXYZ, self).__init__()
        self.Matrix = torch.tensor(([1,0,0],
                                    [0,1,0],
                                    [0,0,1]), dtype = torch.float)
        self.invMatrix = torch.tensor(([1,0,0],
                                        [0,1,0],
                                        [0,0,1]), dtype = torch.float)
        self.gamma = 2.2
        self.eps = 1e-6


    def rgb_to_XYZ(self, img):
        '''
        input :
        img in rgb form in range [0, 1]
        img shape: [B, C, H, W]
        return img of Yxy form
        '''
        device = img.device
        # companding to XYZ
        img_RGB = torch.pow(self.eps + img, self.gamma)
        # img_RGB = torch.clamp(img_RGB, 0, 1)
        return img_RGB

    def XYZ_to_rgb(self, img):
        #converting to XYZ space
        # img = torch.clamp(img, 0, 1)
        img_rgb = torch.pow(self.eps + img, 1 / self.gamma)
        
        return img_rgb

  


def color_inv_Img(img, color):
    '''
    notice!!!:
    Args:
    img: [B, C, W, H], Yxy
    color: one Yxy form color
    Return:
    img_inv: [B, C, W, H]
    '''
    color_rgb = torch.tensor(color)
    convertor = RgbXYZ()
    color_Yxy = convertor.rgb_to_XYZ(color_rgb.unsqueeze(0).unsqueeze(2).unsqueeze(3))
    color = color_Yxy.squeeze()
    Xm = color[0]; Ym = color[1]; Zm = color[2]
    X2 = 2 * Xm - img[:, 0]; Y2 = 2 * Ym - img[:, 1]; Z2 = 2 * Zm - img[:, 2]
    img_des = torch.cat(
        (X2.unsqueeze(1), Y2.unsqueeze(1), Z2.unsqueeze(1)), dim=1)
    img_des = torch.clamp(img_des, 0, 1)
    return img_des

def color_inv_2Img(img, color):
    color_rgb = torch.tensor(color)
    convertor = RgbXYZ()
    color_Yxy = convertor.rgb_to_XYZ(color_rgb.unsqueeze(0).unsqueeze(2).unsqueeze(3))
    color = color_Yxy.squeeze()
    Xm = color[0]; Ym = color[1]; Zm = color[2]
    X2 = (3 * Xm - img[:, 0]) / 2 ; Y2 = (3 * Ym - img[:, 1]) / 2; Z2 = (3 * Zm - img[:, 2]) / 2
    img_des = torch.cat(
        (X2.unsqueeze(1), Y2.unsqueeze(1), Z2.unsqueeze(1)), dim=1)
    img_des = torch.clamp(img_des, 0, 1)
    return img_des

def color_3rdimg(img_frame1, img_frame2, color):
    color_rgb = torch.tensor(color)
    convertor = RgbXYZ()
    color_Yxy = convertor.rgb_to_XYZ(color_rgb.unsqueeze(0).unsqueeze(2).unsqueeze(3))
    color = color_Yxy.squeeze()
    Xm = color[0]; Ym = color[1]; Zm = color[2]
    X2 = 3 * Xm - img_frame1[:, 0] - img_frame2[:, 0]
    Y2 = 3 * Ym - img_frame1[:, 1] - img_frame2[:, 1]
    Z2 = 3 * Zm - img_frame1[:, 2] - img_frame2[:, 2]
    img_des = torch.cat(
        (X2.unsqueeze(1), Y2.unsqueeze(1), Z2.unsqueeze(1)), dim=1)
    img_des = torch.clamp(img_des, 0, 1)
    return img_des

def color_4thimg(img_frame1, img_frame2, img_frame3, color):
    color_rgb = torch.tensor(color)
    convertor = RgbXYZ()
    color_Yxy = convertor.rgb_to_XYZ(color_rgb.unsqueeze(0).unsqueeze(2).unsqueeze(3))
    color = color_Yxy.squeeze()
    Xm = color[0]; Ym = color[1]; Zm = color[2]
    X2 = 4 * Xm - img_frame1[:, 0] - img_frame2[:, 0] - img_frame3[:, 0]
    Y2 = 4 * Ym - img_frame1[:, 1] - img_frame2[:, 1] - img_frame3[:, 1]
    Z2 = 4 * Zm - img_frame1[:, 2] - img_frame2[:, 2] - img_frame3[:, 2]
    img_des = torch.cat(
        (X2.unsqueeze(1), Y2.unsqueeze(1), Z2.unsqueeze(1)), dim=1)
    img_des = torch.clamp(img_des, 0, 1)
    return img_des
