from numpy import dtype
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

class RgbYxy(nn.Module):
    def __init__(self) :
        super(RgbYxy, self).__init__()
        self.Matrix = torch.tensor(([0.4124564,  0.3575761,  0.1804375],
                                    [0.2126729,  0.7151522,  0.0721750],
                                    [0.0193339,  0.1191920,  0.9503041]), dtype = torch.float)
        self.invMatrix = torch.tensor(([3.2404542, -1.5371385, -0.4985314],
                                        [-0.9692660,  1.8760108,  0.0415560],
                                        [0.0556434, -0.2040259,  1.0572252]), dtype = torch.float)
        self.gamma = 2.2
        self.eps = 1e-6


    def rgb_to_Yxy(self, img):
        '''
        input :
        img in rgb form in range [0, 1]
        img shape: [B, C, H, W]
        return img of Yxy form
        '''
        device = img.device
        # companding to XYZ
        img_RGB = torch.pow(self.eps + img, self.gamma)
        Matrix = self.Matrix.to(device)
        img = torch.transpose(torch.matmul(Matrix, torch.transpose(img_RGB, 1, 2)), 1, 2)

        x = img[:, 0] / (img[:, 0] + img[:, 1] + img[:, 2] + self.eps)
        y = img[:, 1] / (img[:, 0] + img[:, 1] + img[:, 2] + self.eps)
        Y = img[:, 1]
        img_Yxy = torch.cat((Y.unsqueeze(1), x.unsqueeze(1), y.unsqueeze(1)), dim = 1)
        img_Yxy = torch.clamp(img_Yxy, 0, 1)
        return img_Yxy

    def Yxy_to_rgb(self, img):
        #converting to XYZ space
        Y = img[:, 0]
        X = (img[:, 1] * Y) / (img[:, 2] + self.eps)
        Z = (1 - img[:, 1] - img[:, 2]) * Y / (img[:, 2] + self.eps)
        img_XYZ = torch.cat(
            (X.unsqueeze(1), Y.unsqueeze(1), Z.unsqueeze(1)), dim=1)

        invMatrix = self.invMatrix.to(img.device)
        img_RGB = torch.transpose(torch.matmul(
            invMatrix, torch.transpose(img_XYZ, 1, 2)), 1, 2)
        img_RGB = torch.clamp(img_RGB, 0, 1)
        img_rgb = torch.pow(self.eps + img_RGB, 1 / self.gamma)

        return img_rgb


    def Yxy_to_rgb_noclip(self, img):
        #converting to XYZ space
        Y = img[:, 0]
        X = (img[:, 1] * Y) / (img[:, 2] + self.eps)
        Z = (1 - img[:, 1] - img[:, 2]) * Y / (img[:, 2] + self.eps)
        img_XYZ = torch.cat(
            (X.unsqueeze(1), Y.unsqueeze(1), Z.unsqueeze(1)), dim=1)

        invMatrix = self.invMatrix.to(img.device)
        img_RGB = torch.transpose(torch.matmul(
            invMatrix, torch.transpose(img_XYZ, 1, 2)), 1, 2)

        return img_RGB
  


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
    convertor = RgbYxy()
    color_Yxy = convertor.rgb_to_Yxy(color_rgb.unsqueeze(0).unsqueeze(2).unsqueeze(3))
    color = color_Yxy.squeeze()
    Ym = color[0]; xm = color[1]; ym = color[2]
    Y1 = img[:, 0]; x1 = img[:, 1]; y1 = img[:, 2]
    # X1 = img[:, 1] / (img[:, 2]+1e-6) * img[:,0]
    # Y1 = img[:, 0]
    # Z1 = (1 - img[:, 1] - img[:, 2]) / (img[:, 2]+1e-6) * img[:, 0]
    # Xm = x/y * Y ; Ym = Y; Zm = (1 - x -y) / y * Y ;
    # X2 = 2*Xm - X1; Y2 = 2*Ym - Y1; Z2 = 2*Zm - Z1;
    # x2 = X2 /(X2 + Y2 + Z2 + 1e-6)
    # y2 = Y2 / (X2 + Y2 + Z2 + 1e-6)
    Y2 = 2*Ym - Y1
    y2 = (Y2 * y1 * ym) / (Y2* y1 - Y1 * (ym - y1) + 1e-6)
    x2 = Y1 / (Y2 + 1e-6) * (y2 / (y1 + 1e-6)) * (xm - x1) + xm
    # Y2 = torch.pow(convertor.rgb_to_Yxy(color_rgb.unsqueeze(0).unsqueeze(2).unsqueeze(3)).squeeze()[0], 2.2) - \
    #         torch.pow(img[:, 0], 2.2)
    # Y2 = torch.pow(Y2, 1/2.2)
    img_des = torch.cat((Y2.unsqueeze(1), x2.unsqueeze(1), y2.unsqueeze(1)), dim = 1)
    # img_des_rgb = convertor.Yxy_to_rgb(img_des)
    # img_des_rgb = torch.pow(img_des_rgb, 1/2.2)
    # img_des = convertor.rgb_to_Yxy(img_des_rgb)
    return img_des


