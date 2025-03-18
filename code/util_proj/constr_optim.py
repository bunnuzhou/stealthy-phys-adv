import cv2
import autograd.numpy as np
from scipy import optimize
from scipy import sparse
from autograd import jacobian, hessian
import torchvision.transforms as transforms
from RgbXYZ import *
import os
import threading
import multiprocessing

channel = 3
length = 0
gamma = 2.2
RGBYxyconvertor = RgbXYZ()

def objective_func(x, y):
    loss = np.sum((x -y)**2) * 0.5
    return loss

# def deriv_func(x, y):
#     gradient = x -y
#     return np.ravel(gradient)

# def Y_constr_func(x):
#     correction = x**gamma
#     Y = np.dot(Matrix[1], correction)
#     return Y

# def Y_constr_deriv(x):
#     x_shape = np.reshape(x, (channel, length))
#     gradient = np.dot(np.diag(Matrix[2]), x_shape**(gamma -1)) * gamma
#     gradient = np.ravel(gradient)
#     row = []
#     col = []
#     for i in range(length):
#         row = row + [i] * channel
#         col = col + [i*channel, i*channel+1, i*channel+2]
#     sparse_gradient = sparse.coo_matrix((gradient, (row, col)), shape = (length, channel*length))
#     return sparse_gradient

# def chro_constr_func(x):
#     correction = x**gamma
#     XYZ = np.dot(Matrix, correction)
#     color = np.array([origin_X, origin_Y, origin_Z]) * 2
#     inv_color = np.dot(invMatrix, color - XYZ)
#     return inv_color

def constraint(x):
    '''
    x: [N, C, B]
    4th frame exist constraint
    '''
    x = np.reshape(x, (3, 3))
    return np.ravel(np.sum(x**gamma, axis=1))
    
    
def constr_optim(target, init, upper_bound, lower_bound):    
    xbound = list(zip([0.0] * len(init), [1.0] * len(init)))
    constrain_chro = optimize.NonlinearConstraint(constraint, lower_bound, upper_bound, jac=jacobian(constraint))
    # constrains = [constrain_Y, constrain_chro]
    constrains = [constrain_chro]
    result = optimize.minimize(objective_func, init, args=(target), method='SLSQP', bounds=xbound, \
        constraints=constrains)#, options={'ftol': 4e-3})
    if result.success:
        ret = np.reshape(result.x, (3, 3))
    else:
        print(result.message)
        ret = init
    return ret
    
    


# def constr_optim(target, init):
#     '''
#     args:
#         target: [C]
#         init: [C]
#     '''
#     global channel
#     channel = np.shape(target)[0]
#     total = channel 

#     x_bounds = list(zip([0.0]*total, [1.0]*total))

#     # Y_lb, Y_ub = [max(origin_Y-0.2, 0.0)], [min(origin_Y+0.2, 1)]
#     # constrain_Y = optimize.NonlinearConstraint(Y_constr_func, Y_lb, Y_ub,jac=jacobian(Y_constr_func))

#     chro_lb, chro_ub = [0.0]*total, [1.0]*total
#     constrain_chro = optimize.NonlinearConstraint(chro_constr_func, chro_lb, chro_ub, jac=jacobian(chro_constr_func))
#     # constrains = [constrain_Y, constrain_chro]
#     constrains = [constrain_chro]

#     result = optimize.minimize(objective_func, init, args=(target), method='SLSQP', bounds=x_bounds, \
#         constraints=constrains)#, options={'ftol': 4e-3})
#     if result.success:
#         ret = np.reshape(result.x, (channel))
#     else:
#         # print(result.message)
#         ret = init
#     return ret

def optim_pixels(target, init, upper_bound, lower_bound):
    '''
    args:
    target: [C, N]
    init: [C,N]
    description: process pixel-wise, note that the pixel in different position is irrelavent 
    '''
    print(target.shape)
    assert target.shape == init.shape
    target = np.reshape(target, (-1, 3, 3))
    init = np.reshape(init, (-1, 3, 3))

    ans = []
    for i in range(len(target)):
        ans.append(constr_optim(np.ravel(target[i, ...]), np.ravel(init[i, ...]), upper_bound, lower_bound))

    return np.stack(ans, axis=0)

def postprocess(save_path, color, lower_bound, upper_bound):
    imgs = []
    for i in range(1, 4):
        img = cv2.imread(save_path + '_' + str(i) + '.png', cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (256, 256))
        img = img.astype(np.float16) / 255
        imgs.append(img)
    # process the origin img to hard constraint
    result = sum(map(lambda x: x**gamma, imgs))
    legitimite = np.logical_and(np.max(result, axis=2) <= upper_bound, np.min(result, axis=2) >= lower_bound)
    update_mask = np.logical_not(legitimite)
    target = np.stack((imgs[0][update_mask, ...], imgs[1][update_mask, ...], imgs[2][update_mask, ...]), axis=2)
    target = np.ravel(target)
    ret = optim_pixels(target, np.ones_like(target)*color, upper_bound, lower_bound)
    for i in range(3):
        imgs[i][update_mask, ...] = ret[..., i]
        img = (np.round(imgs[i] * 255)).astype(np.uint8)
        img = transforms.ToTensor()(img).to(torch.device('cuda:0'))
        # print(img)
        imgs[i] = RGBYxyconvertor.rgb_to_XYZ(img.unsqueeze(0))
    # video generation
    imgs.append(color_4thimg(imgs[0], imgs[1], imgs[2], [color] * 3))
    dirt, basename = os.path.split(save_path)
    for i in range(4):
        img = RGBYxyconvertor.XYZ_to_rgb(imgs[i])[0]
        img = np.transpose(img.cpu().numpy(), (1, 2, 0))
        # print(img)
        imgs[i] = np.uint8(np.round(img * 255))# cv2.cvtColor(np.uint8(img * 255), cv2.COLOR_RGB2BGR)
        # imgs[i] = cv2.resize(imgs[i], (256, 256), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite( dirt+ "/after-" + str(i + 1) + basename +'.png', imgs[i])
    
    # img_size = imgs[0].shape[:2]
    # codec = cv2.VideoWriter_fourcc(*'I420')
    # videowritter = cv2.VideoWriter()
    # videowritter.open(save_path+'.avi', codec, 240, img_size, True)
    # for i in range(3600):
    #     for img in imgs:
    #         videowritter.write(img)
    # videowritter.release()
    
        
        
        