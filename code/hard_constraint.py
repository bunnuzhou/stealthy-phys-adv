import cv2
import autograd.numpy as np
from scipy import optimize
from scipy import sparse
from autograd import jacobian, hessian

channel = 3
length = 0
gamma = 2.2
origin_X = 1.0
origin_Y = 1.0
origin_Z = 1.0
Matrix = np.asarray([[0.4124564,  0.3575761,  0.1804375],
                    [0.2126729,  0.7151522,  0.0721750],
                    [0.0193339,  0.1191920,  0.9503041]])
invMatrix = np.asarray([[3.2404542, -1.5371385, -0.4985314],
                        [-0.9692660,  1.8760108,  0.0415560],
                        [0.0556434, -0.2040259,  1.0572252]])

def objective_func(x, y):
    loss = np.sum((x -y)**2) * 0.5
    return loss

# def deriv_func(x, y):
#     gradient = x -y
#     return np.ravel(gradient)

def Y_constr_func(x):
    correction = x**gamma
    Y = np.dot(Matrix[1], correction)
    return Y

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

def chro_constr_func(x):
    correction = x**gamma
    XYZ = np.dot(Matrix, correction)
    color = np.array([origin_X, origin_Y, origin_Z]) * 2
    inv_color = np.dot(invMatrix, color - XYZ)
    return inv_color
    


def constr_optim(target, init):
    '''
    args:
        target: [C]
        init: [C]
    '''
    global channel
    channel = np.shape(target)[0]
    total = channel 

    x_bounds = list(zip([0.0]*total, [1.0]*total))

    # Y_lb, Y_ub = [max(origin_Y-0.2, 0.0)], [min(origin_Y+0.2, 1)]
    # constrain_Y = optimize.NonlinearConstraint(Y_constr_func, Y_lb, Y_ub,jac=jacobian(Y_constr_func))

    chro_lb, chro_ub = [0.0]*total, [1.0]*total
    constrain_chro = optimize.NonlinearConstraint(chro_constr_func, chro_lb, chro_ub, jac=jacobian(chro_constr_func))
    # constrains = [constrain_Y, constrain_chro]
    constrains = [constrain_chro]

    result = optimize.minimize(objective_func, init, args=(target), method='SLSQP', bounds=x_bounds, \
        constraints=constrains)#, options={'ftol': 4e-3})
    if result.success:
        ret = np.reshape(result.x, (channel))
    else:
        # print(result.message)
        ret = init
    return ret

def optim_pixels(target, init, origin_Y_yxy, origin_x, origin_y):
    '''
    args:
    target: [C, N]
    init: [C,N]
    origin_Y_yxy, origin_x, origin_y: float
    description: process pixel-wise
    '''
    assert target.shape == init.shape
    c, l = target.shape

    global origin_X
    global origin_Y
    global origin_Z
    origin_Y = origin_Y_yxy
    origin_X = (origin_x * origin_Y) / origin_y
    origin_Z = (1 - origin_x - origin_y) * origin_Y/ origin_y

    ans = []
    for i in range(l):
        ans.append(constr_optim(target[:, i], init[:, i]))

    return np.asarray(ans)
    
