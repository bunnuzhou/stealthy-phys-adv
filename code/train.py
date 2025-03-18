import os
from argparse import ArgumentParser

import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from models.experimental import attempt_load
from util_proj.DataAugment_alpha import *
from util_proj.RgbXYZ import *
from util_proj.Simulation import *
from utils.general import (non_max_suppression)
from utils.torch_utils import select_device

import warnings
warnings.filterwarnings("ignore")

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

## packages for color field changing, image pasting and screen distortion simulation
RGBXYZconvertor = RgbXYZ()
simulation = Image_Synthsis()
segmentation = SegImageGenerate()
# Y = {"0.2": [0.4812, 0.4812, 0.4812], 
#     "0.3": [0.5785, 0.5785, 0.5785],
#     "0.4": [0.6593, 0.6593, 0.6593],
#     "0.5": [0.7297, 0.7297, 0.7297]}

names = ['pl5', 'pl20', 'pl30', 'pl40', 'pl50', 'pl60', 'pl70', 'pl80', 'pl100', 'pl120', 'pn', 'pne', 'i5', 'p11', 'p26', 'i4', 'il60', 'i2', 
        'w57', 'p5', 'p10', 'ip', 'il80', 'p23', 'pr40', 'ph4.5', 'w59', 'p12', 'p3', 'w55', 'pm20', 'pg', 'pm55', 'p27', 'il100', 'w13', 'ph4', 'p19', 
        'pm30', 'ph5', 'p6', 'w32']
name_cls = {}

class Background_Dataset(Dataset):
    def __init__(self, path_list, transform):
        self.path_list = path_list
        self.transform = transform

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        image = Image.open(self.path_list[idx])
        sample = self.transform(image)
        return sample


def loss_untargeted(class_num):
    def loss_fn(det, epoch):
        loss = torch.tensor(0.0)
        loss.requires_grad_(True)
        idx_origin = torch.nonzero(det[:, 5].int() == class_num, as_tuple=False)
        det_origin = torch.index_select(det, 0, idx_origin.squeeze())
        if not det_origin.numel() == 0:
            loss = loss + torch.max(det_origin[:, 4])
        return loss
    return loss_fn

m = torch.nn.Sigmoid()
def loss_targeted(target):
    def loss_fn(det, epoch):
        loss = torch.tensor(0.0)
        loss.requires_grad_(True)
        index_tar = torch.nonzero(det[:, 5].int() == target, as_tuple=False)
        det_tar = torch.index_select(det, 0, index_tar.squeeze())
        if not det_tar.numel() == 0:
            loss = loss - torch.max(det_tar[:, 4])
        index_other = torch.nonzero(det[:, 5].int() != target, as_tuple=False)
        if not index_other.numel() == 0:
            poss_max = torch.max(torch.index_select(det, 0, index_other.squeeze())[:,4])
            loss = loss + poss_max * m(100* (poss_max - 0.4)) 
        return loss
    return loss_fn

def loss_disappear():
    def loss_fn(det, epoch):
        loss = torch.tensor(0.0)
        loss.requires_grad_(True)
        loss = loss + torch.max(det[:,4])
        return loss
    return loss_fn

def loss_creation(target):
    def loss_fn(det, epoch):
        loss = torch.tensor(0.0)
        loss.requires_grad_(True)
        index_tar = torch.nonzero(det[:, 5].int() == target, as_tuple=False)
        det_tar = torch.index_select(det, 0, index_tar.squeeze())
        if not det_tar.numel() == 0:
            loss = loss - torch.max(det_tar[:, 4])
        return loss
    return loss_fn

## calculate the weight matrix to simulate the screen distortion
def random_sample4(device):
    '''
    '''
    random_component = torch.diag(torch.rand(4).to(device) * 0.4)
    random_matrix = torch.roll(random_component, 2, dims=1) + torch.roll(torch.diag(torch.ones(4).to(device) * 0.4) - random_component, 3, dims=1)
    sampling_matrix = torch.diag(torch.ones(4)).to(device) + torch.roll(torch.diag(torch.ones(4).to(device)), 1, dims=1) + random_matrix
    # weight 按行计算
    weight_matrix = torch.cat(((torch.nn.functional.pad(sampling_matrix.reshape(-1, 1), (0, 2, 0, 0), mode="constant", value=0).reshape(4, -1)),  
                                 (torch.nn.functional.pad(sampling_matrix.reshape(-1, 1), (1, 1, 0, 0), mode="constant", value=0).reshape(4, -1)),
                                 (torch.nn.functional.pad(sampling_matrix.reshape(-1, 1), (2, 0, 0, 0), mode="constant", value=0).reshape(4, -1))), 1).reshape(-1, 12) / 2.4
    return weight_matrix

def main():
    parser = ArgumentParser()
    parser.add_argument('--color', type=float, default= 0.66, \
        help='the average illumination of finally synthesized video')
    parser.add_argument('--victim', type=str, default= 'pl30', choices=['pl30', 'pl70', 'white'],\
        help='attacked speed limit sign, start with \' pl\', such as \'pl40\'')
    parser.add_argument('--attack', type=str, default='creation', choices=['untargeted', 'targeted', 'disappear', 'creation'])
    parser.add_argument('--target', type=str, default='pl70', \
        help='target class settled in targeted attack')
    parser.add_argument('--light', type=str, default="100", help="ambient light intensity")
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=60, help='training epoch num')
    parser.add_argument('--lr', type=list, default=[0.006, 0.004], help='training learning rate before 0.7 epoch and after')
    parser.add_argument('--img_path', type=str, default='../augment/nosign_3', help='background img path')
    parser.add_argument('--log_path', type=str, default='./log/run_20_none', help='log path for tensorboard')
    parser.add_argument('--device', type=str, default='cuda:1', choices=['cpu', 'cuda:0', 'cuda:1'])
    parser.add_argument('--sign_size_range', type=list, default=[100, 250], \
        help='the size\'s range of sign on a 2000*2000 high resolution image')

    args = parser.parse_args()
    
    #summarywritter
    writter = SummaryWriter(args.log_path)
    
    # compute hard bound
    for i, cls in enumerate(names):
        name_cls[cls] = i
    color_numel = float(args.color)
    color = [color_numel] * 3
    lower_bound = max(0, (np.power(color_numel, 2.2) * 4 - 1))
    upper_bound = min(3, np.power(color_numel, 2.2) * 4)
    os.system("mkdir ./" + str(args.light))
    os.system("mkdir ./" + str(args.light) + '/' + str(args.victim))
    if args.attack == 'creation' or args.attack == 'targeted':
        save_path = './'+ str(args.light) +'/' + str(args.victim) + '/yolo-'+ str(args.attack) + '-' + str(args.target) + '-' + str(args.color)
    else:
        save_path = './'+ str(args.light) +'/' + str(args.victim) + '/yolo-'+'-'+ str(args.attack) + '-' + str(args.color)
    device = torch.device(args.device)

    # origin image pair for image white and dark
    img_dark = Image.open('./template_speedlimit/' + args.victim + '-' + args.light +'-dark.png').convert('RGBA')
    img_bright = Image.open('./template_speedlimit/'+ args.victim + '-' + args.light + '-brig.png').convert('RGBA')
    img_dark = transforms.ToTensor()(img_dark.resize((256,256)))
    img_bright = transforms.ToTensor()(img_bright.resize((256,256)))
    
    mask = img_dark[3].to(device)
    img_dark = img_dark[:3].to(device)
    img_bright = img_bright[:3].to(device)
    mask = mask.repeat(1, 1, 1, 1)

    print('victim speedlimit sign: ' + args.victim)

    # img_backgrounds
    # used 100 pictures of the img in the background file folder
    path = []
    files = os.listdir(args.img_path)
    random.shuffle(files)
    files = files[:100]
    for filename in files:
        path.append(os.path.join(args.img_path, filename))
    backgrounds_dataset = Background_Dataset(path, transforms.ToTensor())
    background_dataloader = DataLoader(backgrounds_dataset, args.batch_size, shuffle=True, num_workers=16)

    
    # model: Yolo v5
    # device = select_device(device)
    model = attempt_load('./yolov5x.pt', inplace=False, fuse=False, map_location=device)
    model.eval()
    
    
    # initialize adversarial examples
    img_grey =  torch.cat((torch.ones(1, 256, 256)* float(args.color), torch.ones(1, 256, 256)* float(args.color), torch.ones(1, 256, 256) * float(args.color))).to(device)
    img_frame1 = torch.Tensor(3, 30, 30).uniform_(0, 0.8308).to(device) 
    img_frame1.requires_grad = True
    img_frame2 = torch.Tensor(3, 30, 30).uniform_(0, 0.8308).to(device)
    img_frame2.requires_grad = True
    img_frame3 = torch.Tensor(3, 30, 30).uniform_(0, 0.8308).to(device)
    img_frame3.requires_grad = True


    # loss function
    if args.attack == 'untargeted':
        class_num = name_cls[args.victim]
        loss_fn = loss_untargeted(class_num)
    elif args.attack == 'targeted':
        target_num = name_cls[args.target]
        loss_fn = loss_targeted(target_num)
    elif args.attack == 'disappear':
        loss_fn = loss_disappear()
    else:
        target_num = name_cls[args.target]
        loss_fn = loss_creation(target_num)

    # training setup
    epochs = args.epoch
    lr = args.lr[0]
    loss_track =[]
    relu = torch.nn.ReLU()
    
    for epoch in tqdm(range(epochs)):
        iter_num = 0.0
        total_loss = 0.0
        for i, img_background in enumerate(background_dataloader):
            iter_num += 1
            # random permutation
            # exchange the position of four frames
            with torch.no_grad():
                img_frame1_1 = torch.clamp(img_frame1, 0, 1).to(device)
                img_frame2_2 = torch.clamp(img_frame2, 0, 1).to(device)
                img_frame3_3 = torch.clamp(img_frame3, 0, 1).to(device)
                img_frame1_XYZ = RGBXYZconvertor.rgb_to_XYZ(img_frame1_1.unsqueeze(0))
                img_frame2_XYZ = RGBXYZconvertor.rgb_to_XYZ(img_frame2_2.unsqueeze(0))
                img_frame3_XYZ = RGBXYZconvertor.rgb_to_XYZ(img_frame3_3.unsqueeze(0))
                img_frame4 = color_4thimg(img_frame1_XYZ, img_frame2_XYZ, img_frame3_XYZ, color)[0] ** (1 / 2.2)
                li = [img_frame1, img_frame2, img_frame3, img_frame4]
                idx = random.randint(0, 3)
                img_frame1, img_frame2, img_frame3= li[idx], li[(idx+1) % 4], li[(idx + 2) %4]
                
            img_frame1.requires_grad_()
            img_frame2.requires_grad_()
            img_frame3.requires_grad_()
            img_frame1_1 = torch.clamp(img_frame1, 0, 1).to(device)
            img_frame2_2 = torch.clamp(img_frame2, 0, 1).to(device)
            img_frame3_3 = torch.clamp(img_frame3, 0, 1).to(device)
            
            
            img_frame1_XYZ = RGBXYZconvertor.rgb_to_XYZ(img_frame1_1.unsqueeze(0))
            img_frame2_XYZ = RGBXYZconvertor.rgb_to_XYZ(img_frame2_2.unsqueeze(0))
            img_frame3_XYZ = RGBXYZconvertor.rgb_to_XYZ(img_frame3_3.unsqueeze(0))
            
            # hard constraint bound
            loss_regularize = torch.tensor(0.0)
            loss_regularize.requires_grad_()
            if lower_bound != 0 :
                suppress_pos = (lower_bound - (img_frame1_XYZ + img_frame2_XYZ + img_frame3_XYZ))
                if torch.sum(suppress_pos > 0) > 0:
                    loss_regularize = loss_regularize + torch.sum(relu(suppress_pos)) / torch.sum(suppress_pos > 0) * 100
            if upper_bound != 2 :
                suppress_pos = ((img_frame1_XYZ + img_frame2_XYZ + img_frame3_XYZ) - upper_bound)
                if torch.sum(suppress_pos > 0) > 0:
                    loss_regularize = loss_regularize + torch.sum(relu(suppress_pos)) / torch.sum(suppress_pos > 0) * 100
            
            
            # calculate the 4th frame with permutated three frame
            img_frame4_XYZ = color_4thimg(img_frame1_XYZ, img_frame2_XYZ, img_frame3_XYZ, color)
            img_input = torch.cat((img_frame1_XYZ, img_frame2_XYZ, img_frame3_XYZ, img_frame4_XYZ), 0)
            
            
            with torch.autograd.set_detect_anomaly(False):
                ## simulate the screen distortion
                weight_matrix = random_sample4(device)
                img_input = torch.nn.functional.conv2d(img_input.view(1, 12, 30, 30), weight_matrix.reshape((-1, 12, 1, 1)), stride=1, groups=1)
                img_input = img_input.reshape(-1, 3, 30, 30) 

                ## simulate the rainbow effect 
                img_input = segmentation.rainbow_effect(img_input, random.uniform(0.8, 0.9))

                ## projection simulation
                img_segmentation = transforms.Resize((256, 256), interpolation = InterpolationMode.BILINEAR)(img_input)
                img_background_input = simulation.imageProjected(img_dark.detach(), img_bright.detach(), img_segmentation)


                ## paste onto background images
                img_background_input_rgba = torch.cat((img_background_input, mask.detach().repeat(img_background_input.shape[0], 1, 1, 1)), dim= 1)
                img_background_input_rgba = transforms.RandomPerspective(0.2, p=1.0, interpolation=InterpolationMode.BILINEAR)(img_background_input_rgba)
                img_net_input_pre, bbox = img_bg_blend(img_background_input_rgba, img_background.to(device).detach(), args)


                img_net_input = torch.clamp(img_net_input_pre, 0, 1)
                img_net_input = transforms.Resize((480, 480))(img_net_input)

                ## rescale the groundtruth bbox coordinate corresponding to the original size of background image
                bbox_gt = bbox * 0.24
                
                # TSR predict
                output = model(img_net_input)[0]
                loss = torch.tensor(0.0)
                loss.requires_grad = True
                pred = non_max_suppression(output, conf_thres=0, multi_label=True, max_det=1000)
                for det in pred:
                    if len(det):
                        ## topk of the maximum iou
                        bbox_pred = det[:, :4]
                        iou_filter = torchvision.ops.box_iou(bbox_gt, bbox_pred).squeeze()
                        if args.attack == 'creation':
                            _, index_valid = torch.topk(iou_filter, 3 * 42)
                        else:
                            index_valid = torch.nonzero(iou_filter > 0, as_tuple=False)
                        det_valid = torch.index_select(det, 0, index_valid.squeeze())
                        if not det_valid.numel() == 0:
                            loss = loss + loss_fn(det_valid, epoch) 

                if args.attack == 'creation' and epoch < 0.4 * epochs:
                    loss = loss
                else:
                    loss = loss + loss_regularize 
                
                loss.backward()
                # output loss in every epoch
                print(loss.cpu().data)
                
                # update
                with torch.no_grad():
                    grad1 = img_frame1.grad
                    grad2 = img_frame2.grad
                    grad3 = img_frame3.grad
                    if grad1 is not None and grad2 is not None:
                        if epoch > 0.4 * epochs:
                            ## new learning rate
                            lr = args.lr[1]
                        img_frame1 = img_frame1.detach() - lr * grad1.sign()
                        img_frame2 = img_frame2.detach() - lr * grad2.sign()
                        img_frame3 = img_frame3.detach() - lr * grad3.sign()
                    else:
                        print("error")
                        iter_num -= 1
                        continue
                        # break

                total_loss += loss.cpu().data

                # dump every 10 epoch
                if epoch % args.save_interval == 0 and i == 100 / args.batch_size - 1:
                    with torch.no_grad():
                        img_save_1 = img_frame1
                        img_save_1 = transforms.Resize((256,256), interpolation=InterpolationMode.BILINEAR)(img_save_1)
                        img_save_1 = transforms.ToPILImage()(img_save_1 * mask[0] + (1-mask[0])*img_grey)
                        
                        img_save_1.save(save_path +'_1.png')
                        img_save_2 = img_frame2
                        img_save_2 = transforms.Resize((256,256), interpolation=InterpolationMode.BILINEAR)(img_save_2)
                        img_save_2 = transforms.ToPILImage()(img_save_2 * mask[0] + (1 -mask[0])*img_grey)
                        img_save_2.save(save_path +'_2.png')
                        
                        img_save_3 = img_frame3
                        img_save_3 = transforms.Resize((256,256), interpolation=InterpolationMode.BILINEAR)(img_save_3)
                        img_save_3 = transforms.ToPILImage()(img_save_3 * mask[0] + (1 -mask[0])*img_grey)
                        img_save_3.save(save_path +'_3.png')

                loss_track.append(loss.cpu().data)


                del img_net_input_pre
                del img_net_input
                del loss
        writter.add_scalar('loss/train', total_loss / iter_num, epoch)


    img_save_1 = img_frame1
    img_save_1 = transforms.Resize((256,256), interpolation=InterpolationMode.BILINEAR)(img_save_1)
    img_save_1 = transforms.ToPILImage()(img_save_1 * mask[0] + (1 - mask[0]) * img_grey)
    img_save_1.save(save_path +'_1.png')
    img_save_2 = img_frame2
    img_save_2 = transforms.Resize((256,256), interpolation=InterpolationMode.BILINEAR)(img_save_2)
    img_save_2 = transforms.ToPILImage()(img_save_2 * mask[0] + (1- mask[0])*img_grey)
    img_save_2.save(save_path +'_2.png')
    img_save_3 = img_frame3
    img_save_3 = transforms.Resize((256,256), interpolation=InterpolationMode.BILINEAR)(img_save_3)
    img_save_3 = transforms.ToPILImage()(img_save_3 * mask[0] + (1 -mask[0])*img_grey)
    img_save_3.save(save_path +'_3.png')
    np.savetxt('./loss_track.csv', np.array(loss_track), delimiter=',')

    
if __name__ == '__main__':
    main()