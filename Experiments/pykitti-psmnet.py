import numpy as np
import argparse
import sys

import torch
import torchvision
from matplotlib import pyplot as plt

import pykitti
import PSMNet

# global preprocessing transform (from PSMNet Test_img.py)
normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]}
infer_transform = \
    torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(**normal_mean_var)])    


def parse_cmdline():
    parser = argparse.ArgumentParser('Python KITTI OpenCV Stereo-SGBM Experiment')
    parser.add_argument('--base_dir',
                        required = True,
                        help='Base directory where KITTI raw data is located')
    parser.add_argument('--date',
                        required = True,
                        help='YYYY_MM_DD formatted date for the drive')
    parser.add_argument('--drive',
                        required = True,
                        help='NNNN formatted Drive number for the given date')
    parser.add_argument('--pretrained_weights_path',
                        required=True,
                        help='Path to pretrained PSMNet weights file')
    parser.add_argument('--use_cuda',
                        action='store_true',
                        default=torch.cuda.is_available(),
                        help='Use GPU (CUDA) [default=True]')
    args = parser.parse_args()

    print('Using args ...')
    print('\tbase_dir = ', args.base_dir)
    print('\tdate = ', args.date)
    print('\tdrive = ', args.drive)
    print('\tpretrained_weights_path = ', args.pretrained_weights_path)
    print('\tcuda = ', args.use_cuda)

    return args
# /parse_cmdline()

def load_model(pretrained_weights_path):
    model = PSMNet.models.stackhourglass.PSMNet(maxdisp=192)
    state_dict = torch.load(pretrained_weights_path)
    model.load_state_dict(state_dict['state_dict'])
    model = torch.nn.DataParallel(model, device_ids=[0])
    model.eval() # prep for inference
    return model
# /load_model()

def load_image_pair(kitti, frame_number):
    imgL_in, imgR_in = kitti.get_rgb(f)

    # following steps from PSMNet Test_img.py
    imgL = infer_transform(imgL_in)
    imgR = infer_transform(imgR_in)

    # pad to width and hight to 16 times
    if imgL.shape[1] % 16 != 0:
        times = imgL.shape[1]//16       
        top_pad = (times+1)*16 -imgL.shape[1]
    else:
        top_pad = 0

    if imgL.shape[2] % 16 != 0:
        times = imgL.shape[2]//16                       
        right_pad = (times+1)*16-imgL.shape[2]
    else:
        right_pad = 0    

    imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
    imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)

    return imgL, imgR
# /load_image_pair()

def do_psmnet(kitti, frame_range, model, use_cuda):
    if use_cuda:
        model.cuda()

    for frame_num in frame_range:
        print('Processing frame {}. Press any key.'.format(frame_num))
        left_img, right_img = load_image_pair(kitti, frame_num)
        if use_cuda:
            left_img.cuda()
            right_img.cuda()
        with torch.no_grad():
            disp = model(left_img, right_img)

        disp = torch.squeeze(disp)
        disp_cpu = disp.data.cpu().numpy()
        plt.imshow(disp_cpu)
        plt.waitforbuttonpress()
    # /for f

# /do_psmnet

def main():
    args = parse_cmdline()
    model = load_model(args.pretrained_weights_path)
    kitti = pykitti.raw(args.base_dir, args.date, args.drive)
    do_psmnet(kitti, range(50,70), model, args.use_cuda)
# /main()

if __name__ == "__main__":
    main()
