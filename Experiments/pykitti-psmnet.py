import numpy as np
import argparse
import sys

from matplotlib import pyplot as plt

import cv2 as cv

import torch
import torch.nn.functional
import torchvision.transforms

import pykitti
import PSMNet.models

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
    model = PSMNet.models.stackhourglass(maxdisp=192)
    state_dict = torch.load(pretrained_weights_path)
    model = torch.nn.DataParallel(model, device_ids=[0])
    model.load_state_dict(state_dict['state_dict'])
    model.eval() # prep for inference
    return model
# /load_model()

def load_image_pair(kitti, frame_number):
    imgL_in, imgR_in = kitti.get_rgb(frame_number)

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

    imgL = torch.nn.functional.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
    imgR = torch.nn.functional.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)

    return imgL, imgR
# /load_image_pair()

def kitti_stereo_left_focal_length(kitti):
    color_stereo_left_cam_K = kitti.calib.K_cam2
    # ... K[0,0] is f * k where f is focal length (along neg z-axis) and k is pixels/m along x-axis

    # KITTI uses FL2-14S3C-C for its color stereo cameras (ref: http://www.cvlibs.net/datasets/kitti/setup.php)
    # Googling for the datasheets, I got
    # *  https://www.gophotonics.com/products/scientific-industrial-cameras/point-grey-research-inc/45-571-fl2-14s3c-c
    # *  https://www.surplusgizmos.com/assets/images/flea2-FW-Datasheet.pdf
    color_stereo_left_cam_x_pixel_width = 4.65e-6 # m
    color_stereo_left_cam_f = color_stereo_left_cam_K[0,0] * color_stereo_left_cam_x_pixel_width
    return color_stereo_left_cam_f
# /kitti_stereo_left_focal_length()

def do_psmnet(kitti, frame_range, model, use_cuda):
    if use_cuda:
        model.cuda()

    B = kitti.calib.b_rgb # baseline, m
    f = kitti_stereo_left_focal_length(kitti) # m
    K = kitti.calib.K_cam2.astype(np.float64) # intrinsics matrix for camera #2 (left stereo)

    for frame_num in frame_range:
        left_img, right_img = load_image_pair(kitti, frame_num)
        print('Processing frame {} shape = {}'.format(frame_num, left_img.shape))
        if use_cuda:
            left_img.cuda()
            right_img.cuda()
        with torch.no_grad():
            disparity_field = model(left_img, right_img)

        disparity_field = torch.squeeze(disparity_field)
        disparity_field_cpu = disparity_field.data.cpu().numpy()
        cv.imwrite('disparity-frame{}.png'.format(frame_num),
                   (disparity_field_cpu / 192.0 * 255).astype(np.uint8))
        print('\tmin = {}, max = {}'.format(np.min(disparity_field_cpu),
                                            np.max(disparity_field_cpu)))

        z_field = (B * K[0,0]) / disparity_field_cpu
        i_pix_coords, j_pix_coords = np.meshgrid(range(disparity_field_cpu.shape[0]),
                                                 range(disparity_field_cpu.shape[1]),
                                                 indexing = 'ij')
        pt_cloud_in = np.dstack((i_pix_coords, j_pix_coords, np.ones(disparity_field_cpu.shape)))

        # [X Y Z] = Z * K^{-1} [x y 1]^T
        pt_cloud_out = z_field[:,:,np.newaxis] * np.linalg.solve(K, pt_cloud_in[..., np.newaxis]).squeeze() 
        print('\tmin-z = {}, max-z = {}'.format(np.min(pt_cloud_out[:,:,2]),
                                                np.max(pt_cloud_out[:,:,2])))
        # plt.imshow(disp_cpu)
        # plt.waitforbuttonpress()
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
