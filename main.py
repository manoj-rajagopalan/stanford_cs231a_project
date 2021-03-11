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

import YOLOv5.models.experimental
import YOLOv5.utils.datasets
import YOLOv5.utils.general
import YOLOv5.utils

# global preprocessing transform (from PSMNet Test_img.py)
normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]}
inference_transform = \
    torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(**normal_mean_var)])    


def parseCmdline():
    parser = argparse.ArgumentParser('stereo_tracker')
    parser.add_argument('--base_dir',
                        required = True,
                        help='Base directory where KITTI raw data is located')
    parser.add_argument('--date',
                        required = True,
                        help='YYYY_MM_DD formatted date for the drive')
    parser.add_argument('--drive',
                        required = True,
                        help='NNNN formatted Drive number for the given date')
    parser.add_argument('--psmnet_pretrained_weights',
                        required=True,
                        help='Path to pretrained PSMNet weights file')
    parser.add_argument('--yolov5_pretrained_weights',
                        required=True,
                        help='Path to pretrained YOLOv5 weights file')
    parser.add_argument('--use_cuda',
                        action='store_true',
                        default=torch.cuda.is_available(),
                        help='Use GPU (CUDA) [default=True]')
    args = parser.parse_args()

    print('Using args ...')
    print('\tbase_dir = ', args.base_dir)
    print('\tdate = ', args.date)
    print('\tdrive = ', args.drive)
    print('\tpsmnet_pretrained_weights = ', args.psmnet_pretrained_weights)
    print('\tyolov5_pretrained_weights = ', args.yolov5_pretrained_weights)
    print('\tcuda = ', args.use_cuda)

    return args
# /parse_cmdline()

def loadPsmnet(psmnet_pretrained_weights, use_cuda):
    psmnet = PSMNet.models.stackhourglass(maxdisp=192)
    if use_cuda:
        state_dict = torch.load(psmnet_pretrained_weights)
    else:
        state_dict = torch.load(psmnet_pretrained_weights, map_location=torch.device('cpu'))

    psmnet = torch.nn.DataParallel(psmnet, device_ids=[0]) # required to be before load_state_dict()
    psmnet.load_state_dict(state_dict['state_dict'])
    psmnet.eval() # prep for inference
    if use_cuda:
        psmnet.cuda()
    else:
        psmnet.cpu()

    return psmnet
# /load_psmnet()

def loadYolov5(yolov5_pretrained_weights, use_cuda):
    device = 'cuda:0' if use_cuda else 'cpu'
    yolov5 = YOLOv5.models.experimental.attempt_load(yolov5_pretrained_weights,
                                                     map_location=device)
    yolov5.eval()
    if use_cuda:
        yolov5.cuda()
    else:
        yolov5.cpu()
    return yolov5
# /loadYolov5()

def preprocessForPsmnet(imgL_in, imgR_in):
    
    # following steps from PSMNet Test_img.py
    imgL = inference_transform(imgL_in)
    imgR = inference_transform(imgR_in)

    # pad to width and height to 16 times
    top_pad = (16 - (imgL.shape[1] % 16)) % 16
    right_pad = (16 - (imgL.shape[2] % 16)) % 16

    imgL = torch.nn.functional.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
    imgR = torch.nn.functional.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)

    return imgL, imgR
# /preprocessForPsmnet()

def preprocessForYolov5(img_in, use_cuda, stride=32, yolov5_img_size=640):
    # From YOLOv5.utils.datasets.LoadImages.__next__()
    # Padded resize
    img = \
        YOLOv5.utils.datasets.letterbox(img_in, yolov5_img_size, stride=stride)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    # From YOLOv5.detect.detect()
    device = 'cuda:0' if use_cuda else 'cpu'
    img = torch.from_numpy(img).to(device)
    img = img.half() if use_cuda else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    return img
# /preprocessForYolov5()

def psmnetDisparityField(psmnet, left_img, right_img, use_cuda):
    psmnet_left_img, psmnet_right_img = \
        preprocessForPsmnet(left_img, right_img)
    if use_cuda:
        psmnet_left_img.cuda()
        psmnet_right_img.cuda()
    with torch.no_grad():
        disparity_field = psmnet(psmnet_left_img, psmnet_right_img)

    disparity_field = torch.squeeze(disparity_field)
    disparity_field_cpu = disparity_field.data.cpu().numpy()
    print('\tmin = {}, max = {}'.format(np.min(disparity_field_cpu),
                                        np.max(disparity_field_cpu)))
    return disparity_field_cpu
# /psmnetDisparityField()

def pointCloud(disparity_field, B, K):
    # H x W numpy array of float
    z_field = (B * K[0,0]) / (-disparity_field) # remember that disparity is signed, actually

    i_pix_coords, j_pix_coords = np.meshgrid(range(disparity_field.shape[0]),
                                                range(disparity_field.shape[1]),
                                                indexing = 'ij')
    # Note: swapped order of i and j in going to x and y
    pt_cloud_in = np.dstack((j_pix_coords, i_pix_coords, np.ones(disparity_field.shape)))

    # [X Y Z] = Z * K^{-1} [x y 1]^T
    pt_cloud_out = z_field[:,:,np.newaxis] * np.linalg.solve(K, pt_cloud_in[..., np.newaxis]).squeeze() 
    print('\tmin-z = {}, max-z = {}'.format(np.min(pt_cloud_out[:,:,2]),
                                            np.max(pt_cloud_out[:,:,2])))
    pt_cloud_out = pt_cloud_out.reshape(-1,3)
    return pt_cloud_out
# /pointCloud()

def yolov5Detections(yolov5, img, use_cuda):
    img = preprocessForYolov5(img, use_cuda, stride=int(yolov5.stride.max()))
    yolov5_output = yolov5(img)[0]
    # Non-max suppression is required regardless of specific classes of interest.
    # But in this project we are interested only in cars, so filter on that.
    # See YOLOv5/data/coco.yaml for list of classes.
    predictions = \
        YOLOv5.utils.general.non_max_suppression(yolov5_output, classes=[2])

    # predictions is a list of (n,6) array where
    # - n is the number of detections and
    # - columns 0..3 store (x1,y1) and (x2,y2) for bounding boxes
    # - column 4 stores confidence values
    # - column 5 stores the integer class value

    assert len(predictions) == 1 # we process only 1 image at a time
    predictions = predictions[0] # extract the singleton (n,6) array
    predictions = predictions[predictions[:,4] > 0.8] # be reasonably sure!
    bounding_boxes, confidences = predictions[:, 0:4], predictions[:,4]
    return bounding_boxes, confidences, np.array(img.shape[2:4])
# /yolov5Detections()

class BoundingBox:
    def __init__(self, array_of_3d_pts):
        self.xmin = np.min(array_of_3d_pts[:,0])
        self.xmax = np.max(array_of_3d_pts[:,0])
        self.ymin = np.min(array_of_3d_pts[:,1])
        self.ymax = np.max(array_of_3d_pts[:,1])
        self.zmin = np.min(array_of_3d_pts[:,2])
        self.zmax = np.max(array_of_3d_pts[:,2])
    # /def __init__()

    def __str__(self):
        return f"({self.xmin:.2f}, {self.ymin:.2f}, {self.zmin:.2f}) -> ({self.xmax:.2f}, {self.ymax:.2f}, {self.zmax:.2f})"

# /class BoundingBox

def processFrames(kitti, frame_range, psmnet, yolov5, use_cuda):

    B = kitti.calib.b_rgb # baseline, m
    K = kitti.calib.K_cam2.astype(np.float64) # intrinsics matrix for camera #2 (left stereo)

    for frame_num in frame_range:
        print('\nProcessing frame {}'.format(frame_num))

        #  pair of H x W x C=3 numpy arrays
        left_img, right_img = kitti.get_rgb(frame_num) # PIL images
        left_img = np.array(left_img)
        right_img = np.array(right_img)

        # H x W numpy array of float
        disparity_field = \
            psmnetDisparityField(psmnet, left_img, right_img, use_cuda)

        # sequence of 3D points (x: right, y:top, z: back)
        pt_cloud = pointCloud(disparity_field, B, K)

        axis_aligned_bb = BoundingBox(pt_cloud)
        print('\taxis_aligned_bb = ', axis_aligned_bb)

        # Use YOLOv5 detections to place a red 2D bounding box on disparities
        bboxes2D, confidences, yolo_img_shape = \
            yolov5Detections(yolov5, left_img, use_cuda)
        disparity_field = (disparity_field / 192.0 * 255).astype(np.uint8)
        disparity_field = np.dstack([disparity_field] * 3) # colorize
        bboxes2D = YOLOv5.utils.general.scale_coords(yolo_img_shape, bboxes2D, left_img.shape)
        for bb2D in bboxes2D:
            x1, y1, x2, y2 = bb2D.int()
            cv.rectangle(disparity_field,
                        (x1, y1), (x2, y2),
                        color=[0,0,255], # BGR order for OpenCV
                        thickness=2, lineType=cv.LINE_AA)
        cv.imwrite('disparity-frame{}.png'.format(frame_num),
                    disparity_field)

        #- plt.imshow(disp_cpu)
        #- plt.waitforbuttonpress()

    # /for frame_num

# /do_psmnet


def main():
    args = parseCmdline()
    kitti = pykitti.raw(args.base_dir, args.date, args.drive)
    psmnet = loadPsmnet(args.psmnet_pretrained_weights, args.use_cuda)
    yolov5 = loadYolov5(args.yolov5_pretrained_weights, args.use_cuda)
    processFrames(kitti, range(50,70), psmnet, yolov5, args.use_cuda)
# /main()

if __name__ == "__main__":
    main()
