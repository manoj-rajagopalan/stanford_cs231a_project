import numpy as np
import argparse
import sys
import time

import cv2 as cv
import torch

import pykitti

# modules in this project
import datasets
import psm
import yolov5
import opencv_sgbm


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
    parser.add_argument('--frames',
                        required=True,
                        help='Range of frames to process in <first>,<last+1> format')
    parser.add_argument('--output_dir',
                        default='.',
                        help='Directory to (over)write results')
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
    parser.add_argument('--use_opencv_for_stereo',
                        action='store_true',
                        default=False,
                        help='Use OpenCV Stereo SGBM for stereo disparity calculations, instead of PSMNet')
    parser.add_argument('--visualize_using_pptk',
                        action='store_true',
                        default=False,
                        help='Interactively visualize point clouds using PPTK viewer')
    parser.add_argument('--dump_objects_in_lidar_orientation',
                         action='store_true',
                         default=False,
                         help='Write object point clouds, per-frame, into .npy files in LIDAR axes')
    args = parser.parse_args()
    args.frames = tuple([int(x) for x in args.frames.split(',')])

    print('Using args ...')
    print('\tbase_dir = ', args.base_dir)
    print('\tdate = ', args.date)
    print('\tdrive = ', args.drive)
    print('\tframes = ', args.frames)
    print('\toutput_dir = ', args.output_dir)
    print('\tpsmnet_pretrained_weights = ', args.psmnet_pretrained_weights)
    print('\tyolov5_pretrained_weights = ', args.yolov5_pretrained_weights)
    print('\tuse_opencv_for_stereo = ', args.use_opencv_for_stereo)
    print('\tcuda = ', args.use_cuda)
    print('\tvisualize_using_pptk = ', args.visualize_using_pptk)
    print('\tdump_objects_in_lidar_orientation = ', args.dump_objects_in_lidar_orientation)

    return args
# /parse_cmdline()

def dumpObjPointCloudInLidarOrientation(frame_num, obj_pt_cloud_list_left_stereo, output_dir):
    # KITTI coord system:  x:right, y:down, z:front
    # OpenPCD coord system: x:front, y:left, z:up
    lidar_from_stereo = np.array([[ 0,  0, 1],  # x <-- z
                                  [-1,  0, 0],  # y <-- -x
                                  [ 0, -1, 0]]) # z --> -y
    for obj_num, obj_pt_cloud_stereo in enumerate(obj_pt_cloud_list_left_stereo):
        obj_pt_cloud_lidar = \
            np.einsum('ij,nj->ni', lidar_from_stereo, obj_pt_cloud_stereo)
        with(open(output_dir + '/' + f'frame_{frame_num:03}-obj_{obj_num:03}.npy', 'wb')) as f:
            np.save(f, obj_pt_cloud_lidar)
    # /for
# /dumpObjPointCloudInLidarOrientation()


def pointCloud(disparity_field, B, K):
    # H x W numpy array of float
    assert len(disparity_field.shape) == 2
    z_field = (B * K[0,0]) / disparity_field

    # Note: swapped order of i and j in going to x and y
    pix_y_coords, pix_x_coords = np.meshgrid(range(disparity_field.shape[0]),
                                             range(disparity_field.shape[1]),
                                             indexing = 'ij')
    pt_cloud_field = np.dstack((pix_x_coords, pix_y_coords, np.ones(disparity_field.shape)))

    # [X Y Z] = Z * K^{-1} [x y 1]^T
    K_inv = np.linalg.inv(K)
    pt_cloud_field = z_field[:,:,np.newaxis] * pt_cloud_field
    pt_cloud_field = np.einsum('ij,xyj->xyi', K_inv, pt_cloud_field)
    print('\tmin-z = {:.2f}, max-z = {:.2f}'.format(np.min(pt_cloud_field[:,:,2]),
                                                    np.max(pt_cloud_field[:,:,2])))
    return pt_cloud_field
# /pointCloud()


class Oriented3dBoundingBox:
    def __init__(self, origin, axes, xmin, xmax, ymin, ymax, zmin, zmax):
        assert origin.shape == (3,)
        self.origin = origin
        self.cam_from_obj = axes # col vectors form orthoNORMAL basis
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
    # /__init__()

    def draw(self, cv_img, K):
        '''Draw self onto numpy array with cv color-channel order
           using intrinsic matrix K'''

        # OpenCV BGR color constants
        red = (0,0,255)
        green = (0,255,0)
        blue = (255,0,0)

        def uv(P_obj):
            '''Compute pixel coords from object coords'''
            P_cam = self.origin + np.dot(self.cam_from_obj, P_obj)
            P_cam = P_cam / P_cam[2]
            KP_cam = np.dot(K, P_cam).squeeze()
            u, v = KP_cam[0:2]
            return int(u), int(v)
        # /uv()

        one_pixel_thick = 1

        # Compute pixel locations for bounding rectangle PQRS on z-min plane (lo)
        p_lo = uv(np.array([self.xmin, self.ymin, self.zmin]))
        q_lo = uv(np.array([self.xmax, self.ymin, self.zmin]))
        r_lo = uv(np.array([self.xmax, self.ymax, self.zmin]))
        s_lo = uv(np.array([self.xmin, self.ymax, self.zmin]))
        cv.line(cv_img, p_lo, q_lo, red,   one_pixel_thick)
        cv.line(cv_img, r_lo, s_lo, red,   one_pixel_thick)
        cv.line(cv_img, q_lo, r_lo, green, one_pixel_thick)
        cv.line(cv_img, p_lo, s_lo, green, one_pixel_thick)

        # Compute pixel locations for bounding rectangle PQRS on z-max plane (hi)
        p_hi = uv(np.array([self.xmin, self.ymin, self.zmax]))
        q_hi = uv(np.array([self.xmax, self.ymin, self.zmax]))
        r_hi = uv(np.array([self.xmax, self.ymax, self.zmax]))
        s_hi = uv(np.array([self.xmin, self.ymax, self.zmax]))
        cv.line(cv_img, p_hi, q_hi, red,   one_pixel_thick)
        cv.line(cv_img, r_hi, s_hi, red,   one_pixel_thick)
        cv.line(cv_img, q_hi, r_hi, green, one_pixel_thick)
        cv.line(cv_img, p_hi, s_hi, green, one_pixel_thick)

        # "Vertically" connect corresponding pixels for vertices on lo and hi planes
        cv.line(cv_img, p_lo, p_hi, blue, one_pixel_thick)
        cv.line(cv_img, q_lo, q_hi, blue, one_pixel_thick)
        cv.line(cv_img, r_lo, r_hi, blue, one_pixel_thick)
        cv.line(cv_img, s_lo, s_hi, blue, one_pixel_thick)
    # /draw()

# /class Oriented3dBoundingBox



def detect3dObjects(pt_cloud_field, bboxes2D):
    H, W = pt_cloud_field.shape[0:2]
    obj_3dbbox_list = []
    obj_pt_cloud_list = []
    kmeans_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    kmeans_attempts = 10
    kmeans_flags = cv.KMEANS_RANDOM_CENTERS

    for bbox2D in bboxes2D:
        # Slightly pad the 2D bounding box before extracting points
        x1, y1, x2, y2 = bbox2D.astype(int)

        # extract
        obj_pt_cloud = pt_cloud_field[y1:y2, x1:x2, :] # Note swapped xy --> ij indexing
        obj_pt_cloud = obj_pt_cloud.reshape(-1, 3) # make single list of 3D points

        print('\t-- Clustering --')
        print('\t   Initial pt cloud size = ', len(obj_pt_cloud))

        # filter out ground plane (KITTI cameras at 1.65 m from ground)
        obj_pt_cloud = obj_pt_cloud[(obj_pt_cloud[:,1] < 1.6), :]
        print('\t   After removing ground plane = ', len(obj_pt_cloud))

        # 2-means clustering to filter out background based on distances of points from camera
        obj_pt_cloud_distances = \
            np.linalg.norm(obj_pt_cloud, axis=1).astype(np.float32)
            # ... OpenCV KMeans algo asserts on single-precision float

        compactness, labels, centers = \
            cv.kmeans(obj_pt_cloud_distances, 2, None, kmeans_criteria, kmeans_attempts, kmeans_flags)
        labels = labels.squeeze() # labels.shape = (N,1) for some reason
        assert len(centers) == 2
        obj_label = 0 if (centers[0] < centers[1]) else 1 # index of foreground obj
        obj_pt_cloud = obj_pt_cloud[labels == obj_label]
        print('\t   After clustering = ', len(obj_pt_cloud))

        # debug using axis-aligned bounding box
        use_axis_aligned_bb_for_debug = False
        if use_axis_aligned_bb_for_debug:
            obj_pt_cloud_center = np.mean(obj_pt_cloud, axis=0, keepdims=True)
            obj_pt_cloud_relative = obj_pt_cloud - obj_pt_cloud_center
            eigvecs = np.eye(3)
            pca_xmin = np.min(obj_pt_cloud_relative[:,0])
            pca_xmax = np.max(obj_pt_cloud_relative[:,0])
            pca_ymin = np.min(obj_pt_cloud_relative[:,1])
            pca_ymax = np.max(obj_pt_cloud_relative[:,1])
            pca_zmin = np.min(obj_pt_cloud_relative[:,2])
            pca_zmax = np.max(obj_pt_cloud_relative[:,2])

        else: # PCA, the real deal
            obj_pt_cloud_center, eigvecs = cv.PCACompute(obj_pt_cloud, mean=None) # Note: 'center' is 2D matrix of dimension 1x3
            obj_pt_cloud_in_pca_space = cv.PCAProject(obj_pt_cloud, obj_pt_cloud_center, eigvecs)
            pca_xmin = np.min(obj_pt_cloud_in_pca_space[:,0])
            pca_xmax = np.max(obj_pt_cloud_in_pca_space[:,0])
            pca_ymin = np.min(obj_pt_cloud_in_pca_space[:,1])
            pca_ymax = np.max(obj_pt_cloud_in_pca_space[:,1])
            pca_zmin = np.min(obj_pt_cloud_in_pca_space[:,2])
            pca_zmax = np.max(obj_pt_cloud_in_pca_space[:,2])
        # if/else

        obj_3dbbox = Oriented3dBoundingBox(obj_pt_cloud_center.squeeze(),
                                           eigvecs.T, # want basis vecs in columns
                                           pca_xmin, pca_xmax,
                                           pca_ymin, pca_ymax,
                                           pca_zmin, pca_zmax)
        obj_3dbbox_list.append(obj_3dbbox)
        obj_pt_cloud_list.append(obj_pt_cloud)
    # /for bbox2D

    return obj_3dbbox_list, obj_pt_cloud_list
# /detect3dObjects()

def renderImages(frame_num, left_img, disparity_field, bboxes2D, bboxes3D, K, output_dir):
    cv.imwrite(output_dir+'/'+f'{frame_num:03}-0_left.png', left_img)
    cv.imwrite(output_dir+'/'+f'{frame_num:03}-1_disp.png', disparity_field)

    # YOLO 2D bounding boxes
    left_yolo_bb2d_img = left_img.copy()
    disp_yolo_bb2d_img = np.dstack([disparity_field.copy()]*3)
    num_bboxes = len(bboxes2D)
    assert num_bboxes == len(bboxes3D)
    for n in range(num_bboxes):
        bb2D = bboxes2D[n]
        x1, y1, x2, y2 = bb2D
        cv.rectangle(left_yolo_bb2d_img,
                    (x1, y1), (x2, y2),
                    color=[0,0,255], # BGR order for OpenCV
                    thickness=1, lineType=cv.LINE_AA)
        cv.rectangle(disp_yolo_bb2d_img,
                    (x1, y1), (x2, y2),
                    color=[0,255,255], # BGR order for OpenCV
                    thickness=1, lineType=cv.LINE_AA)
    cv.imwrite(output_dir+'/'+f'{frame_num:03}-2_left_yolo.png', left_yolo_bb2d_img)
    cv.imwrite(output_dir+'/'+f'{frame_num:03}-3_disp_yolo.png', disp_yolo_bb2d_img)

    # PCA 3D bounding boxes
    left_pca_bb3d_img = left_img.copy()
    disp_pca_bb3d_img = np.dstack([disparity_field.copy()]*3)
    for n in range(num_bboxes):
        bb3D = bboxes3D[n]
        bb3D.draw(left_pca_bb3d_img, K)
        bb3D.draw(disp_pca_bb3d_img,  K)
    cv.imwrite(output_dir+'/'+f'{frame_num:03}-4_left_pca.png', left_pca_bb3d_img)
    cv.imwrite(output_dir+'/'+f'{frame_num:03}-5_disp_pca.png', disp_pca_bb3d_img)
# /renderImages()

def processFrames(kitti_raw_dataset, psmnet, opencv_stereo, yolonet, args):

    #- B = kitti.calib.b_rgb # baseline, m
    #- K = kitti.calib.K_cam2.astype(np.float64) # intrinsics matrix for camera #2 (left stereo)

    for data_frame in kitti_raw_dataset:
        frame_num, left_img, right_img, B, K = data_frame
        print(f'\nProcessing frame {frame_num}. Image size = {left_img.shape}')
        # B is baseline, in metres
        # K is intrinsic calibration matrix

        begin_time_ns = time.time_ns()

        # H x W numpy array of float
        if args.use_opencv_for_stereo:
            disparity_field = opencv_stereo.disparityField(left_img, right_img)
        else:
            disparity_field = psmnet.disparityField(left_img, right_img, args.use_cuda)
        end_disparity_time_ns = time.time_ns()
        print('\t@disparity: {:.2f} s'.format(1.0e-9 * (end_disparity_time_ns - begin_time_ns)))

        # H x W array of 3D points in camera coords (x:right, y:down, z:forward)
        pt_cloud_field = pointCloud(disparity_field, B, K)
        end_point_cloud_time_ns = time.time_ns()
        print('\t@point_cloud: {:.2f} s'.format(1.0e-9 * (end_point_cloud_time_ns - end_disparity_time_ns)))

        # YOLOv5
        # - bboxes2D: (N,4) array of N 2D bounding boxes, each in (x1,y1,x2,y2) format
        # - confidences: 1D array of confidences for each BB (N values)
        bboxes2D, confidences = yolonet.detect(left_img, args.use_cuda)
        end_yolonet_time_ns = time.time_ns()
        print('\t@yolonet: {:.2f} s'.format(1.0e-9 * (end_yolonet_time_ns - end_point_cloud_time_ns)))

        # Diagnose objects
        for n, bb in enumerate(bboxes2D):
            disparity_field_obj = disparity_field[bb[1]:bb[3], bb[0]:bb[2]]
            min_obj_disp = np.min(disparity_field_obj)
            max_obj_disp = np.max(disparity_field_obj)
            print('\t#{} obj min disp = {:.2f} ({:.2f} m)'.format(n, min_obj_disp, B*K[0,0]/ min_obj_disp),
                  '\t#{} obj max disp = {:.2f} ({:.2f} m)'.format(n, max_obj_disp, B*K[0,0]/ max_obj_disp))
        # /for n,bb

        # Save intermediate results, for debug
        npz_filename = f'frame-{frame_num:03}.npz'
        with open(args.output_dir + '/' + npz_filename, 'wb') as f:
            np.savez(f, pt_cloud_field=pt_cloud_field,
                        bboxes2D=bboxes2D,
                        confidences=confidences)

        # list of Oriented3dBoundingBox
        begin_detect_time_ns = time.time_ns()
        bboxes3D, obj_pt_clouds = detect3dObjects(pt_cloud_field, bboxes2D)
        end_detect_time_ns = time.time_ns()
        print('\t@detect: {:.3f} s'.format(1.0e-9 * (end_detect_time_ns - begin_detect_time_ns)))

        print('\t@frame {:.3f} s'.format(1.0e-9 * (end_detect_time_ns - begin_time_ns)))

        renderImages(frame_num, left_img, disparity_field, bboxes2D, bboxes3D, K, args.output_dir)
        if(args.dump_objects_in_lidar_orientation):
            dumpObjPointCloudInLidarOrientation(frame_num, obj_pt_clouds, args.output_dir)
    # /for frame_num

# /processFrames()


def main():
    args = parseCmdline()
    kitti_raw_dataset = \
        datasets.KittiRawDataset(args.base_dir, args.date, args.drive, args.frames)
    psmnet = psm.PyramidalStereoMatchingNet(args.psmnet_pretrained_weights, args.use_cuda, max_disp=240)
    opencv_stereo = opencv_sgbm.OpencvStereoSGBM(4, 192)
    yolonet = yolov5.YOLOnet(args.yolov5_pretrained_weights, args.use_cuda)
    processFrames(kitti_raw_dataset, psmnet, opencv_stereo, yolonet, args)
# /main()

if __name__ == "__main__":
    main()
