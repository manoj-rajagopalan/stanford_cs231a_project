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
    print('\tmin disp = {}, max disp = {}'.format(np.min(disparity_field_cpu),
                                                  np.max(disparity_field_cpu)))
    return disparity_field_cpu
# /psmnetDisparityField()

def pointCloud(disparity_field, B, K):
    # H x W numpy array of float
    z_field = (B * K[0,0]) / disparity_field

    # Note: swapped order of i and j in going to x and y
    pix_y_coords, pix_x_coords = np.meshgrid(range(disparity_field.shape[0]),
                                             range(disparity_field.shape[1]),
                                             indexing = 'ij')
    # move origin to center of image
    pix_x_coords = pix_x_coords - K[0,2]
    pix_y_coords = pix_y_coords - K[1,2]
    pt_cloud_init = np.dstack((pix_x_coords, pix_y_coords, np.ones(disparity_field.shape)))
    K = K.copy() # modifying
    K[0:2,2] = 0
    # [X Y Z] = Z * K^{-1} [x y 1]^T
    K_inv = np.linalg.inv(K)
    pt_cloud_field = z_field[:,:,np.newaxis] * np.einsum('ij,xyj->xyi', K_inv, pt_cloud_init)
    print('\tmin-z = {}, max-z = {}'.format(np.min(pt_cloud_field[:,:,2]),
                                            np.max(pt_cloud_field[:,:,2])))
    return pt_cloud_field
# /pointCloud()

def yolov5Detections(yolov5, img, use_cuda):
    orig_img_shape = img.shape
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
    bboxes2D_yolo_scale, confidences = predictions[:, 0:4], predictions[:,4]

    # Restore original-image scale for bounding boxes (YOLO downsamples)
    yolo_img_shape = np.array(img.shape[2:4])
    bboxes2D = YOLOv5.utils.general.scale_coords(yolo_img_shape,
                                                 bboxes2D_yolo_scale,
                                                 orig_img_shape)
    bboxes2D = bboxes2D.numpy().astype(int) # tensor --> numpy, int-ify for pix coords
    return bboxes2D, confidences
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
    objects = []
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

        obj_3d_bb = Oriented3dBoundingBox(obj_pt_cloud_center.squeeze(),
                                          eigvecs.T, # want basis vecs in columns
                                          pca_xmin, pca_xmax,
                                          pca_ymin, pca_ymax,
                                          pca_zmin, pca_zmax)
        objects.append(obj_3d_bb)
    # /for bbox2D

    return objects
# /detect3dObjects()

def renderDebugImg(img_name, disparity_field, bboxes2D, bboxes3D, K):
    '''Converts disparity_field into an image.
       Draws a yellow 2D bounding box around objects of interest.
       Projects 3D bounding box onto the image (rgb colors for xyz directions).'''

    img_data = (disparity_field / 192.0 * 255).astype(np.uint8)
    img_data = np.dstack([img_data] * 3) # mono --> color (BGR order for OpenCV)
    num_bboxes = len(bboxes2D)
    assert num_bboxes == len(bboxes3D)
    for n in range(num_bboxes):
        bb3D = bboxes3D[n]
        bb3D.draw(img_data, K)

        bb2D = bboxes2D[n]
        x1, y1, x2, y2 = bb2D
        cv.rectangle(img_data,
                    (x1, y1), (x2, y2),
                    color=[0,175,175], # BGR order for OpenCV
                    thickness=1, lineType=cv.LINE_AA)

    cv.imwrite(img_name, img_data)
# /renderDebugImg()

def processFrames(kitti, frame_range, psmnet, yolov5, use_cuda):

    B = kitti.calib.b_rgb # baseline, m
    K = kitti.calib.K_cam2.astype(np.float64) # intrinsics matrix for camera #2 (left stereo)

    for frame_num in frame_range:
        print('\nProcessing frame {}'.format(frame_num))

        #  pair of H x W x C=3 numpy arrays
        left_img, right_img = kitti.get_rgb(frame_num) # PIL images
        left_img, right_img = np.array(left_img), np.array(right_img) # PIL --> numpy

        # H x W numpy array of float
        disparity_field = \
            psmnetDisparityField(psmnet, left_img, right_img, use_cuda)

        # H x W array of 3D points in camera coords (x:right, y:down, z:forward)
        pt_cloud_field = pointCloud(disparity_field, B, K)

        # YOLOv5
        # - bboxes2D: (N,4) array of N 2D bounding boxes, each in (x1,y1,x2,y2) format
        # - confidences: 1D array of confidences for each BB (N values)
        bboxes2D, confidences = yolov5Detections(yolov5, left_img, use_cuda)

        # Save intermediate results, for debug
        with open(f'frame-{frame_num}.npz', 'wb') as f:
            np.savez(f, pt_cloud_field=pt_cloud_field,
                        bboxes2D=bboxes2D,
                        confidences=confidences)

        # list of Oriented3dBoundingBox
        bboxes3D = detect3dObjects(pt_cloud_field, bboxes2D)

        renderDebugImg(f'disparity-frame{frame_num}.png',
                       disparity_field,
                       bboxes2D, bboxes3D,
                       kitti.calib.K_cam2)

    # /for frame_num

# /processFrames()


def main():
    args = parseCmdline()
    kitti = pykitti.raw(args.base_dir, args.date, args.drive)
    psmnet = loadPsmnet(args.psmnet_pretrained_weights, args.use_cuda)
    yolov5 = loadYolov5(args.yolov5_pretrained_weights, args.use_cuda)
    processFrames(kitti, range(57,70), psmnet, yolov5, args.use_cuda)
# /main()

if __name__ == "__main__":
    main()