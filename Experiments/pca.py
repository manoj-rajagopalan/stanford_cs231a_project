import sys
import numpy as np

import cv2 as cv

class Oriented3dBoundingBox:
    def __init__(self, origin, axes, xmin, xmax, ymin, ymax, zmin, zmax):
        self.origin = origin
        self.obj_to_cam = axes.T # col vectors are orthoNORMAL vectors
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
# /class Oriented3dBoundingBox

def detect3dObjects(pt_cloud_field, bboxes2D):
    H, W = pt_cloud_field.shape[0:2]
    objects = []
    for bbox2D in bboxes2D:
        # Slightly pad the 2D bounding box before extracting points
        x1, y1, x2, y2 = bbox2D.astype(int)
        x1 = max(0, x1 - 3) # inclusive
        y1 = max(0, y1 - 3) # inclusive
        x2 = min(W, x2 + 4) # exclusive
        y2 = min(H, y2 + 4) # exclusive

        # extract
        obj_pt_cloud = pt_cloud_field[x1:x2, y1:y2, :]
        obj_pt_cloud = obj_pt_cloud.reshape(-1, 3) # make single list of 3D points

        # filter out background
        obj_pt_cloud = obj_pt_cloud[(obj_pt_cloud[:,2] > -200.0), :]

        # PCA
        obj_pt_cloud_center, eigvecs = cv.PCACompute(obj_pt_cloud, mean=None)
        obj_pt_cloud_in_pca_space = cv.PCAProject(obj_pt_cloud, obj_pt_cloud_center, eigvecs)
        pca_xmin = np.min(obj_pt_cloud_in_pca_space[:,0])
        pca_xmax = np.max(obj_pt_cloud_in_pca_space[:,0])
        pca_ymin = np.min(obj_pt_cloud_in_pca_space[:,1])
        pca_ymax = np.max(obj_pt_cloud_in_pca_space[:,1])
        pca_zmin = np.min(obj_pt_cloud_in_pca_space[:,2])
        pca_zmax = np.max(obj_pt_cloud_in_pca_space[:,2])
        obj_3d_bb = Oriented3dBoundingBox(eigvecs,
                                          pca_xmin, pca_xmax,
                                          pca_ymin, pca_ymax,
                                          pca_zmin, pca_zmax)
        objects.append(obj_3d_bb)
    # /for bbox2D

    return objects
# /detect3dObjects()

def main():
    with open(sys.argv[1], 'rb') as f:
        npzfile = np.load(f)
        pt_cloud_field = npzfile['pt_cloud_field']
        bboxes2D = npzfile['bboxes2D']
        confidences = npzfile['confidences']
    
    obj_3d = detect3dObjects(pt_cloud_field, bboxes2D)
# /main()

if __name__ == "__main__":
    main()
