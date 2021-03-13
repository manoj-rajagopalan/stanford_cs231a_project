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
    kmeans_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    kmeans_attempts = 10
    kmeans_flags = cv.KMEANS_RANDOM_CENTERS

    for bbox2D in bboxes2D:
        # Slightly pad the 2D bounding box before extracting points
        x1, y1, x2, y2 = bbox2D.astype(int)
        # x1 = max(0, x1 - 3) # inclusive
        # y1 = max(0, y1 - 3) # inclusive
        # x2 = min(W, x2 + 4) # exclusive
        # y2 = min(H, y2 + 4) # exclusive

        # extract
        obj_pt_cloud = pt_cloud_field[y1:y2, x1:x2, :] # Note swapped xy --> ij indexing
        obj_pt_cloud = obj_pt_cloud.reshape(-1, 3) # make single list of 3D points

        print('\t-- Clustering --')
        print('\t   Initial pt cloud size = ', len(obj_pt_cloud))

        # filter out ground plane (KITTI cameras at 1.65 m from ground)
        obj_pt_cloud = obj_pt_cloud[(obj_pt_cloud[:,1] < 1.6), :]
        print('\t   After removing ground plane = ', len(obj_pt_cloud))

        # 2-means clustering to separate background from object
        obj_pt_cloud_distances = \
            np.linalg.norm(obj_pt_cloud, axis=1).astype(np.float32)
            # ... OpenCV KMeans algo asserts on single-precision float

        compactness, labels, centers = \
            cv.kmeans(obj_pt_cloud_distances, 2, None, kmeans_criteria, kmeans_attempts, kmeans_flags)
        labels = labels.squeeze()
        assert len(centers) == 2
        obj_label = 0 if (centers[0] < centers[1]) else 1
        obj_indices = np.where(labels == obj_label)[0]
        obj_pt_cloud = obj_pt_cloud[obj_indices,:]
        print('\t   After clustering = ', len(obj_pt_cloud))

        # debug using axis-aligned bounding box
        use_axis_aligned_bb_for_debug = True
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

        else: # PCA
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
                                          eigvecs.T,
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
