import cv2 as cv

class OpencvStereoSGBM:
    def __init__(self, min_disparity, max_disparity, block_size=9):
        self.stereo_sgbm = \
            cv.StereoSGBM_create(minDisparity=min_disparity,
                                 numDisparities=(max_disparity - min_disparity + 1),
                                 blockSize=block_size,
                                 P1 = 8 * 3 * block_size * block_size,
                                 P2 = 32 * 5 * block_size * block_size,
                                 disp12MaxDiff = 1,
                                 uniquenessRatio = 10,
                                 speckleWindowSize = 100,
                                 speckleRange = 32)

    def disparityField(self, left_img, right_img):
        return self.stereo_sgbm.compute(left_img, right_img)

#/class OpencvStereoSGBM