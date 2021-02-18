'''
    Example to show how to invoke OpenCV's stereo-matching algorithms on arbitrary stereo image-pairs
'''
import cv2 as cv
import numpy as np
import sys

def main():
    assert len(sys.argv) == 3, "Usage: {}  <left image>  <right image>".format(sys.argv[0])
    l_img = cv.imread(sys.argv[1])
    r_img = cv.imread(sys.argv[2])
    assert l_img.shape == r_img.shape
    print('image shape = ', l_img.shape)

    # Semi-Global Block Matching
    print('Doing SGBM ...')
    stereo_sgbm = cv.StereoSGBM_create(minDisparity=-32, numDisparities=64, blockSize=11)
    disparities = stereo_sgbm.compute(l_img, r_img)
    cv.imwrite('disparity-sgbm.png', disparities)

    # BM = Block Matching?
    print('Doing BM ...')
    stereo_bm = cv.StereoBM_create(numDisparities=64, blockSize=11)
    # BM only works on grayscale, so run a crude method to make grayscale
    # cv.cvtColor() can also be used but this is just a quick-and-dirty experiment
    disparities = stereo_bm.compute(np.mean(l_img, axis=2, dtype=np.uint8),
                                    np.mean(r_img, axis=2, dtype=np.uint8))
    cv.imwrite('disparity-bm.png', disparities)

# /main()

if __name__ == "__main__":
    main()

