import pykitti
import cv2 as cv
import numpy as np
import argparse
import sys

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
    args = parser.parse_args()
    return args
# /parse_cmdline()

def do_stereo_sgbm(kitti, frame_range):
    stereo_sgbm = cv.StereoSGBM_create(minDisparity=-320, numDisparities=640, blockSize=11, mode=cv.StereoSGBM_MODE_HH)
    for f in frame_range:
        print('Processing frame {}. Press any key.'.format(f))
        left_img, right_img = kitti.get_rgb(f)
        left_img = cv.cvtColor(np.array(left_img), cv.COLOR_RGB2BGR)
        right_img = cv.cvtColor(np.array(right_img), cv.COLOR_RGB2BGR)
        disparity_img = stereo_sgbm.compute(left_img, right_img)
        cv.imshow('Left', left_img)
        cv.imshow('Right', right_img)
        cv.imshow('Disparity', disparity_img)
        cv.waitKey(0)
    # /for f

    cv.destroyAllWindows()
# /do_stere_sgbm()

def main():
    print('Using OpenCV version ', cv.__version__)
    args = parse_cmdline()
    kitti = pykitti.raw(args.base_dir, args.date, args.drive)
    do_stereo_sgbm(kitti, range(50,70))
# /main()

if __name__ == "__main__":
    main()
