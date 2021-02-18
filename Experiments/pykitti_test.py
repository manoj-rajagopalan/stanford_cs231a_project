'''
    Learn how to use PyKITTI to access KITTI raw data
    KITTI raw data available at http://www.cvlibs.net/datasets/kitti/raw_data.php
    Download and example the raw_dataset_download_script at this link
    Observe that each drive has a bunch of '_sync' files that has sensor data but that all
    drives for any one day have a common calibration file. Both need to be unpacked into the
    directory organization shown below.

    Assumption: KITTI raw data is present as ${SOMEPATH}/<yyyy>_<mm>_<dd>/
    Example: /home/johndoe/CS231_Project/KITTI-2015/2021_01_23/

    After unzipping the '*_calib.zip' and '*_sync.zip' files, directory contents should look like the following:
    - calib_cam_to_cam.txt
    - calib_imu_to_velo.txt
    - calib_velo_to_cam.txt
    - 2021_01_23_drive_0005/
      - image_00/ (gray stereo left)
      - image_01/ (gray stereo right)
      - image_02/ (color stereo left)
      - image_03/ (color stereo right)
      - oxts/  (GPS + IMU)
      - velodyne_points/ (LIDAR)

    The use the following command line:
    python3 pykitti_test.py  /home/johndoe/CS231_Project/KITTI-2015   2021_01_23   0005
'''

import pykitti
import sys

assert len(sys.argv) == 4, "Usage: {} <base_dir> <date> <drive>"

base_dir = sys.argv[1]
date = sys.argv[2]
drive = sys.argv[3]
kitti = pykitti.raw(base_dir, date, drive)

print('Cam2 (color stereo left) intrinsics: ')
print(kitti.calib.K_cam2)

print('\nCam1 (gray stereo right) matrix (full): ')
print(kitti.calib.P_rect_10)

print('\nColor stereo baseline = ', kitti.calib.b_rgb, 'm')

# Calculate color stereo left camera focal length
color_stereo_left_cam_K = kitti.calib.K_cam2
# ... K[0,0] is f * k where f is focal length (along neg z-axis) and k is pixels/m along x-axis

# KITTI uses FL2-14S3C-C for its color stereo cameras (ref: http://www.cvlibs.net/datasets/kitti/setup.php)
# Googling for the datasheets, I got
# *  https://www.gophotonics.com/products/scientific-industrial-cameras/point-grey-research-inc/45-571-fl2-14s3c-c
# *  https://www.surplusgizmos.com/assets/images/flea2-FW-Datasheet.pdf 
color_stereo_left_cam_x_pixel_width = 4.65e-6 # m
color_stereo_left_cam_f = color_stereo_left_cam_K[0,0] * color_stereo_left_cam_x_pixel_width
print('\nLeft color stereo camera focal length = ', color_stereo_left_cam_f * 1000, 'mm')

# Try to comprehend what depths the stereo system can resolve
# depth = Baseline * focal_length / disparity
color_stereo_baseline = kitti.calib.b_rgb
color_stereo_disparity_for_1m_depth = color_stereo_baseline * color_stereo_left_cam_f / 1.0
# convert disparity from metres to pixels
color_stereo_disparity_pixels_for_1m_depth = \
    color_stereo_disparity_for_1m_depth / color_stereo_left_cam_x_pixel_width
print('\n1m depth gives a disparity of ', color_stereo_disparity_pixels_for_1m_depth, 'pixels')