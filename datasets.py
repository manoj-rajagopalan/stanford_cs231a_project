import numpy as np
import pykitti
import cv2 as cv
class KittiRawDataset:
    def __init__(self, base_dir, date, drive, frames):
        self.base_dir = base_dir
        self.date = date
        self.drive = drive
        self.first_frame = frames[0]
        self.last_frame_plus_1 = frames[1]

        self.kitti = pykitti.raw(base_dir, date, drive)

    def __iter__(self):
        self.i_frame = self.first_frame
        return self

    def __next__(self):
        if self.i_frame == self.last_frame_plus_1:
            raise StopIteration
        else:
            baseline = self.kitti.calib.b_rgb
            K = self.kitti.calib.K_cam2
            left_img, right_img = self.kitti.get_rgb(self.i_frame) # PIL images
            left_img, right_img = np.array(left_img), np.array(right_img) # PIL --> numpy

            self.i_frame += 1
            return (self.i_frame-1, left_img, right_img, baseline, K)

    def frameNum(self):
        return self.i_frame
#/class KittiRawDataset

class KittDetectionDataset:
    def __init__(self, dir):
        self.dir = dir

    def __iter__(self):
        self.i_frame = 0
        return self

    def __next__(self):
        if self.i_frame == 7840:
            raise StopIteration

        else:
            left_img = cv.imread(self.dir + f'/image_2/{self.counter:06}.png')
            right_img = cv.imread(self.dir + f'/image_3/{self.counter:06}.png')
            left_img = np.array(left_img)
            right_img = np.array(right_img)

            calib_file = self.dir + f'/calib_2/{self.counter:06}.txt'
            baseline, K = _load_calib(calib_file)

            self.i_frame += 1
            return (self.i_frame-1, left_img, right_img, baseline, K)
    #/__next__()

    def frameNum(self):
        return self.i_frame

    def _load_calib(self, calib_file):
        # TODO - was not clear how to extract baseline from calib files (poor documentation)
        pass
#/class KittiDetectionDataset
