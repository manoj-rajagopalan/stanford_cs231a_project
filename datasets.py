import numpy as np
import pykitti

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