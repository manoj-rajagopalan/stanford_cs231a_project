# Read video files in Python using OpenCV

import cv2 as cv
import sys
from tqdm import tqdm # nice progress bar on cmdline

print('OpenCV version: ', cv.__version__)
assert len(sys.argv) == 2, "Usage: {} <video file>".format(sys.argv[0])

video_file_reader = cv.VideoCapture(sys.argv[1]) # handles files also

for i in tqdm(range(200)):
    # print('Displaying frame ', (i+1))
    bool_status, img = video_file_reader.read()  # returns (bool, ndarray of shape (height, width, channels))
    assert bool_status
    if(i == 0):
        print('frame dims = ', img.shape)
    # /if

    # Experiment: overlay red box
    # OpenCV uses BGR img format so red is channel 2 not channel 0
    row0 = img.shape[0] // 4
    row1 = 3 * row0
    col0 = img.shape[1] // 4
    col1 = 3 * col0
    img[row0:row1, col0, 2] = 255
    img[row0:row1, col1, 2] = 255
    img[row0, col0:col1, 2] = 255
    img[row1, col0:col1, 2] = 255

    # Display frame in a window
    # Different frames can simultaneously be shown in different windows by simply using a different name
    cv.imshow('Frame', img) # syntax: window-name, data
    
    # required for flushing to screen, waits for given delay in ms (indefinitely if < 0)
    cv.waitKey(30) 
# for i

cv.destroyAllWindows()
