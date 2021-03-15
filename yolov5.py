import YOLOv5.models.experimental
import YOLOv5.utils.datasets
import YOLOv5.utils.general

import numpy as np
import torch

class YOLOnet:
    def __init__(self, yolov5_pretrained_weights, use_cuda):
        device = 'cuda:0' if use_cuda else 'cpu'
        self.yolov5 = \
            YOLOv5.models.experimental.attempt_load(yolov5_pretrained_weights,
                                                    map_location=device)
        self.yolov5.eval()
        if use_cuda:
            self.yolov5.cuda()
        else:
            self.yolov5.cpu()
    # /__init__()

    def _preprocess(self, img_in, use_cuda, stride=32, yolov5_img_size=640):
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
        # img = img.half() if use_cuda else img.float()  # uint8 to fp16/32
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img
    # /_preprocess()

    def detect(self, img, use_cuda):
        orig_img_shape = img.shape
        img = self._preprocess(img, use_cuda, stride=int(self.yolov5.stride.max()))
        yolov5_output = self.yolov5(img)[0]
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
        bboxes2D = bboxes2D.cpu().numpy().astype(int) # tensor --> numpy, int-ify for pix coords
        confidences = confidences.cpu()
        return bboxes2D, confidences
    # /detect()

# /class YOLOnet
