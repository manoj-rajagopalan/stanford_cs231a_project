import numpy as np

import torch
import torch.nn.functional
import torchvision.transforms

import PSMNet.models

# global preprocessing transform (from PSMNet Test_img.py)
_normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}
_inference_transform = \
   torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(**_normal_mean_var)])    

class PyramidalStereoMatchingNet:

    def __init__(self, psmnet_pretrained_weights, use_cuda, max_disp=192):
        self.psmnet = PSMNet.models.stackhourglass(maxdisp=max_disp)
        if use_cuda:
            state_dict = torch.load(psmnet_pretrained_weights)
        else:
            state_dict = torch.load(psmnet_pretrained_weights, map_location=torch.device('cpu'))

        self.psmnet = torch.nn.DataParallel(self.psmnet, device_ids=[0]) # required to be before load_state_dict()
        self.psmnet.load_state_dict(state_dict['state_dict'])
        self.psmnet.eval() # prep for inference
        if use_cuda:
            self.psmnet.cuda()
        else:
            self.psmnet.cpu()
    # /__init__()

    def _preprocess(self, imgL_in, imgR_in):
        
        # following steps from PSMNet Test_img.py
        imgL = _inference_transform(imgL_in)
        imgR = _inference_transform(imgR_in)

        # pad to width and height to 16 times
        top_pad = (16 - (imgL.shape[1] % 16)) % 16
        right_pad = (16 - (imgL.shape[2] % 16)) % 16

        imgL = torch.nn.functional.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
        imgR = torch.nn.functional.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)

        return imgL, imgR
    # /_preprocess()

    def disparityField(self, left_img, right_img, use_cuda):
        psmnet_left_img, psmnet_right_img = \
            self._preprocess(left_img, right_img)
        if use_cuda:
            psmnet_left_img.cuda()
            psmnet_right_img.cuda()
        with torch.no_grad():
            disparity_field = self.psmnet(psmnet_left_img, psmnet_right_img)

        disparity_field = torch.squeeze(disparity_field)
        disparity_field_cpu = disparity_field.data.cpu().numpy()
        print('\tmin disp = {}, max disp = {}'.format(np.min(disparity_field_cpu),
                                                    np.max(disparity_field_cpu)))
        return disparity_field_cpu
    # /disparityField()

# /class PyramidalStereoMatching
