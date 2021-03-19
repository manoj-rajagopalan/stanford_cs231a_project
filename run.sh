
/usr/bin/python3 main.py \
                 --base_dir ../KITTI-2015/Raw_Data \
                 --date 2011_09_26 \
                 --drive 0001 \
                 --frames 0,108 \
                 --psmnet_pretrained_weights ../psmnet-pretrained_model_KITTI2015.tar \
                 --yolov5_pretrained_weights ../yolov5x.pt \
                 --output_dir ../Results/KITTI-2011_09_26-0001-OpenCV \
                 --use_opencv_for_stereo


