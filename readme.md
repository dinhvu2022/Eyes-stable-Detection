# Nhận diện trạng thái mắt với YOLOV5-Face và DenseNet Model
## Installation

Python environment (recommended)
<!-- <details><summary> <b>Expand</b> </summary> -->

``` shell
pip install requirements.txt
```
## Train Eyes Classification
``` shell
python train_eyes_stable.py --train_dir /content/data/train --val_dir /content/data/test --save_weights /content/weights --batch_size 32 --epochs 5
```

## Run 
``` shell
# webcam
python main.py --source 0 --path_npy_file data/test.out --weights_eyes_stables weights/Eyes_stable_model_best_07-0.04.hdf5 --weights_face_reg weights/Embedding_DenseNet.hdf5

# video 
python main.py --source data/video-f1b7c41a-0add-4af1-adb7-2e7b8c7e4e67-1665057340.mp4 --path_npy_file data/test.out --weights_eyes_stables weights/Eyes_stable_model_best_07-0.04.hdf5 --weights_face_reg weights/Embedding_DenseNet.hdf5
```
