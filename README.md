===========================================
Lesion Detection Net
===========================================
This is a pytorch implementation of Mask R-CNN that is largely based on multimodallearning's [pytorch-mask-rcnn](https://github.com/multimodallearning/pytorch-mask-rcnn) (a pytorch implementation) and matterport's [Mask_RCNN](https://github.com/matterport/Mask_RCNN).

1. Clone this repository.
```git clone https://github.com/thomaschristinck/detection_net```
2. On blacktusk go to "/usr/local/data/thomasc/unet_out/all_img" and copy these files somewhere ("data_path")
3. You can use the imagenet weights for now; go to "usr/local/data/thomasc/checkpoints" and copy "resnet50_imagenet.pth" to "det_net" folder.
4. Make sure you're using pytorch0.3 (install with pip by ```pip install http://download.pytorch.org/whl/cu80/torch-0.3.1-cp35-cp35m-linux_x86_64.whl```). Later I'll hopefully upgrade everything to 0.4.
5. ```pip install -r requirements.txt```
6.  We use two repositories (non-maximum suppression from ruotianluo's [pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn) and longcw's [RoiAlign](https://github.com/longcw/RoIAlign.pytorch)) that need to be built with the right --arch option for cuda support.

    | GPU | arch |
    | --- | --- |
    | TitanX | sm_52 |
    | Tesla K40C | sm_30 |
    | GTX 1070 | sm_61 |
    | GTX 1080 (Ti) | sm_61 |

        cd nms/src/cuda/
        nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=[arch]
        cd ../../
        python build.py
        cd ../

        cd roialign/roi_align/src/cuda/
        nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=[arch]
        cd ../../
        python build.py
        cd ../../


Training usage is:
```python launcher.py train --dataset=data_path ---model=imagenet```


