# detection_net

- On blacktusk go to /usr/local/data/thomasc/unet_out/all_img and copy these files somewhere ('data_path')
- You can use the image net weights for now; go to usr/local/data/thomasc/for_paul and copy 'resnet50_imagenet.pth' to 'det_net' folder

Usage is:

'python launcher.py train --dataset=data_path ---model=imagenet'
