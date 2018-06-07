===========================================
Lesion Detection Net
===========================================

1. On blacktusk go to "/usr/local/data/thomasc/unet_out/all_img" and copy these files somewhere ("data_path")
2. You can use the imagenet weights for now; go to "usr/local/data/thomasc/checkpoints" and copy "resnet50_imagenet.pth" to "det_net" folder.
3. Make sure you're using pytorch0.3 (install with pip by ```pip install http://download.pytorch.org/whl/cu80/torch-0.3.1-cp35-cp35m-linux_x86_64.whl```)
4. ```pip install -r requirements.txt```

Usage is:
```python launcher.py train --dataset=data_path ---model=imagenet```


