"""
Mask R-CNN
Configurations and data loading code for the MSLAQ dataset. Heavily modified 
code originally written by Waleed Abdulla for the COCO dataset.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: run from the command line as such:

    # Continue training a model that you had trained earlier
    python3 data_loader.py train --dataset=/path/to/dataset/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 data_loader.py train --dataset=/path/to/dataset/ --model=last

    # Run evaluatoin on the last model you trained
    python3 data_loader.py evaluate --dataset=/path/to/dataset/ --model=last
"""
import os
import time
import numpy as np
import h5py

from config import Config
import utils
import model as modellib

# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file
MODEL_PATH = os.path.join(ROOT_DIR, "lesion_mask_rcnn.pth")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################

class DataConfig(Config):
    """Configuration for training on MSLAQ data.
    Derives from the base Config class and overrides values specific to the way I've set up
    the hdf5 files
    """
    # Give the configuration a recognizable name
    NAME = "mask_hdf5"

    # We use one GPU with 12GB memory (I think), which can fit ~one image.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 16

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background). Small, medium, and large lesions (each is a 'class')
    NUM_CLASSES = 4  


############################################################
#  Dataset
############################################################

class MSDataset(utils.Dataset):
      def load_data(self, dataset_dir, config, class_ids=None,
                  class_map=None, return_object=False):
        """ TODO: Remove this and fix model.Dataset

        Load a subset of the dataset. TODO: Remove this and fix model.Dataset
        dataset_dir: The root directory of the dataset.
        subset: What to load (train, val)
        class_ids: TODO: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_object: TODO: If True, returns the object.
        """
        f = h5py(join(dataset_dir, 'det_data.h5py'))
        image_ids = list(f.keys())
        self._mode = config.get('mode')
        self._shuffle = config.get('shuffle', True)
        self._image_ids = np.asarray(image_ids)
        self._image_ids = [i[4:] for i in self._subjects]
        self._nb_folds = config.get('nb-folds', 10)

        fold_length = len(self._image_ids) // self._nb_folds
        self._image_ids = self._rotate(self._image_ids, config.get('fold', 0) * fold_length)
        train_idx = (self._nb_folds - 2) * fold_length
        valid_idx = (self._nb_folds - 1) * fold_length
        if self._mode == 'train':
            self._subjects = self._image_ids[:train_idx]
        elif self._mode == 'valid':
            self._subjects = self._image_ids[train_idx:valid_idx]
        elif self._mode == 'test':
            self._subjects = self._image_ids[valid_idx:]

        class_ids = ['small', 'medium', 'large']
        for size, idx in enumerate(class_ids):
            self.add_class("MSLAQ", idx, size)
        for i in image_ids:
            self.add_image("MSLAQ", image_id=i, path = os.path.join(dataset_dir, i))


############################################################
#  Evaluation
############################################################

def build_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls 
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": np.asfortranarray(mask)
            }
            results.append(result)
    return results


def evaluate(model, dataset, eval_type="bbox", limit=0, image_ids=None):
    """Runs official evaluation.
    dataset: A Dataset object with validation data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding image IDs.
    subj_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image])[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        image_results = build_coco_results(dataset, image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"], r["masks"])
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    Eval = eval(results, eval_type)
    Eval.params.imgIds = image_ids
    Eval.evaluate()
    Eval.accumulate()
    Eval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


#################################################################
#  Training - Possibly will start with COCO pretrained weights??
#################################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MSLAQ dataset.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate'")
    parser.add_argument('-d', '--dataset', required=True,
                        metavar="/path/to/mslaq.h5",
                        help='Directory of the dataset dataset - /usr/local/data/thomasc/unet_out/all_img)')
    #Will probably get rid of this:
    parser.add_argument('-c', '--config', required=True,
                        metavar="-c /path/to/config.json",
                        help='json Configuration File')
    parser.add_argument('-m', '--model', required=False,
                        metavar="-m /path/to/weights.pth",
                        help="Path to weights .pth file")
    parser.add_argument('-l' '--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="-l /path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=50,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=50)')
    
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = DataConfig()
    else:
        class InferenceConfig(DataConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(config=config,
                                  model_dir=args.logs)
    if config.GPU_COUNT:
        model = model.cuda()

    # Select weights file to load
    if args.model:
        if args.model.lower() == "last":
            model_path = model.find_last()[1]
        else:
            model_path = args.model
    else:
        model_path = ""

    
    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path)

    # Train or evaluate
    if args.command == "train":
        # Training dataset (possibly modify so some examples come from validation set as in MaskRCNN paper)

        dataset_train = MSDataset()
        dataset_train.load_data(dataset, {'mode': 'train', 'shuffle': True if config.SHUFFLE is 1 else False, 'dim': config.BRAIN_DIMENSIONS,'mods': config.MODALITIES})    
        dataset_train.prepare()

        # Validation dataset
        dataset_val = MSDataset()
        dataset_val.load_data(args.dataset, {'mode': 'valid', 'shuffle': False})
        dataset_val.prepare()

        # Training - Stage 1
        print("Training network heads")
        model.train_model(dataset_train, dataset_val, data_path,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads', config)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train_model(train_generator, valid_generator, data_path,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers='4+')

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train_model(train_generator, valid_generator, data_path,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all')

    elif args.command == "evaluate":
        # Validation dataset
        dataset_test = MSDataset()
        dataset_test.load_data(args.dataset, {'mode': 'test', 'shuffle': False})
        dataset_test.prepare()
        print("Running evaluation on {} images.".format(args.limit))
        evaluate(model, dataset_test, "bbox", limit=int(args.limit))
        evaluate(model, dataset_test, "segm", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
