# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import os
import glob
from typing import List, Tuple, Union

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager

from utils.constants import IMAGENET_CLASSES, IMAGENET_FOLDER_NAMES

__all__ = ["load_imagenet_images", "register_imagenet"]


def load_imagenet_images(dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]]):
    """
    Load ImageNet annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    image_folders = sorted(glob.glob(os.path.join(dirname, split, 'n*')))       # glob.glob(): return a list of directories with a specified pattern

    dicts = []
    for image_folder in image_folders:
        folder_name = image_folder.split('/')[-1]                               # n*
        image_pths = sorted(glob.glob(os.path.join(image_folder, "*.JPEG")))    # all image directories in n* folder
        for img_pth in image_pths:
            r = {
                "file_name": img_pth,
                "class_name": IMAGENET_CLASSES[IMAGENET_FOLDER_NAMES.index(folder_name)],   # str
                "class_id": IMAGENET_FOLDER_NAMES.index(folder_name),                       # int
            }
            dicts.append(r)
    return dicts


def register_imagenet(name, dirname, split, year, class_names=IMAGENET_CLASSES):
    DatasetCatalog.register(name, lambda: load_imagenet_images(dirname, split, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )


def register_all_imagenet(root):
    SPLITS = [
            ("imagenet_train", "imagenet-2012", "train", "2012"),
            ("imagenet_val", "imagenet-2012", "val", "2012"),
        ]
    for name, dirname, split, year in SPLITS:
        register_imagenet(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "classification"


_root = os.getenv("DATASET", "datasets")    # returns the value of the environment key if exists otherwise returns the default value(second argument) 
register_all_imagenet(_root)                # _root = "/mnt/hard2/lbk-cvpr/dataset"