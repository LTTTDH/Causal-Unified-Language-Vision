import copy
import logging

import io
import os
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
import transformers

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import MetadataCatalog
from typing import Dict, Optional, Sequence, List


from modeling.language.LangEncoder import build_tokenizer
from modeling.utils import configurable
from modeling.language.LangEncoder.constant import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from modeling.language.LangEncoder import conversation as conversation_lib
from modeling.language.LangEncoder.mm_utils import tokenizer_image_token


__all__ = ["VQADatasetMapper"]
_root = os.getenv("DATASET2", "datasets") #may need a different root name?

def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    # The scope of vlp dataset may not need any augmentation.
    cfg_input = cfg['INPUT']
    image_size = cfg_input['IMAGE_SIZE']
    augmentation = []

    augmentation.extend([
        T.Resize((image_size, image_size)),
    ])
    
    return augmentation

# This is specifically designed for the COCO dataset.
class VQADatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        dataset_name=None,
        *,
        tfm_gens,
        image_format,
        tokenizer=None,
        max_token_num=None,
        num_classes=None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            crop_gen: crop augmentation
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[PretrainDatasetMapper] Full TransformGens used in training: {}".format(
                str(self.tfm_gens)
            )
        )

        self.img_format = image_format
        self.is_train = is_train

        self.all_arrows = MetadataCatalog.get(dataset_name).arrows

        self.tokenizer = tokenizer
        self.max_token_num = max_token_num
        self.num_classes = num_classes

    @classmethod
    def from_config(cls, cfg, is_train=True, dataset_name=None):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        tokenizer = build_tokenizer(cfg['MODEL']['TEXT'])
        max_token_num = 1024

        ret = {
            "is_train": is_train,
            "dataset_name": dataset_name,
            "tfm_gens": tfm_gens,
            "image_format": cfg['INPUT']['FORMAT'],
            "tokenizer": tokenizer,
            "max_token_num": max_token_num,
            "num_classes": cfg['VQA']['INPUT']['NUM_CLASSES'],
        }
        return ret

    def get_image(self, split, inp):
        image_file = "VQAv2/{}/COCO_{}_".format(split, split) + '{0:012d}'.format(int(inp)) + ".jpg"
        image = Image.open(os.path.join(_root, image_file)).convert('RGB')
        return image
        
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = self.get_image(self.all_arrows[0]['data_subtype'], 
                               self.all_arrows[0]['questions'][dataset_dict['cur_id']]['image_id'])

        image = utils._apply_exif_orientation(image)
        image = utils.convert_PIL_to_numpy(image, self.img_format)
        utils.check_image_size(dataset_dict, image)

        # image, transforms = T.apply_transform_gens(self.tfm_gens, image)

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        return dataset_dict