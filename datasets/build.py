# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.

import os
import itertools
import logging
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.utils.data
import torch.utils.data as torchdata

import detectron2.utils.comm as comm
from detectron2.data.build import (
    build_batch_data_loader,
    load_proposals_into_dataset,
    trivial_batch_collator,
)
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    verify_results,
)
from fvcore.common.config import CfgNode

from .dataset_mappers import *
from .evaluation import (InstanceSegEvaluator, 
                         ClassificationEvaluator, 
                         SemSegEvaluator, 
                         RetrievalEvaluator, 
                         CaptioningEvaluator, 
                         COCOPanopticEvaluator,
                         GroundingEvaluator,
                         InteractiveEvaluator,
                         VQAEvaluator
)
from modeling.utils import configurable
from utils.distributed import get_world_size

def filter_images_with_only_crowd_annotations(dataset_dicts, dataset_names):
    """
    Filter out images with none annotations or only crowd annotations
    (i.e., images without non-crowd annotations).
    A common training-time preprocessing on COCO dataset.

    Args:
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.

    Returns:
        list[dict]: the same format, but filtered.
    """
    num_before = len(dataset_dicts)

    def valid(anns):
        for ann in anns:
            if isinstance(ann, list):
                for instance in ann:
                    if instance.get("iscrowd", 0) == 0:
                        return True
            else:
                if ann.get("iscrowd", 0) == 0:
                    return True
        return False

    dataset_dicts = [x for x in dataset_dicts if valid(x["annotations"])]
    num_after = len(dataset_dicts)
    logger = logging.getLogger(__name__)
    logger.info(
        "Removed {} images with no usable annotations. {} images left.".format(
            num_before - num_after, num_after
        )
    )
    return dataset_dicts


def get_detection_dataset_dicts(
    dataset_names, filter_empty=True, proposal_files=None
):
    """
    Load and prepare dataset dicts for instance detection/segmentation and semantic segmentation.

    Args:
        dataset_names (str or list[str]): a dataset name or a list of dataset names
        filter_empty (bool): whether to filter out images without instance annotations
        proposal_files (list[str]): if given, a list of object proposal files
            that match each dataset in `dataset_names`.

    Returns:
        list[dict]: a list of dicts following the standard dataset dict format.
    """
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    assert len(dataset_names)
    
    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in dataset_names]
    for dataset_name, dicts in zip(dataset_names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    if proposal_files is not None:
        assert len(dataset_names) == len(proposal_files)
        # load precomputed proposals from proposal files
        dataset_dicts = [
            load_proposals_into_dataset(dataset_i_dicts, proposal_file)
            for dataset_i_dicts, proposal_file in zip(dataset_dicts, proposal_files)
        ]

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    has_instances = "annotations" in dataset_dicts[0]
    if filter_empty and has_instances:
        dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts, dataset_names)

    assert len(dataset_dicts), "No valid data found in {}.".format(",".join(dataset_names))
    return dataset_dicts


def _test_loader_from_config(cfg, dataset_name, mapper=None):
    """
    Uses the given `dataset_name` argument (instead of the names in cfg), because the
    standard practice is to evaluate each test set individually (not combining them).
    """
    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]

    dataset = get_detection_dataset_dicts(
        dataset_name,
        filter_empty=False,
        proposal_files=None,
    )
    if mapper is None:
        mapper_cfg = CfgNode({'INPUT': cfg['INPUT'], 'MODEL': cfg['MODEL'], 'DATASETS': cfg['DATASETS']})
        mapper = DatasetMapper(mapper_cfg, False)
    assert cfg['TEST']['BATCH_SIZE_TOTAL'] % get_world_size() == 0, "Evaluation total batchsize is not divisible by gpu number"
    batch_size = cfg['TEST']['BATCH_SIZE_TOTAL'] // get_world_size()

    return {
        "dataset": dataset,
        "mapper": mapper,
        "num_workers": cfg['DATALOADER']['NUM_WORKERS'],
        "sampler": None,
        "batch_size": batch_size,
    }


@configurable(from_config=_test_loader_from_config)
def build_detection_test_loader(
    dataset: Union[List[Any], torchdata.Dataset],
    *,
    mapper: Callable[[Dict[str, Any]], Any],
    sampler: Optional[torchdata.Sampler] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    collate_fn: Optional[Callable[[List[Any]], Any]] = None,
) -> torchdata.DataLoader:

    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    return torchdata.DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
    )


def _train_loader_from_config(cfg, dataset_name, mapper, *, dataset=None, sampler=None):
    cfg_datasets = cfg['DATASETS']
    cfg_dataloader = cfg['DATALOADER']
    
    if dataset is None:
        dataset = get_detection_dataset_dicts(
            dataset_name,
            filter_empty=cfg_dataloader['FILTER_EMPTY_ANNOTATIONS'],
            proposal_files=cfg_datasets['PROPOSAL_FILES_TRAIN'] if cfg_dataloader['LOAD_PROPOSALS'] else None,
        )

    if mapper is None:
        mapper = DatasetMapper(cfg, True)

    return {
        "dataset": dataset,
        "sampler": sampler,
        "mapper": mapper,
        "batch_size": cfg['TRAIN']['BATCH_SIZE_PER_GPU'],
        "aspect_ratio_grouping": cfg_dataloader['ASPECT_RATIO_GROUPING'],
        "num_workers": cfg_dataloader['NUM_WORKERS'],
    }


@configurable(from_config=_train_loader_from_config)
def build_detection_train_loader(
    dataset, *, mapper, sampler=None, batch_size, aspect_ratio_grouping=True, num_workers=0
):
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    return torchdata.DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator
    )


def get_config_from_name(cfg, dataset_name):
    # adjust config according to dataset
    if 'refcoco' in dataset_name:
        cfg.update(cfg['REF'])
        return cfg
    elif 'cocomini' in dataset_name:
        cfg.update(cfg['DAVIS'])
        return cfg
    elif 'ytvos' in dataset_name:
        cfg.update(cfg['VOS'])
        return cfg
    elif 'ade600' in dataset_name:
        cfg.update(cfg['DAVIS'])
        return cfg
    elif 'openimage600' in dataset_name:
        cfg.update(cfg['DAVIS'])
        return cfg
    elif 'ade' in dataset_name:
        if 'ADE20K' in cfg.keys():
            cfg.update(cfg['ADE20K'])
        return cfg
    elif 'imagenet' in dataset_name:
        if 'IMAGENET' in cfg.keys():
            cfg.update(cfg['IMAGENET'])
        return cfg
    elif 'vlp' in dataset_name:
        cfg.update(cfg['VLP'])
        return cfg
    elif 'coco' in dataset_name:
        if 'COCO' in cfg.keys():
            cfg.update(cfg['COCO'])
        return cfg
    elif 'voc' in dataset_name:
        cfg.update(cfg['VOC'])
        return cfg
    elif 'context' in dataset_name:
        cfg.update(cfg['CONTEXT'])
        return cfg
    elif 'sun' in dataset_name:
        cfg.update(cfg['SUN'])
        return cfg
    elif 'scan' in dataset_name:
        cfg.update(cfg['SCAN'])
        return cfg
    elif 'cityscape' in dataset_name:
        cfg.update(cfg['CITY'])
        return cfg
    elif 'bdd' in dataset_name:
        cfg.update(cfg['BDD'])
        return cfg
    elif 'tsv' in dataset_name:
        cfg.update(cfg['TSV'])
        return cfg
    elif 'phrasecut' in dataset_name:
        cfg.update(cfg['PHRASE'])
        return cfg
    elif 'object365' in dataset_name:
        cfg.update(cfg['OBJECT365'])
        return cfg
    elif 'openimage' in dataset_name:
        cfg.update(cfg['OPENIMAGE'])
        return cfg
    elif 'lvis' in dataset_name:
        cfg.update(cfg['LVIS'])
        return cfg
    elif 'seginw' in dataset_name:
        cfg.update(cfg['SEGINW'])
        return cfg
    elif 'sbd' in dataset_name:
        cfg.update(cfg['SBD'])
        return cfg
    elif 'davis' in dataset_name:
        cfg.update(cfg['DAVIS'])
        return cfg
    elif 'sam' in dataset_name:
        cfg.update(cfg['SAM'])
        return cfg
    elif 'vqa' in dataset_name:
        cfg.update(cfg['VQA'])
        return cfg
    elif 'instruction' in dataset_name:
        cfg.update(cfg['INSTRUCT'])
        return cfg
    elif 'instp' in dataset_name:
        cfg.update(cfg['INSTP'])
        return cfg
    elif 'sharegpt' in dataset_name:
        cfg.update(cfg['SHAREGPT'])
        return cfg
    else:
        assert False, "dataset not support."


def build_eval_dataloader(cfg, ):
    dataloaders = []
    for dataset_name in cfg['DATASETS']['TEST']:
        cfg = get_config_from_name(cfg, dataset_name)
        # adjust mapper according to dataset
        if dataset_name == 'imagenet_val':
            mapper = ImageNetDatasetMapper(cfg, False)
        elif dataset_name == 'bdd10k_val_sem_seg':
            mapper = BDDSemDatasetMapper(cfg, False)
        elif dataset_name in ["vlp_val", "vlp_captioning_val", "vlp_val2017", "vlp_captioning_val2017"]:
            mapper = VLPreDatasetMapper(cfg, False, dataset_name)
        elif dataset_name in ["scannet_21_val_seg", "scannet_38_val_seg", "scannet_41_val_seg"]:
            mapper = ScanNetSegDatasetMapper(cfg, False)
        elif dataset_name in ["scannet_21_panoptic_val", 'bdd10k_40_panoptic_val']:
            mapper = ScanNetPanoDatasetMapper(cfg, False)
        elif 'sun' in dataset_name:
            mapper = SunRGBDSegDatasetMapper(cfg, False)
        elif 'refcoco' in dataset_name:
            mapper = RefCOCODatasetMapper(cfg, False)
        elif dataset_name in ["vqav2_test", "vqav2_test-dev", "vqav2_val"]:
            mapper = VQADatasetMapper(cfg, False, dataset_name)
        elif dataset_name in ["instruction_val", "instruction_captioning_val", "instruction_val2017", "instruction_captioning_val2017"]:
            mapper = InstructionDatasetMapper(cfg, False, dataset_name)
        elif dataset_name in ["instp_val", "instp_captioning_val", "instp_val2017", "instp_captioning_val2017"]:
            mapper = InstPreDatasetMapper(cfg, False, dataset_name)
        elif dataset_name in ["sharegpt4v"]:
            mapper = ShareGPTDatasetMapper(cfg, False, dataset_name)
        else:
            mapper = None
        dataloaders += [build_detection_test_loader(cfg, dataset_name, mapper=mapper)]
    return dataloaders


def build_train_dataloader(cfg, ):
    dataset_names = cfg['DATASETS']['TRAIN']
    
    for dataset_name in dataset_names:
        cfg = get_config_from_name(cfg, dataset_name)
        mapper_name = cfg['INPUT']['DATASET_MAPPER_NAME']
        # Semantic segmentation dataset mapper
        if mapper_name == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            loaders = build_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)
        # Panoptic segmentation dataset mapper
        elif mapper_name == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
            loaders = build_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)
        # Instance segmentation dataset mapper
        elif mapper_name == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
            loaders = build_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)
        # coco instance segmentation lsj new baseline
        elif mapper_name == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            loaders = build_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)
        # coco panoptic segmentation lsj new baseline
        elif mapper_name == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            loaders = build_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)
        elif mapper_name == "vlpretrain":
            mapper = VLPreDatasetMapper(cfg, True, dataset_name)
            loaders = build_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)
        elif mapper_name == "refcoco":
            mapper = RefCOCODatasetMapper(cfg, True)
            loaders = build_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)
        elif mapper_name == "instruction_train":
            mapper = InstructionDatasetMapper(cfg, True, dataset_name)
            loaders = build_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)
        elif mapper_name == "instp_train":
            mapper = InstPreDatasetMapper(cfg, True, dataset_name)
            loaders = build_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)
        elif mapper_name == "sharegpt":
            mapper = ShareGPTDatasetMapper(cfg, True, dataset_name)
            loaders = build_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)
        else:
            mapper = None
            loaders = build_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)

        return loaders

    
def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each
    builtin dataset. For your own dataset, you can simply create an
    evaluator manually in your script and do not have to worry about the
    hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg["SAVE_DIR"], "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

    # semantic segmentation
    if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    # instance segmentation
    if evaluator_type == "coco":
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    
    cfg_model_decoder_test = cfg["MODEL"]["DECODER"]["TEST"]
    # panoptic segmentation
    if evaluator_type in [
        "coco_panoptic_seg",
        "ade20k_panoptic_seg",
        "cityscapes_panoptic_seg",
        "mapillary_vistas_panoptic_seg",
        "scannet_panoptic_seg",
        "bdd_panoptic_pano"
    ]:
        if cfg_model_decoder_test["PANOPTIC_ON"]:
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    # COCO
    if (evaluator_type == "coco_panoptic_seg" and cfg_model_decoder_test["INSTANCE_ON"]) or evaluator_type == "object365_od":
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if (evaluator_type == "coco_panoptic_seg" and cfg_model_decoder_test["SEMANTIC_ON"]) or evaluator_type == "coco_sem_seg":
        evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
    # Mapillary Vistas
    if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg_model_decoder_test["INSTANCE_ON"]:
        evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg_model_decoder_test["SEMANTIC_ON"]:
        evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
    # Cityscapes
    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "cityscapes_panoptic_seg":
        if cfg_model_decoder_test["SEMANTIC_ON"]:
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
        if cfg_model_decoder_test["INSTANCE_ON"]:
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
    # ADE20K
    if evaluator_type == "ade20k_panoptic_seg" and cfg_model_decoder_test["INSTANCE_ON"]:
        evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
    # SEGINW
    if evaluator_type == "seginw" and cfg_model_decoder_test["INSTANCE_ON"]:
        evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
    # LVIS
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    # Classification
    if evaluator_type == "classification":
        evaluator_list.append(ClassificationEvaluator(dataset_name, output_folder))
    # Retrieval
    if evaluator_type in ["retrieval"]:
        evaluator_list.append(RetrievalEvaluator(dataset_name, output_folder, cfg['MODEL']['DECODER']['RETRIEVAL']['ENSEMBLE']))
    if evaluator_type == "captioning":
        evaluator_list.append(CaptioningEvaluator(dataset_name, output_folder, MetadataCatalog.get(dataset_name).gt_json))
    if evaluator_type in ["grounding_refcoco", "grounding_phrasecut", "grounding_spatial", "grounding_entity"]:
        evaluator_list.append(GroundingEvaluator(dataset_name))
    # VQAv2
    if evaluator_type == "vqa":
        evaluator_list.append(VQAEvaluator(dataset_name, output_dir=output_folder))
    # Interactive
    if evaluator_type in ["interactive", "interactive_grounding"]:
        evaluator_list.append(InteractiveEvaluator(dataset_name, output_dir=output_folder, max_clicks=cfg['STROKE_SAMPLER']['EVAL']['MAX_ITER'], iou_iter=cfg['STROKE_SAMPLER']['EVAL']['IOU_ITER']))

    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(
                dataset_name, evaluator_type
            )
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
        
    
    return DatasetEvaluators(evaluator_list)