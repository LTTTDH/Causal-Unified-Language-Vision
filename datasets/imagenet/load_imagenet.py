import datasets.registration.register_imagenet_cls  # register imagenet dataset
from datasets.dataset_mappers.imagenet_dataset_mapper import ImageNetDatasetMapper

from detectron2.data import build_detection_train_loader, build_detection_test_loader

from detectron2.config import get_cfg

cfg = get_cfg()

# cfg.INPUT.SIZE_TRAIN = 
# cfg.INPUT.SIZE_TEST = 
# cfg.INPUT.SIZE_CROP = 

cfg.DATASETS.TRAIN = "imagenet_train"
cfg.DATASETS.TEST = "imagenet_val"
# cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS =

cfg.SOLVER.IMS_PER_BATCH = 4
# cfg.DATALOADER.ASPECT_RATIO_GROUPING = 
# cfg.DATALOADER.NUM_WORKERS = 

train_loader = build_detection_train_loader(cfg, 
                                            mapper=ImageNetDatasetMapper(cfg,
                                                                         is_train=True,))

for batch in train_loader:
    for x in batch:
        for key, val in x.items():
            print(key,":", val)
        print()
    break


val_loader = build_detection_test_loader(cfg, 
                                         dataset_name=cfg.DATASETS.TEST,
                                         mapper=ImageNetDatasetMapper(cfg,
                                                                      is_train=False,))

for batch in val_loader:
    for x in batch:
        for key, val in x.items():
            print(key, ":", val)
        print()
    break