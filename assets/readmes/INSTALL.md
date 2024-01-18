# Installation Guide

**Installation**

```sh
# Python Package Installation
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r assets/requirements/requirements.txt --upgrade
pip install -r assets/requirements/requirements_custom.txt --upgrade
```

**Evaluation Tool**
```sh
# save coco_caption.zip to .xdecoder_data
wget https://huggingface.co/xdecoder/X-Decoder/resolve/main/coco_caption.zip
unzip coco_caption.zip
```

**Environment Variables**
```sh
export DETECTRON2_DATASETS=/pth/to/xdecoder_data
export DATASET=/pth/to/xdecoder_data
export DATASET2=/pth/to/xdecoder_data
export VLDATASET=/pth/to/xdecoder_data
export PATH=$PATH:/pth/to/xdecoder_data/coco_caption/jre1.8.0_321/bin
export PYTHONPATH=$PYTHONPATH:/pth/to/xdecoder_data/coco_caption
```