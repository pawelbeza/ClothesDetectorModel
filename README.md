# Clothes Detector Model
Convolutional Neural Network Model detecting clothes

# Architecture
Faster R-CNN with ResNet+FPN backbone
![faster-rcnn](https://user-images.githubusercontent.com/43823276/123557469-39824300-d791-11eb-83d2-07b70701cb7c.png)
## Literature
- Faster R-CNN https://arxiv.org/pdf/1506.01497.pdf
- ResNet https://arxiv.org/pdf/1512.03385.pdf
- FPN https://arxiv.org/pdf/1612.03144.pdf

# Setup
## Clone repository
```
git clone https://github.com/pawelbeza/ClothesDetectorModel.git
```
## Create conda environment
```
conda env create --file env.yml
conda activate clothes-detection
```
## Prepare dataset
1. Download DeepFashion2 dataset to *dataset* dir from [google drive](https://drive.google.com/drive/folders/125F48fsMBz2EF0Cpqk6aaHet5VH399Ok)
2. Create coco annotations 
```
mkdir dataset/detectron_annos  
python3 dataset/deepfashion2_to_coco.py --out_dir ./dataset/detectron_annos/train_annos_all.json --image_dir ./dataset/train/image --annotation_dir ./dataset/train/annos --num_images 191961  
python3 dataset/deepfashion2_to_coco.py --out_dir ./dataset/detectron_annos/validation_annos_all.json --image_dir ./dataset/validation/image --annotation_dir ./dataset/validation/annos --num_images 32153
```
## Training
```
python3 train_net.py --config configs/Faster_RCNN_R101_FPN_3x.yaml --resume
```
