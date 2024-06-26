# Multi-Spectral-Image-Synthesis-for-Crop-Weed-Segmentation-in-Precision-Farming
 In this work, we propose an alternative solution with respect to the common data augmentation techniques, applying it to the fundamental problem of crop/weed segmentation in precision farming. Starting from real images, we create semi-artificial samples by replacing the most relevant object classes (i.e., crop and weeds) with synthesized counterparts. To do that, we employ a conditional GAN (cGAN), where the generative model is trained by conditioning the shape of the generated object. Moreover, in addition to RGB data, we take into account also near-infrared information, generating four channel multi-spectral synthetic images.



## Dataset

Dataset used:

[Bonn Sugar Beets Dataset](http://www.ipb.uni-bonn.de/datasets_IJRR2017/annotations/cropweed/)

Annotations:

[Bonn Sugar Beets Annotations (not in the original dataset anymore)](https://gitlab.com/ibiscp/sugar_beet_annotation)

[Sunflower dataset ](http://www.dis.uniroma1.it/~labrococo/fsd/sunflowerdatasets.html)

[GAN Dataset](https://drive.google.com/drive/folders/1e5WU75RQvFrrzJLkzlz9Tkzv5JFPpQy2?usp=sharing)

## Instructions

* Generate dataset for GAN

`python preprocess.py --dataset_path --annotation_path --plant_type --dimension --background --blur`

* Train DCGAN network (GAN for mask images)

```
cd stage_1
python main.py --dataset_path
```

* Train SPADE network (GAN for RGB and NIR images)

```
cd stage_2
python main.py --dataset_path
```

* Generate dataser for Segmentation

```
cd stage_2
python create_dataset.py --dataset-path  --annotation_path --output_path --background --blur
```

* Train Segmentation

```
cd segmentation
python segmentation.py --dataset-path  --dataset_type
```
* Paper 

If you use this code in your project, please cite the associated paper:

M. Fawakherji, C. Potena, A. Pretto, D.D. Bloisi, and D. Nardi
Multi-Spectral Image Synthesis for Crop/Weed Segmentation in Precision Farming
In: Robotics and Autonomous Systems, Volume 146, December 2021
Paper: https://www.sciencedirect.com/science/article/pii/S0921889021001469
Pre-print on Arxiv: https://arxiv.org/abs/2009.05750


BibTeX:
@article{fppbnRAS2021,
title = {Multi-Spectral Image Synthesis for Crop/Weed Segmentation in Precision Farming},
author = {Mulham Fawakherji and Ciro Potena and Alberto Pretto and Domenico D. Bloisi and Daniele Nardi},
journal = {Robotics and Autonomous Systems},
volume = {146},
pages = {103861},
year = {2021},
issn = {0921-8890},
doi = {https://doi.org/10.1016/j.robot.2021.103861}}



