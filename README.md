# Multi-Spectral-Image-Synthesis-for-Crop-Weed-Segmentation-in-Precision-Farming
 In this work, we propose an alternative solution with respect to the common data augmentation techniques, applying it to the fundamental problem of crop/weed segmentation in precision farming. Starting from real images, we create semi-artificial samples by replacing the most relevant object classes (i.e., crop and weeds) with synthesized counterparts. To do that, we employ a conditional GAN (cGAN), where the generative model is trained by conditioning the shape of the generated object. Moreover, in addition to RGB data, we take into account also near-infrared information, generating four channel multi-spectral synthetic images.


<p align="center">
<img src="images/style.png" width="900"/><br>
</p>

## Dataset

Dataset used:

[Bonn Sugar Beets Dataset](http://www.ipb.uni-bonn.de/datasets_IJRR2017/annotations/cropweed/)

Annotations:

[Bonn Sugar Beets Annotations (not in the original dataset anymore)](https://gitlab.com/ibiscp/sugar_beet_annotation)

## Training

* Mask Image Generation

[![Mask Image Generation](https://i.ytimg.com/vi/v2xjxWj6xKI/1.jpg)](https://www.youtube.com/watch?v=v2xjxWj6xKI)

* RGB Image Generation

[![RGB Image Generation](https://i.ytimg.com/vi/6gSF-rcAYKI/1.jpg)](https://www.youtube.com/watch?v=6gSF-rcAYKI)

* NIR Image Generation

[![NIR Image Generation](https://i.ytimg.com/vi/v6mq-mdmbDI/1.jpg)](https://www.youtube.com/watch?v=v6mq-mdmbDI)


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
