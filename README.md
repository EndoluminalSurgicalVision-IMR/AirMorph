# Digitalized-Atlas-for-Pulmonary-Airway
> By Team of  Institute of Medical Robotics, Shanghai Jiao Tong University, Shanghai, China

<div align=center><img src="figs/example1.png"></div>


## Introduction
>> In this work, we introduce AirwayAtlas, a robust, end-to-end deep learning pipeline enabling fully automatic and comprehensive airway anatomical labeling at lobar, segmental, and subsegmental resolutions. To facilitate precise clinical interpretation, we further propose AirwaySign, a compact anatomical signature quantifying critical morphological airway features—including stenosis, ectasia, tortuosity, divergence, length, and complexity. Additionally, AirwayAtlas supports efficient automated branching pattern analysis, significantly enhancing bronchoscopic navigation planning and procedural safety.



## Usage
<div align=center><img src="figs/example3.png"></div>

### Binary Airway Modeling
Please refer to ```segmentator/airway_segmentator.py```.

### Airway Anatomical Modeling
Please refer to ```classifier/airway_classifier.py```.


Optionally, you can use the script for a quick start:

```
python airwayatlas_pipeline.py
```

### Airway Signature

<div align=center><img src="figs/example2.png"></div>

The morphological airway signatures can be found in ```features/airway_morph_features.py```.

Optionally, you can use the script for a quick start:

```
python airwaysign_pipeline.py
```

### Airway BranchingPattern

<div align=center><img src="figs/example4.png"></div>

Please refer to ```branchingpattern/airwaybranchpattern_pipeline.py```.


### Pretraind Model
The pretrained model could be accessed by this [link](https://drive.google.com/drive/folders/1T6VwUnHSkWzL7ghkImbWTqk6SGB-pan-?usp=sharing)

### Sample Data
The sample data could be accessed by this [link](https://drive.google.com/drive/folders/1CvkkL_EP1QcgvKiNIt7I_Yypij1ibflq?usp=sharing)


## Citation
If you find this repository or our paper useful, please consider citing our paper:

```bibTex
@article{zhang2024digitalized,
  title={A Digitalized Atlas for Pulmonary Airway},
  author={Zhang, Minghui and Li, Chenyu and Zhang, Hanxiao and Liu, Yaoyu and Gu, Yun},
  journal={arXiv preprint arXiv:2412.11039},
  year={2024}
}
```