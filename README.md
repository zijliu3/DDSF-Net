# DDSF-Net: A Dual-Domain Collaborative Spatial–Frequency Network for Low-Light Image Enhancement

> This repository is directly associated with our manuscript **"DDSF-Net: A Dual-Domain Collaborative Spatial–Frequency Network for Low-Light Image Enhancement"**, which has been submitted to *The Visual Computer*.  
> At this stage, this repository provides a partial implementation and project documentation for academic communication and preliminary understanding.  
> To protect the originality of our work and ensure compliance with publication requirements, the **complete codebase will be released after the paper is officially accepted**.  
> Pretrained models will be released after publication.  
> If you find this work useful, please cite the corresponding paper once it becomes available.  
> We appreciate your understanding and support.

---

## 🔹 Framework Overview

The overall architecture of **DDSF-Net** is illustrated in the figure below.

DDSF-Net adopts an encoder–decoder architecture and integrates Fourier-domain global illumination modeling, wavelet-domain directional detail reconstruction, and adaptive cross-level feature fusion for efficient low-light image enhancement.

<p align="center">
  <img src="Figures/framework_cadf.png" alt="DDSF-Net Framework" width="85%">
  <br/>
  <em>Fig. 1: Overall architecture of DDSF-Net.</em>
</p>

---

## 🔹 Key Modules

The structures of the proposed key modules are shown below.

DDSF-Net mainly includes the following modules:

- **Fourier Enhancement Fusion Block (FEFB)** for efficient global illumination modeling and spatial–frequency feature collaboration.
- **Wavelet Reconstruction Attention Block (WRAB)** for direction-sensitive edge and texture reconstruction.
- **Context-Aware Detail Fusion (CADF)** for adaptive encoder–decoder feature fusion and degradation suppression.

<p align="center">
  <img src="Figures/wramfefb.png" alt="Key Modules of DDSF-Net" width="85%">
  <br/>
  <em>Fig. 2: Structures of the proposed WRAB and FEFB modules.</em>
</p>

---

## 🔹 Dataset

Please download the datasets from their official sources or commonly used public dataset repositories, and organize them according to the required directory structure.

This project uses the following datasets:

- **LOL-v1**: https://daooshee.github.io/BMVC2018website/
- **LOL-v2-Real**: https://github.com/flyywh/CVPR-2020-Semi-Low-Light
- **LIME**: https://sites.google.com/view/xjguo/lime
- **MEF**: https://ivc.uwaterloo.ca/database/MEF.html
- **NPE**: Please refer to common low-light image enhancement dataset collections.
- **DICM**: Please refer to common low-light image enhancement dataset collections.
- **VV**: https://sites.google.com/site/vonikakis/datasets
- **DarkFace**: https://flyywh.github.io/CVPRW2019LowLight/

The paired datasets, including **LOL-v1** and **LOL-v2-Real**, are used for quantitative evaluation with full-reference metrics. The unpaired datasets, including **LIME**, **MEF**, **NPE**, **DICM**, and **VV**, are used for qualitative and no-reference evaluation. **DarkFace** is used to evaluate the effectiveness of enhanced images for downstream low-light face detection.

---

## 🔹 Model Training

To train **DDSF-Net**, run:

```bash
python train.py
```

Please modify the dataset paths and training settings in the corresponding script before training.

---

## 🔹 Model Testing

To test **DDSF-Net**, run:

```bash
python test.py
```

The enhanced images will be saved in the specified output folder.

---

## 🔹 Pretrained Models

Pretrained models will be released after publication.

---

## 🔹 Visual Results

Representative qualitative comparison results are shown below. The results demonstrate the effectiveness of DDSF-Net in brightness enhancement, color restoration, and detail preservation under different low-light conditions.

<p align="center">
  <img src="Figures/result.jpg" alt="Visual comparison results on LOL-v1 and LOL-v2-Real" width="90%">
  <br/>
  <em>Fig. 3: Visual comparison results on LOL-v1 and LOL-v2-Real.</em>
</p>

<p align="center">
  <img src="Figures/result2.png" alt="Visual comparison results on unpaired low-light datasets" width="90%">
  <br/>
  <em>Fig. 4: Visual comparison results on unpaired low-light datasets.</em>
</p>

---

## 🔹 Citation

If you find this work useful for your research, please consider citing our paper once it becomes available.

```bibtex
@article{liu2026ddsfnet,
  title={DDSF-Net: A Dual-Domain Collaborative Spatial-Frequency Network for Low-Light Image Enhancement},
  author={Liu, Zijun and Hu, Peng and Chang, Renkai and Ma, Junjie},
  journal={The Visual Computer},
  year={2026}
}
```

The official citation information will be updated after publication.

---

## 🔹 Contact

If you have any questions, please feel free to contact us.

- Zijun Liu: zijliu3@gmail.com

---

## 🔹 Acknowledgement

This repository is built for academic research and communication. We thank the authors of the public datasets and open-source projects used in this work.
