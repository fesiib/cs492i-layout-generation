# Structure-aware Layout Generation

### Team 9: Bekzat Tilekbay, Shyngys Aitkazinov

## Installation

1. Clone this repository

    ```bash
    git clone https://github.com/fesiib/cs492i-layout-generation.git
    cd cs492i-layout-generation
    ```

2. Create a new [conda](https://docs.conda.io/en/latest/) environment (Python 3.8)

    ```bash
    conda env create -f environment.yml
    conda activate layout-generation
    ```

3. Change the directories appropriately in `train` and `test` files in each `src`
We assume that all the pretrained models are in `results` folder. Avoid using prefix `trial_`, as it might get deleted while training

## Development environment

-   Ubuntu 18.04, CUDA 11.3

## Test

Access one of `src_*` and run `test.ipynb`

## Train

Let `SRC` be one of `src_lstm, src_transformer` and `TRAIN` be one of `train_*.py`

```
python SRC/train TRAIN
```

Checkpoints with metavariables will be saved in folder `./results`

## Models
| Models            | Epochs|      Link     | Comments |
| ----------------- | ----: | :-----------: | :------: |
| LSTM-GAN          | 329   | [Drive](https://drive.google.com/file/d/1yJxYFjGnMfNz97c5OwbLm3h6-xyZiy-4/view?usp=sharing)  |
| Transformer-GAN   | 249   | [Drive](https://drive.google.com/file/d/1L2ED0_JRfttPX7DACwNDAgotCJ-buqPh/view?usp=sharing) | Requires LayoutGAN++
| Transformer-MSE   | 249   | [Drive](https://drive.google.com/file/d/1yMfsRCt-x127k8aUmtbuf_jCTj9y7DOW/view?usp=sharing)
| [LayoutGAN++](https://github.com/ktrk115/const_layout)       | 499   | [Drive](https://drive.google.com/file/d/1dZAJQXXosnLcFqMhVxB6IrDeDQVIaqZt/view?usp=sharing)


Transformer-GAN is adapted LayouGAN++[3] and uses pretrained frozen LayoutGAN++ that we provide above. 

## Dataset

Dataset is located in `./data/bbs/` in `.csv` format.

Was Generated from [DOC2PPT](https://doc2ppt.github.io/)[1] Dataset with [FitVid layout detection](https://github.com/imurs34/lecture_design_detection) (fine-tuned [CenterNet](https://github.com/xingyizhou/CenterNet)[2]) model.

The structure is as follows:

```
Slide Deck Id,Slide Id,Image Height,Image Width,Type,X,Y,BB Width,BB Height
```

## Results

### Quantitative Results

| Models            |  mIOU   | Accuracy (MSE) |  Overlap   |
| ----------------- |  ----:  | :-----------:  |  :------:  |
| LSTM-GAN          | 0.0304  | 0.0352         |  0.3579
| Transformer-GAN   | 0.0098  | 0.2422         |  1.4003
| Transformer-MSE   | 0.0798  | 0.0151         |  1.0448

Overlap in the actual dataset: `0.1700`.

### Qualitative Results

<div style="display: flex; flex-direction: row; gap: 1%">
    <img src="./evaluation/qualitative/tr-mse.png" alt="transformer-mse" title="Transformer-MSE Results" width="200">
    <img src="./evaluation/qualitative/tr-gan.png" alt="transformer-gan" title="Transformer-GAN Results" width="200">
    <img src="./evaluation/qualitative/lstm.png" alt="lstm" title="LSTM-GAN Results" width="200">
</div>


## References

[1] DOC2PPT: Automatic Presentation Slides Generation from Scientific Documents, Tsu-Jui Fu, William Yang Wang, Daniel McDuff, Yale Song, 2021

[2] CenterNet: Keypoint Triplets for Object Detection, Kaiwen Duan, Song Bai, Lingxi Xie, Honggang Qi, Qingming Huang, Qi Tian, 2019

[3] Constrained Graphic Layout Generation via Latent Optimization, Kotaro Kikuchi, Edgar Simo-Serra, Mayu Otani, Kota Yamaguchi, 2021


