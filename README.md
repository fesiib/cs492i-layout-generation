# Structure-aware Layout Generation

### Team 9: Bekzat Tilekbay, Shyngys Aitkazinov

## Installation

1. Clone this repository

    ```bash
    git clone https://github.com/fesiib/cs492i-layout-generation.git
    cd cs492i-layout-generation
    ```

2. Create a new [conda](https://docs.conda.io/en/latest/miniconda.html) environment (Python 3.8)

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

## Results

### Comparisons



### Qualitative Results

![Transformer-MSE Results](./evaluation/qualitative/tr-mse.png "Transformer-MSE Results")
![Transformer-GAN Results](./evaluation/qualitative/tr-gan.png "Transformer-GAN Results")
![LSTM-GAN Results](./evaluation/qualitative/lstm.png "LSTM-GAN Results")

## Models
| Models            | Epochs|      Link     | Comments |
| ----------------- | ----: | :-----------: | :------: |
| LSTM-GAN          | 329   | [Drive](https://drive.google.com/file/d/1yJxYFjGnMfNz97c5OwbLm3h6-xyZiy-4/view?usp=sharing)  |
| Transformer-GAN   | 249   | [Drive](https://drive.google.com/file/d/1L2ED0_JRfttPX7DACwNDAgotCJ-buqPh/view?usp=sharing) | Requires LayoutGAN++
| Transformer-MSE   | 249   | [Drive](https://drive.google.com/file/d/1yMfsRCt-x127k8aUmtbuf_jCTj9y7DOW/view?usp=sharing)
| LayoutGAN++       | 499   | [Drive](https://drive.google.com/file/d/1dZAJQXXosnLcFqMhVxB6IrDeDQVIaqZt/view?usp=sharing)

## References
