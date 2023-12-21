# CrystalGenerativeModels


## Getting Started
1. Environment Setup: see [requirements.txt](requirements.txt).

2. Clone this repo.

3. Clone our modified version of 'equiformer_v2' and put it into our root folder.
   ```sh
   cd CrystalGenerativeModels
   ```
   ```sh
   git clone https://github.com/zhantaochen/equiformer_v2.git
   ```
4. Install `ocp-git`
   ```sh
   git clone https://github.com/Open-Catalyst-Project/ocp/tree/main
   ```
   ```sh
   cd ocp
   ```
   ```sh
   pip install -e .
   ```

## Code description:

In [src/](src):
* [main_pretrain.py](src/main_pretrain.py) and [trainer_pretrain.py](src/trainer_pretrain.py): code and trainer for self-supervised pre-training
* [main_finetune_regression.py](src/main_finetune_regression.py) and [trainer_finetune_regression.py](src/trainer_finetune_regression.py): code and trainer for fine-tuning on downstream regression tasks
* [main_finetune_classification.py](src/main_finetune_classification.py) and [trainer_finetune_classification.py](src/trainer_finetune_classification.py): code and trainer for fine-tuning on downstream classification tasks
* [main_gan_lit.py](src/main_gan_lit.py) and [trainer_gan_lit.py](src/trainer_gan_lit.py): code and trainer for self-supervised generative adversarial networks
* [mask.py](src/mask.py): ultility function for masking atoms
* [position.py](src/position.py): ultility functions related to operations on atomic positions

In [src/external](src/external):
* [ase_dataset.py](src/external/ase_dataset.py) and [ase_dataset_attribute.py](src/external/ase_dataset_attribute.py): data loader for ASE Atoms object


## Pre-Training:

<p align="center">
	<img src="plots/pretrain_procedure.png" alt="photo not available" width="98%" height="98%">
</p>

Pre-training parameters:

* `--weight_loss_pos`: weight of the loss related to positions (while the weigt of atomic number is set to 1).
* `--data_path`: path to load dataset, with each sample saved in seperate CIF files.
* `--checkpoint_path`: path to checkpoint for loading model.
* `--modelsave_path`: path for saving the trained model.
* `--n_epochs`: number of training epochs.

Script example for running [main_pretrain.py](src/main_pretrain.py):
```sh
python src/main_pretrain.py --weight_loss_pos <float> --data_path <str> --checkpoint_path <str> --modelsave_path <str> --n_epochs <int>
```

## Fine-Tuning:

<p align="center">
	<img src="plots/finetune_procedure.png" alt="photo not available" width="98%" height="98%">
</p>

Fine-Tuning parameters:

* `--data_path`: path to load dataset of input crystal structures, with each sample saved in seperate CIF files.
* `--attr_path`: path to load labels (outputs) corresponding to each input crystal structure, with each sample saved in seperate Pickle files.
* `--output_attr`: a string named the target quantity for prediction.
* `--attr_func`: a string named the function applied to the output quantity, e.g., 'log'.
* `--output_block_channels`: the dimension of the output.
* `--checkpoint_path`: path to checkpoint for loading model.
* `--modelsave_path`: path for saving the trained model.
* `--n_epochs`: number of training epochs.

Script example for running [main_finetune_regression.py](src/main_finetune_regression.py) and [main_finetune_classification.py](src/main_finetune_classification.py):
```sh
python src/main_finetune_regression.py --data_path <str> --attr_path <str> --output_attr <str> --attr_func <str> --output_block_channels <int> --checkpoint_path <str> --modelsave_path <str> --n_epochs <int>
```
```sh
python src/main_finetune_classification.py --data_path <str> --attr_path <str> --output_attr <str> --attr_func <str> --output_block_channels <int> --checkpoint_path <str> --modelsave_path <str> --n_epochs <int>
```

## GAN

<p align="center">
	<img src="plots/gan_procedure.png" alt="photo not available" width="70%" height="70%">
</p>

GAN parameters:

* `--data_path`: path to load dataset of input crystal structures, with each sample saved in seperate CIF files.
* `--discriminator_label_path`: path to load stability of each original crystal structure (which is not necessary if all input structures are stable.)
* `--checkpoint_path`: path to checkpoint for loading model.
* `--modelsave_path`: path for saving the trained model.
* `--n_epochs`: number of training epochs.

Script example for running [main_gan_lit.py](src/main_gan_lit.py):
```sh
python src/main_gan_lit.py --data_path <str> --discriminator_label_path <str> --checkpoint_path <str> --modelsave_path <str> --n_epochs <int>
```











