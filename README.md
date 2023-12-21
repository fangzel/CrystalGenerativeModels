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


## Fine-Tuning:

<p align="center">
	<img src="plots/finetune_procedure.png" alt="photo not available" width="98%" height="98%">
</p>

## GAN

<p align="center">
	<img src="plots/gan_procedure.png" alt="photo not available" width="70%" height="70%">
</p>













