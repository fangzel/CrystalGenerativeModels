import torch
import time
import argparse
import os, glob
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from src import TrainerGANLit
from src import AseReadDatasetAttr
from src import AseReadDataset
from equiformer_v2.nets import EquiformerV2

def main(data_path, discriminator_label_path, checkpoint_path, modelsave_path, n_epochs):
    
    world_size = torch.cuda.device_count()
    version_save_path = modelsave_path + '/version_'+time.strftime("%Y%m%d-%H")
    train_batch_size = world_size
    val_batch_size = world_size

    ############### Trainer ###############
    print("Checkpoints saving in "+version_save_path)
    # # save top-K checkpoints based on discriminator loss
    cp_callback_best_d = ModelCheckpoint(
        save_top_k=3,
        monitor="acc_d",
        mode="max",
        dirpath=version_save_path,
        filename='best-d-e{epoch:02d}-acc{acc_d:.3f}',
        every_n_epochs=1,
        auto_insert_metric_name=False,
    )
    # # save top-K checkpoints based on generator loss
    cp_callback_best_g = ModelCheckpoint(
        save_top_k=3,
        monitor="loss_g",
        mode="min",
        dirpath=version_save_path,
        filename='best-g-e{epoch:02d}-loss{loss_g:.3f}',
        every_n_epochs=1,
        auto_insert_metric_name=False,
    )
    # # save last
    cp_callback_last = ModelCheckpoint(
        save_last =True,
        dirpath=version_save_path,
        filename='last-e{epoch:02d}',
        auto_insert_metric_name=False,
    )
    # # save every 10 epoch
    cp_callback_epoch = ModelCheckpoint(
        every_n_epochs =10,
        dirpath=version_save_path,
        filename='epoch-{epoch:02d}',
        auto_insert_metric_name=False,
    )
    
    trainer = L.Trainer(num_sanity_val_steps=-1, # Use the entire validation dataset for the sanity check
                        accelerator="gpu", devices = world_size, max_epochs= n_epochs, 
                        strategy='ddp_find_unused_parameters_true', # other strategy e.g., "fsdp"
                        precision="bf16-mixed", # 16-bit precision to reduce memory usage
                        callbacks=[cp_callback_best_d, cp_callback_best_g, cp_callback_last, cp_callback_epoch])

    ############### GAN ###############
    generator = EquiformerV2(output_block_channels=118, output_aggr_mode='node', regress_forces=True)
    discriminator = EquiformerV2(output_block_channels=2, output_aggr_mode='mean', regress_forces=False)
    gan = TrainerGANLit(generator, discriminator)
    # gan.load_state_dict(torch.load(checkpoint_path))
    gan = TrainerGANLit.load_from_checkpoint(checkpoint_path, generator=generator, discriminator=discriminator)

    ############### Dataset ###############
    cif_files = [_.split('/')[-1] for _ in glob.glob(os.path.join(data_path, '*.cif'))]
    train_files, val_test_files = train_test_split(cif_files, test_size=0.2, random_state=42)
    val_files, _ = train_test_split(val_test_files, test_size=0.5, random_state=42)

    # If the original data includes unstable crystal structures (expect to be discriminated as 'fake'), 
    #    the information of stability is saved in 'discriminator_label_path'.
    # Otherwise, there is no need to load attribute 'is_stable'
    if discriminator_label_path:
        train_config = {
                'src': data_path,
                'pattern': '.cif',
                'files': train_files,
                'attr_name': 'is_stable',
                'attr_pattern': '.pickle',
                'attr_pkl_dir': discriminator_label_path,
                'attr_func': int,
            }
        val_config = {
                'src': data_path,
                'pattern': '.cif',
                'files': val_files,
                'attr_name': 'is_stable',
                'attr_pattern': '.pickle',
                'attr_pkl_dir': discriminator_label_path,
                'attr_func': int,
            }
        train_dataset = AseReadDatasetAttr(train_config)
        val_dataset = AseReadDatasetAttr(val_config)
    else:
        train_config = {'src': data_path, 'pattern': '*.cif', 'files': train_files}
        val_config = {'src': data_path, 'pattern': '*.cif', 'files': val_files}
        train_dataset = AseReadDataset(train_config)
        val_dataset = AseReadDataset(val_config)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,  num_replicas=trainer.world_size, rank=trainer.global_rank)
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        sampler=train_sampler)

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=trainer.world_size, rank=trainer.global_rank)
    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        sampler=val_sampler)
    
    ############### Training ###############
    trainer.fit(model=gan, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to load training and validation set.")
    parser.add_argument("--discriminator_label_path", type=str, default=None, help="Path to load stability of each original crystal structure.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint for loading GAN.")
    parser.add_argument("--modelsave_path", type=str, required=True, help="Path to save GAN.")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs.")
    args = parser.parse_args()
    main(args.data_path, args.discriminator_label_path, args.checkpoint_path, args.modelsave_path, args.n_epochs)
