import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import os, glob
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import socket

from .external.ase_dataset_attribute import AseReadDatasetAttr
from .trainer_finetune_regression import TrainerFineTuneRegression

import sys
sys.path.append('/crystal-transformer')
from equiformer_v2.nets import EquiformerV2


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))            # Bind to a port that is free
        return s.getsockname()[1]  # Return the port number

def setup(rank, world_size):
    port = find_free_port()
    url = f'tcp://localhost:{port}'

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method=url)

def cleanup():
    dist.destroy_process_group()

def worker(rank, world_size, 
           data_path, attr_path, 
           output_attr, attr_func, output_block_channels,
           checkpoint_path, modelsave_path, n_epochs):

    setup(rank, world_size)

    # Attribute functions
    functions = {
        'None': None,
        'log': torch.log,
        # Add other functions here if needed
    }

    # Prepare training and validation data loaders
    cif_files = [_.split('/')[-1] for _ in glob.glob(os.path.join(data_path, '*.cif'))]
    train_files, val_test_files = train_test_split(cif_files, test_size=0.2, random_state=42)
    val_files, _ = train_test_split(val_test_files, test_size=0.5, random_state=42)

    train_config = {
            'src': data_path,
            'pattern': '.cif',
            'files': train_files,
            'attr_name': output_attr,
            'attr_pattern': '.pickle',
            'attr_pkl_dir': attr_path,
            'attr_func': functions[attr_func],
        }
    val_config = {
            'src': data_path,
            'pattern': '.cif',
            'files': val_files,
            'attr_name': output_attr,
            'attr_pattern': '.pickle',
            'attr_pkl_dir': attr_path,
            'attr_func': functions[attr_func],
        }

    train_dataset = AseReadDatasetAttr(train_config)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler)
    
    val_dataset = AseReadDatasetAttr(val_config)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank)
    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=val_sampler)

    # Prepare model and trainer
    model = EquiformerV2(output_block_channels=output_block_channels, output_aggr_mode='mean', regress_forces=False)
    trainer = TrainerFineTuneRegression(
        model.to(rank), 
        path=modelsave_path,
        parallel=True,
        world_size=world_size,
        device=rank,
        output_attr=output_attr)
    
    # Load checkpoint
    trainer.load_from_checkpoint(checkpoint_path)

    print("Load dataset from ", data_path)
    print("Load attributes "+ output_attr + " from ", attr_path)
    print("Load checkpoint from ", checkpoint_path)
    print("Save model to ", trainer.path)

    # Start training
    trainer.train(train_loader, val_loader=val_loader, epochs=n_epochs)

    cleanup()

def main(world_size, 
        data_path, attr_path, 
        output_attr, attr_func, output_block_channels,
        checkpoint_path, modelsave_path, n_epochs):

    world_size = torch.cuda.device_count()
    mp.spawn(worker, args=(world_size, data_path, attr_path, output_attr, attr_func, output_block_channels, checkpoint_path, modelsave_path, n_epochs), nprocs=world_size, join=True)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to load inputs.")
    parser.add_argument("--attr_path", type=str, required=True, help="Path to load outputs.")
    parser.add_argument("--output_attr", type=str, required=True, help="The quantity to be predicted.")
    parser.add_argument("--attr_func", type=str, default='None', help="A function applied to the output quantity.")
    parser.add_argument("--output_block_channels", type=int, required=True, help="The dimension of outputs.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path for loading model checkpoint.")
    parser.add_argument("--modelsave_path", type=str, required=True, help="Path for saving model.")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs.")
    args = parser.parse_args()
    main(args.data_path , args.attr_path, args.output_attr, args.attr_func, args.output_block_channels, args.checkpoint_path, args.modelsave_path, args.n_epochs)
