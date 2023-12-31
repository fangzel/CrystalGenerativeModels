import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import os, glob
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import socket

from src import AseReadDataset
from src import TrainerPreTrain
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

def worker(rank, world_size, weight_loss_pos, data_path, checkpoint_path, modelsave_path, n_epochs):
    setup(rank, world_size)

    # Prepare training and validation data loaders
    cif_files = [_.split('/')[-1] for _ in glob.glob(os.path.join(data_path, '*.cif'))]
    train_files, val_test_files = train_test_split(cif_files, test_size=0.2, random_state=42)
    val_files, _ = train_test_split(val_test_files, test_size=0.5, random_state=42)

    train_config = {'src': data_path, 'pattern': '*.cif', 'files': train_files}
    val_config = {'src': data_path, 'pattern': '*.cif', 'files': val_files}

    train_dataset = AseReadDataset(train_config)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=5,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler)

    val_dataset = AseReadDataset(val_config)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=val_sampler)

    # Prepare model and trainer
    model = EquiformerV2(output_block_channels=118, output_aggr_mode='node', regress_forces=True)
    trainer = TrainerPreTrain(
        model.to(rank), 
        path=modelsave_path,
        parallel=True,
        world_size=world_size,
        device=rank,
        weight_loss_pos=weight_loss_pos)
    
    # Load checkpoint
    if checkpoint_path:
        trainer.load_from_checkpoint(checkpoint_path)
    
    print("Weight of position loss is ", trainer.weight_loss_pos)
    print("Load dataset from ", data_path)
    print("Load checkpoint from ", checkpoint_path)
    print("Save model to ", trainer.path)

    # Start training
    trainer.train(train_loader, val_loader=val_loader, epochs=n_epochs)

    cleanup()

def main(weight_loss_pos, data_path, checkpoint_path, modelsave_path, n_epochs):

    world_size = torch.cuda.device_count()
    mp.spawn(worker, args=(world_size, weight_loss_pos, data_path, checkpoint_path, modelsave_path, n_epochs), nprocs=world_size, join=True)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_loss_pos", type=float, required=True, help="Weight of loss_pos (while weight of loss_atom is set to 1).")
    parser.add_argument("--data_path", type=str, default=None, help="Path to load training and validation set.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint for loading model.")
    parser.add_argument("--modelsave_path", type=str, required=True, help="Path for saving model.")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs.")
    args = parser.parse_args()
    main(args.weight_loss_pos, args.data_path, args.checkpoint_path, args.modelsave_path, args.n_epochs)
