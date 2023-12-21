from tqdm import tqdm
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from .mask import mask_graph_batch
from .position import add_noise_to_batch_pos, scale_batch_pos

class TrainerPreTrain:
    
    def __init__(self, model, path='./', parallel=False, world_size=None, device='cpu', weight_loss_pos=1):
        self.model = model
        self.device = device
        self.time_stamp = time.strftime("%Y%m%d-%H%M")
        self.path = Path(path).joinpath(
            f'version_w{weight_loss_pos:g}_{self.time_stamp}')
        self.path.mkdir(parents=True, exist_ok=True)
        self.configure_optimizer()
        self.reset_loss_history()

        # Setting up distributed training
        self.parallel = parallel
        if self.parallel:
            print("init DistributedDataParallel at device:", end=' ')
            self.world_size = world_size
            self.model = DistributedDataParallel(self.model, device_ids=[self.device])
            print(self.device)

        # Logging training info to tensorboard
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) and dist.get_rank() == 0:
            self.log_to_tensorboard = True
        elif not isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.log_to_tensorboard = True
        else:
            self.log_to_tensorboard = False

        if self.log_to_tensorboard:
            self.writer = SummaryWriter(self.path)
        
        # Loss functions
        self.criterion_atom = torch.nn.NLLLoss()
        self.criterion_pos = torch.nn.MSELoss()
        self.weight_loss_pos = weight_loss_pos

    def end_training(self, ):
        if self.log_to_tensorboard:
            self.writer.close()

    def reset_timer(self):
        self.start_time = time.time()
    
    def get_wall_time(self):
        return time.time() - self.start_time
    
    def reset_loss_history(self):
        self.train_loss_history = []
        self.train_loss_atom_history = []
        self.train_loss_pos_history = []
        self.val_loss_history = []
        self.val_loss_atom_history = []
        self.val_loss_pos_history = []

    def configure_optimizer(self,):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def gather_and_save_loss(self, loss, loss_history):
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            loss_tensor = torch.tensor(loss).to(self.device)
            gather_list = [torch.zeros_like(loss_tensor) for _ in range(self.world_size)]
            dist.all_gather(gather_list, loss_tensor)
            loss_history.append(np.concatenate([losses.cpu().numpy() for losses in gather_list]))
        else:
            loss_history.append(loss)

    def train(self, train_loader, val_loader=None, epochs=1):
        for epoch in range(epochs):
            self.current_epoch = epoch

            # Train for one epoch
            train_loss, train_loss_atom, train_loss_pos = self.train_epoch(train_loader)

            # Gather and save train loss
            self.gather_and_save_loss(train_loss, self.train_loss_history)
            self.gather_and_save_loss(train_loss_atom, self.train_loss_atom_history)
            self.gather_and_save_loss(train_loss_pos, self.train_loss_pos_history)
            print(f"Epoch {self.current_epoch} train loss on device {self.device}: {np.mean(self.train_loss_history[-1])}")
            if self.log_to_tensorboard:
                self.writer.add_scalar('loss/train', np.mean(self.train_loss_history[-1]), epoch)

            # Gather and save val loss
            if val_loader is not None:
                val_loss, val_loss_atom, val_loss_pos = self.val_epoch(val_loader)

                # Gather and save val loss
                self.gather_and_save_loss(val_loss, self.val_loss_history)
                self.gather_and_save_loss(val_loss_atom, self.val_loss_atom_history)
                self.gather_and_save_loss(val_loss_pos, self.val_loss_pos_history)
                print(f"Epoch {self.current_epoch} val loss on device {self.device}: {np.mean(self.val_loss_history[-1])}")
                if self.log_to_tensorboard:
                    self.writer.add_scalar('loss/val', np.mean(self.val_loss_history[-1]), epoch)

                # Save the best model
                if self.current_epoch > 0:
                    if (np.mean(self.val_loss_history[-1]) < np.array(self.val_loss_history[:-1]).mean(axis=1).min()):
                        self.save_to_checkpoint(self.path.joinpath(f'best.pt'))
                    if (np.mean(self.val_loss_atom_history[-1]) < np.array(self.val_loss_atom_history[:-1]).mean(axis=1).min()):
                        self.save_to_checkpoint(self.path.joinpath(f'best_atom.pt'))
                    if (np.mean(self.val_loss_pos_history[-1]) < np.array(self.val_loss_pos_history[:-1]).mean(axis=1).min()):
                        self.save_to_checkpoint(self.path.joinpath(f'best_pos.pt'))
            
            # Save the last model
            self.save_to_checkpoint(self.path.joinpath(f'last.pt'))

        # Close the writer when done with training to flush any remaining events to disk
        self.end_training()
    
    def train_epoch(self, train_loader):
        # Switch to train mode
        self.model.train()

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            # If parallel, only rank 0 will show progress bar
            if dist.get_rank() != 0:
                pbar = train_loader
            else:
                pbar = tqdm(train_loader)
        else:
            # If not parallel, show progress bar
            pbar = tqdm(train_loader)

        losses, losses_atom, losses_pos = [], [], []
        for batch in pbar:
            self.optimizer.zero_grad()
            batch = batch.to(self.device)

            # Target atomic number of speices
            target_atom = batch.atomic_numbers.clone()
            # Mask part of atoms
            mask = mask_graph_batch(batch)
            batch.atomic_numbers[mask] = 0.

            # Add noise to Cartesian positions
            batch, true_pos_noise = add_noise_to_batch_pos(batch, scaling=False)

            # Generate crystal structures
            #   Aim to predict the masked atoms and the noise added to scaled positions
            pred_atom, pred_pos_noise_scaled = self.model(batch)

            # Loss for atomic numbers of speices
            loss_atom = self.criterion_atom(torch.nn.LogSoftmax(dim=-1)(pred_atom.squeeze(1)), \
                                            target_atom.to(torch.int64) - 1) # (atomic number - 1) = class index
        
            # Scale true_pos_noise from Cartesian positions
            true_pos_noise_scaled = scale_batch_pos(batch, true_pos_noise)
            # Loss for atomic positions
            loss_pos = self.criterion_pos(pred_pos_noise_scaled, true_pos_noise_scaled)

            # Total loss
            loss = loss_atom + self.weight_loss_pos * loss_pos

            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            losses_atom.append(loss_atom.item())
            losses_pos.append(loss_pos.item())
            if isinstance(pbar, tqdm):
                pbar.set_description(f'Epoch {self.current_epoch} Training Rank {self.device} Loss: {loss.item():.4f}')

        return losses, losses_atom, losses_pos
    
    def val_epoch(self, val_loader):
        # Switch to val mode
        self.model.eval()

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            if dist.get_rank() != 0:
                pbar = val_loader
            else:
                pbar = tqdm(val_loader)
        else:
            pbar = tqdm(val_loader)

        losses, losses_atom, losses_pos = [], [], []
        with torch.no_grad():
            for batch in pbar:
                batch = batch.to(self.device)

                # Target atomic number of speices
                target_atom = batch.atomic_numbers.clone()
                # Mask part of atoms
                mask = mask_graph_batch(batch)
                batch.atomic_numbers[mask] = 0.

                # Add noise to Cartesian positions
                batch, true_pos_noise = add_noise_to_batch_pos(batch, scaling=False)

                # Generate crystal structures
                pred_atom, pred_pos_noise_scaled = self.model(batch)

                # Loss for atomic numbers of speices
                loss_atom = self.criterion_atom(torch.nn.LogSoftmax(dim=-1)(pred_atom.squeeze(1)), \
                                                target_atom.to(torch.int64) - 1) # (atomic number - 1) = class index
            
                # Scale true_pos_noise from Cartesian positions
                true_pos_noise_scaled = scale_batch_pos(batch, true_pos_noise)
                # Loss for atomic positions
                loss_pos = self.criterion_pos(pred_pos_noise_scaled, true_pos_noise_scaled)

                # Total loss
                loss = loss_atom + self.weight_loss_pos * loss_pos
                losses.append(loss.item())
                losses_atom.append(loss_atom.item())
                losses_pos.append(loss_pos.item())
                if isinstance(pbar, tqdm):
                    pbar.set_description(f'Epoch {self.current_epoch} Validation Rank {self.device} Loss: {loss.item():.4f}')
        return losses, losses_atom, losses_pos

    def save_to_checkpoint(self, path):
        # Check if the model is an instance of DistributedDataParallel
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            if dist.get_rank() != 0:
                return
            else:
                model_dict = self.model.module.state_dict()  # Access the original model
        else:
            model_dict = self.model.state_dict()

        output_dict = {
            'model': model_dict,
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'weight_loss_pos': self.weight_loss_pos,
            'loss_history': {
                'train': self.train_loss_history,
                'train_atom': self.train_loss_atom_history,
                'train_pos': self.train_loss_pos_history,
                'val': self.val_loss_history,
                'val_atom': self.val_loss_atom_history,
                'val_pos': self.val_loss_pos_history,
            }
        }
        torch.save(output_dict, path)

    def load_from_checkpoint(self, path):
        # Move model out of DDP wrapper if it's DDP wrapped
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module
        else:
            model = self.model

        if isinstance(self.device, int):
            map_location = f'cuda:{self.device}'
        else:
            map_location = self.device
        # Load the checkpoint
        loaded_dict = torch.load(path, map_location=map_location)
        
        # Load state into the model
        model.load_state_dict(loaded_dict['model'])

        # If we have DDP model, wrap it back into DDP
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.model = DistributedDataParallel(model.to(self.device), device_ids=[self.device])
            dist.barrier()
        else:
            self.model = model.to(self.device)
        
        # Load optimizer state
        self.optimizer.load_state_dict(loaded_dict['optimizer'])
        
        # Load loss history
        self.weight_loss_pos = loaded_dict['weight_loss_pos']
        self.train_loss_history = loaded_dict['loss_history']['train']
        self.train_loss_atom_history = loaded_dict['loss_history']['train_atom']
        self.train_loss_pos_history = loaded_dict['loss_history']['train_pos']
        self.val_loss_history = loaded_dict['loss_history']['val']
        self.val_loss_atom_history = loaded_dict['loss_history']['val_atom']
        self.val_loss_pos_history = loaded_dict['loss_history']['val_pos']
        self.current_epoch = loaded_dict['epoch']