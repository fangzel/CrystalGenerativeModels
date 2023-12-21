import torch
from torch.nn.parallel import DistributedDataParallel
import numpy as np
from tqdm import tqdm
from pathlib import Path
import time
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

class TrainerFineTuneRegression:
    
    def __init__(self, model, path='./', parallel=False, world_size=None, device='cpu', output_attr=''):
        self.model = model
        self.device = device
        # output attribute
        if output_attr == '':
            raise ValueError("AttributeError for output")
        else:
            self.output_attr = output_attr
        
        self.time_stamp = time.strftime("%Y%m%d-%H%M")
        self.path = Path(path).joinpath(
            f'finetune_{self.output_attr}_{self.time_stamp}')
        self.path.mkdir(parents=True, exist_ok=True)
        self.configure_optimizer()
        self.reset_loss_history()

        # setting up distributed training
        self.parallel = parallel
        if self.parallel:
            print("init DistributedDataParallel at device:", end=' ')
            self.world_size = world_size
            self.model = DistributedDataParallel(self.model, device_ids=[self.device])
            print(self.device)

        # logging training info to tensorboard
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) and dist.get_rank() == 0:
            self.log_to_tensorboard = True
        elif not isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.log_to_tensorboard = True
        else:
            self.log_to_tensorboard = False

        if self.log_to_tensorboard:
            self.writer = SummaryWriter(self.path)
        
        # Loss function
        self.criterion = torch.nn.MSELoss()

    def end_training(self, ):
        if self.log_to_tensorboard:
            self.writer.close()

    def reset_timer(self):
        self.start_time = time.time()
    
    def get_wall_time(self):
        return time.time() - self.start_time
    
    def reset_loss_history(self):
        self.train_loss_history = []
        self.val_loss_history = []

    def configure_optimizer(self,):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def gather_and_save_loss(self, loss, loss_history):
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            # Convert list to tensor and send to device
            loss_tensor = torch.tensor(loss).to(self.device)

            # Create a list to hold the tensors that will be gathered from all processes
            gather_list = [torch.zeros_like(loss_tensor) for _ in range(self.world_size)]
                
            # Gather loss history from all processes
            dist.all_gather(gather_list, loss_tensor)

            # Convert tensors back to lists, flatten the list, and append to loss_history
            loss_history.append(np.concatenate([losses.cpu().numpy() for losses in gather_list]))
        else:
            loss_history.append(loss)

    def train(self, train_loader, val_loader=None, epochs=1):
        for epoch in range(epochs):
            self.current_epoch = epoch

            # train for one epoch
            train_loss = self.train_epoch(train_loader)

            # gather and save train loss
            self.gather_and_save_loss(train_loss, self.train_loss_history)
            print(f"Epoch {self.current_epoch} train loss on device {self.device}: {np.mean(self.train_loss_history[-1])}")
            if self.log_to_tensorboard:
                self.writer.add_scalar('loss/train', np.mean(self.train_loss_history[-1]), epoch)

            # save last model
            self.save_to_checkpoint(self.path.joinpath(f'last.pt'))

            if val_loader is not None:
                val_loss = self.val_epoch(val_loader)

                # gather and save val loss
                self.gather_and_save_loss(val_loss, self.val_loss_history)
                print(f"Epoch {self.current_epoch} val loss on device {self.device}: {np.mean(self.val_loss_history[-1])}")
                if self.log_to_tensorboard:
                    self.writer.add_scalar('loss/val', np.mean(self.val_loss_history[-1]), epoch)

                # save best model
                if self.current_epoch > 0:
                    if (np.mean(self.val_loss_history[-1]) < np.array(self.val_loss_history[:-1]).mean(axis=1).min()):
                        self.save_to_checkpoint(self.path.joinpath(f'best.pt'))

        # save last model
        self.save_to_checkpoint(self.path.joinpath(f'last.pt'))
        # Close the writer when done with training to flush any remaining events to disk
        self.end_training()
        
    def train_epoch(self, train_loader):
        # switch to train mode
        # self.model.to(self.device)
        self.model.train()

        losses = []

        # if parallel, only rank 0 will show progress bar
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            if dist.get_rank() != 0:
                pbar = train_loader
            else:
                pbar = tqdm(train_loader)
        # if not parallel, show progress bar
        else:
            pbar = tqdm(train_loader)

        for batch in pbar:
            self.optimizer.zero_grad()
            batch = batch.to(self.device)

            expectation = getattr(batch, self.output_attr)
            prediction = self.model(batch)

            # loss
            loss = self.criterion(prediction.squeeze(), expectation)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            if isinstance(pbar, tqdm):
                pbar.set_description(f'Epoch {self.current_epoch} Training Rank {self.device} Loss: {loss.item():.4f}')

        return losses
    
    def val_epoch(self, val_loader):
        # self.model.to(self.device)
        self.model.eval()
        losses = []

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            if dist.get_rank() != 0:
                pbar = val_loader
            else:
                pbar = tqdm(val_loader)
        else:
            pbar = tqdm(val_loader)

        with torch.no_grad():
            for batch in pbar:
                batch = batch.to(self.device)

                expectation = getattr(batch, self.output_attr)
                prediction = self.model(batch)

                # loss
                loss = self.criterion(prediction.squeeze(), expectation)
                losses.append(loss.item())

                if isinstance(pbar, tqdm):
                    pbar.set_description(f'Epoch {self.current_epoch} Validation Rank {self.device} Loss: {loss.item():.4f}')
        return losses

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
            'loss_history': {
                'train': self.train_loss_history,
                'val': self.val_loss_history,
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
        self.train_loss_history = loaded_dict['loss_history']['train']
        self.val_loss_history = loaded_dict['loss_history']['val']
        self.current_epoch = loaded_dict['epoch']