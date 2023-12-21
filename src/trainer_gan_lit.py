import torch
from torch import optim
import lightning as L
import sys
from .mask import mask_graph_batch
from .position import add_noise_to_batch_pos, scale_batch_pos, unscale_batch_pos, wrap_batch_pos

sys.path.append('/crystal-transformer')

class TrainerGANLit(L.LightningModule):
    
    def __init__(self, generator, discriminator, weight_loss_pos=200):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        
        # activates manual optimizer
        self.automatic_optimization = False
        
        # Loss function
        self.weight_loss_pos = weight_loss_pos
        self.criterion_generator_atom = torch.nn.NLLLoss()
        self.criterion_generator_pos = torch.nn.MSELoss()
        self.criterion_discriminator = torch.nn.NLLLoss()
        self.weight_loss_label = 0.1

        # Number of steps to train discriminator
        self.n_steps_discriminator = 10
        # Deterimine the step to start training discriminator
        self.starting_step_discriminator = 0

        # record
        self.train_loss_history = {'d_real': [], 'd_fake': [], 'g_label': [], 'g_atom': [], 'g_pos': []}
        self.val_loss_history = {'d_real': [], 'd_fake': [], 'g_label': [], 'g_atom': [], 'g_pos': []}
        self.val_metric_history = {'acc_d': [], 'loss_g':[], 'acc_d_real': [], 'acc_d_fake': [], 'acc_g_label': []}
        self.reset_train_loss_each_epoch()
        self.reset_val_loss_each_epoch()
    
    def wasserstein_distance(self, probabilities, labels):
        """
        Args:
        probabilities (torch.Tensor): Probabilities from the model, shape (n_sample, K)
        labels (torch.Tensor): Ground truth labels, shape (n_sample,)
        
        Returns:
         (torch.Tensor): Calculated Wasserstein distance
        """
        one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=probabilities.size(1))
        return torch.mean(torch.sum(torch.abs(probabilities - one_hot_labels.float()), dim=1))

    def reset_train_loss_each_epoch(self):
        self.train_loss_each_epoch = {'d_real': [], 'd_fake': [], 'g_label': [], 'g_atom': [], 'g_pos': []}

    def reset_val_loss_each_epoch(self):
        self.val_loss_each_epoch = {'d_real': [], 'd_fake': [], 'g_label': [], 'g_atom': [], 'g_pos': []}
        self.val_acc_each_epoch = {'d_real': [], 'd_fake': [], 'g_label': []}

    def configure_optimizers(self):
        optimizer_generator = optim.Adam(self.generator.parameters(), lr=1e-4)
        optimizer_discriminator = optim.Adam(self.discriminator.parameters(), lr=1e-5, betas=(0.5, 0.999))
        return optimizer_generator, optimizer_discriminator
    
    def training_step(self, batch, batch_idx):
        optimizer_generator, optimizer_discriminator = self.optimizers()

        # true & fake sturucture
        real_batch = batch.clone().detach()
        fake_batch = real_batch.clone().detach()
        if hasattr(real_batch, 'is_stable'):
            real_labels = real_batch.is_stable
            delattr(real_batch, 'is_stable')
        else:
            real_labels = torch.ones(len(batch.natoms))
        fake_labels = torch.zeros_like(real_labels)
        right_labels = torch.ones_like(real_labels)
        
        ##### Contaminated Input Structure #####
        # Mask atominc numbers
        mask = mask_graph_batch(fake_batch)
        fake_batch.atomic_numbers[mask] = 0.
        # Add noise to Cartesian positions
        fake_batch, target_pos_noise = add_noise_to_batch_pos(fake_batch, sigma_ratio=0.2, scaling=False)
        target_pos_noise_scaled = scale_batch_pos(fake_batch, target_pos_noise).detach()

        #################### Train Generator ####################
        # Loss based on discriminator
        # self.generator.zero_grad()
        self.toggle_optimizer(optimizer_generator) # Makes sure only the gradients of the current optimizer’s parameters are calculated in the training step

        # Generate fake structure
        fake_atom, fake_pos_noise_scaled = self.generator(fake_batch.detach())
        fake_batch.atomic_numbers = torch.argmax(fake_atom.squeeze(1).log_softmax(dim=-1).detach(), dim=-1) + 1
        fake_pos_noise = unscale_batch_pos(fake_batch, fake_pos_noise_scaled)
        fake_batch.pos = wrap_batch_pos(fake_batch, fake_batch.pos - fake_pos_noise)

        # Generate labels for fake structure by discriminator
        with torch.no_grad():
            output_d_label = self.discriminator(fake_batch)
        loss_g_label = self.wasserstein_distance(output_d_label.squeeze(1).softmax(dim=-1), right_labels) # loss_g_label = self.criterion_discriminator(output_d_label.squeeze(1).log_softmax(dim=-1), right_labels) 

        # Loss based on generative model
        loss_g_atom = self.criterion_generator_atom(fake_atom.squeeze(1).log_softmax(dim=-1), real_batch.atomic_numbers.to(torch.int64) - 1) # (atomic number - 1) = class index
        loss_g_pos = self.criterion_generator_pos(fake_pos_noise_scaled, target_pos_noise_scaled)

        # Update
        loss_g_total = self.weight_loss_label * loss_g_label + loss_g_atom + self.weight_loss_pos * loss_g_pos
        self.manual_backward(loss_g_total)
        # self.clip_gradients(optimizer_generator, gradient_clip_val=1, gradient_clip_algorithm="norm")
        optimizer_generator.step()
        # Reset
        optimizer_generator.zero_grad()
        self.untoggle_optimizer(optimizer_generator) 

        ##### Record #####
        self.train_loss_each_epoch['g_label'].append(loss_g_label.item())
        self.train_loss_each_epoch['g_atom'].append(loss_g_atom.item())
        self.train_loss_each_epoch['g_pos'].append(loss_g_pos.item())

        if (batch_idx + self.starting_step_discriminator) % self.n_steps_discriminator == 0:
            #################### Train Discriminator ####################
            # self.discriminator.zero_grad()
            # Add noise to real_batch
            real_batch, _ = add_noise_to_batch_pos(real_batch, sigma_ratio=0.001, distribution ='uniform', scaling=False)

            self.toggle_optimizer(optimizer_discriminator) # Makes sure only the gradients of the current optimizer’s parameters are calculated in the training step
            ##### Train on Real Structure #####
            output_d_real = self.discriminator(real_batch.detach())
            loss_d_real = self.wasserstein_distance(output_d_real.squeeze(1).softmax(dim=-1), real_labels) # loss_d_real = self.criterion_discriminator(output_d_real.squeeze(1).log_softmax(dim=-1), real_labels) # 

            ##### Train on Fake Structure #####
            output_d_fake = self.discriminator(fake_batch.detach())
            loss_d_fake = self.wasserstein_distance(output_d_fake.squeeze(1).softmax(dim=-1), fake_labels) # loss_d_fake = self.criterion_discriminator(output_d_fake.squeeze(1).log_softmax(dim=-1), fake_labels) # 

            # Backward
            loss_d_total = loss_d_real+loss_d_fake
            self.manual_backward(loss_d_total)
            # self.clip_gradients(optimizer_discriminator, gradient_clip_val=1, gradient_clip_algorithm="norm")
            optimizer_discriminator.step()
            # Reset
            optimizer_discriminator.zero_grad()
            self.untoggle_optimizer(optimizer_discriminator)

            ##### Record #####
            self.train_loss_each_epoch['d_real'].append(loss_d_real.item())
            self.train_loss_each_epoch['d_fake'].append(loss_d_fake.item())

    def validation_step(self, batch, batch_idx):
        # true & fake sturucture
        real_batch = batch.clone().detach()
        fake_batch = real_batch.clone().detach()
        if hasattr(real_batch, 'is_stable'):
            real_labels = real_batch.is_stable
            delattr(real_batch, 'is_stable')
        else:
            real_labels = torch.ones(len(batch.natoms))
        fake_labels = torch.zeros_like(real_labels)
        right_labels = torch.ones_like(real_labels)

        ##### Contaminated Input Structure #####
        # Mask atominc numbers
        mask = mask_graph_batch(fake_batch)
        fake_batch.atomic_numbers[mask] = 0.
        # Add noise to Cartesian positions
        fake_batch, target_pos_noise = add_noise_to_batch_pos(fake_batch, sigma_ratio=0.2, scaling=False)
        target_pos_noise_scaled = scale_batch_pos(fake_batch, target_pos_noise).detach()


        #################### Evaluate Generator ####################
        # Generate fake structure
        fake_atom, fake_pos_noise_scaled = self.generator(fake_batch.detach())
        fake_batch.atomic_numbers = torch.argmax(fake_atom.squeeze(1).log_softmax(dim=-1).detach(), dim=-1) + 1
        fake_pos_noise = unscale_batch_pos(fake_batch, fake_pos_noise_scaled)
        fake_batch.pos = wrap_batch_pos(fake_batch, fake_batch.pos - fake_pos_noise)

        # Generate labels for fake structure by discriminator
        output_d_label = self.discriminator(fake_batch)
        loss_g_label = self.wasserstein_distance(output_d_label.squeeze(1).softmax(dim=-1), right_labels) # loss_g_label = self.criterion_discriminator(output_d_label.squeeze(1).log_softmax(dim=-1), right_labels) # 

        # Loss based on generative model
        loss_g_atom = self.criterion_generator_atom(fake_atom.squeeze(1).log_softmax(dim=-1), real_batch.atomic_numbers.to(torch.int64) - 1) # (atomic number - 1) = class index
        loss_g_pos = self.criterion_generator_pos(fake_pos_noise_scaled, target_pos_noise_scaled)

        # Accuracies
        predicted_labels = torch.argmax(output_d_label.squeeze(1).log_softmax(dim=-1), dim=1)
        acc_g_label = (predicted_labels == right_labels).sum().item() /right_labels.size(0)


        #################### Evaluate Discriminator ####################
        ##### Evaluate on Real Structure #####
        output_d_real = self.discriminator(real_batch)
        loss_d_real = self.wasserstein_distance(output_d_real.squeeze(1).softmax(dim=-1), real_labels) # loss_d_real = self.criterion_discriminator(output_d_real.squeeze(1).log_softmax(dim=-1), real_labels) # 

        # Accuracies
        predicted_labels = torch.argmax(output_d_real.squeeze(1).log_softmax(dim=-1), dim=1)
        acc_d_real = (predicted_labels == real_labels).sum().item() / real_labels.size(0)

        ##### Evaluate on Fake Structure #####
        output_d_fake = self.discriminator(fake_batch.detach())
        loss_d_fake = self.wasserstein_distance(output_d_fake.squeeze(1).softmax(dim=-1), fake_labels)  # loss_d_fake = self.criterion_discriminator(output_d_fake.squeeze(1).log_softmax(dim=-1), fake_labels) # 
        # Accuracies
        predicted_labels = torch.argmax(output_d_fake.squeeze(1).log_softmax(dim=-1), dim=1)
        acc_d_fake = (predicted_labels == fake_labels).sum().item() / fake_labels.size(0)

        #################### Record ####################
        # Record loss
        self.val_loss_each_epoch['d_real'].append(loss_d_real.item())
        self.val_loss_each_epoch['d_fake'].append(loss_d_fake.item())
        self.val_loss_each_epoch['g_label'].append(loss_g_label.item())
        self.val_loss_each_epoch['g_atom'].append(loss_g_atom.item())
        self.val_loss_each_epoch['g_pos'].append(loss_g_pos.item())
        # Record accuracy
        self.val_acc_each_epoch['d_real'].append(acc_d_real)
        self.val_acc_each_epoch['d_fake'].append(acc_d_fake)
        self.val_acc_each_epoch['g_label'].append(acc_g_label)

    def on_train_epoch_end(self):
        for loss_key in ['d_real', 'd_fake', 'g_label', 'g_atom', 'g_pos']:
            if len(self.train_loss_each_epoch[loss_key]) > 0:
                self.train_loss_history[loss_key].append(sum(self.train_loss_each_epoch[loss_key])/len(self.train_loss_each_epoch[loss_key]))
        self.reset_train_loss_each_epoch()

        # Shift the batches that will be trained by discriminator
        self.starting_step_discriminator = (self.starting_step_discriminator + 1) % self.n_steps_discriminator

 
    def on_validation_epoch_end(self):
        for loss_key in ['d_real', 'd_fake', 'g_label', 'g_atom', 'g_pos']:
            self.val_loss_history[loss_key].append(sum(self.val_loss_each_epoch[loss_key])/len(self.val_loss_each_epoch[loss_key]))

        for loss_key in ['d_real', 'd_fake', 'g_label']:
            self.val_metric_history['acc_'+loss_key].append(sum(self.val_acc_each_epoch[loss_key])/len(self.val_acc_each_epoch[loss_key]))
        self.val_metric_history['acc_d'].append((sum(self.val_acc_each_epoch['d_real']) + sum(self.val_acc_each_epoch['d_fake']))/(2*len(self.val_acc_each_epoch['d_real'])))
        self.val_metric_history['loss_g'].append((sum(self.val_loss_each_epoch['g_label']) + sum(self.val_loss_each_epoch['g_atom'])+sum(self.val_loss_each_epoch['g_pos'])) / len(self.val_loss_each_epoch['g_label']))

        self.log_dict({"loss_g": self.val_metric_history['loss_g'][-1], "acc_d": self.val_metric_history['acc_d'][-1], \
                       "acc_d_real": self.val_metric_history['acc_d_real'][-1], "acc_d_fake": self.val_metric_history['acc_d_fake'][-1], "acc_g_label": self.val_metric_history['acc_g_label'][-1]}, \
                        prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        self.reset_val_loss_each_epoch()

    def on_save_checkpoint(self, checkpoint):
        checkpoint['train_loss_history'] = self.train_loss_history
        checkpoint['val_loss_history'] = self.val_loss_history
        checkpoint['val_metric_history'] = self.val_metric_history

    def on_load_checkpoint(self, checkpoint):
        if 'train_loss_history' in checkpoint.keys():
            self.train_loss_history = checkpoint['train_loss_history']
            self.val_loss_history = checkpoint['val_loss_history']
            self.val_metric_history = checkpoint['val_metric_history']