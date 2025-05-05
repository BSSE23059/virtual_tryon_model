import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from ..models.generator import Generator
from ..models.discriminator import Discriminator
from ...data.data_preprocessor import VirtualTryOnDataset

class Trainer:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_models()
        self.setup_data()
        self.setup_optimizers()
        
    def setup_models(self):
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        
    def setup_data(self):
        train_dataset = VirtualTryOnDataset(
            self.config['data']['train_dir'],
            split='train',
            img_size=self.config['data']['img_size']
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True
        )
        
    def setup_optimizers(self):
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.config['training']['lr'],
            betas=(self.config['training']['beta1'], 0.999)
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config['training']['lr'],
            betas=(self.config['training']['beta1'], 0.999)
        )

    def train(self):
        for epoch in range(self.config['training']['epochs']):
            self.train_epoch(epoch)
            
    # ...existing code...
    def train_epoch(self, epoch):
        self.generator.train()
        self.discriminator.train()
        total_g_loss = 0
        total_d_loss = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            person_img = batch['person'].to(self.device)
            bag_img = batch['bag'].to(self.device)
            pose_map = batch['pose'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # Combine inputs for generator
            gen_input = torch.cat([person_img, bag_img, pose_map], dim=1)
            
            # Train Discriminator
            self.d_optimizer.zero_grad()
            
            # Generate fake image
            fake_img = self.generator(gen_input)
            
            # Real image discriminator loss
            real_pred = self.discriminator(person_img)
            d_real_loss = torch.mean((real_pred - 1) ** 2)
            
            # Fake image discriminator loss
            fake_pred = self.discriminator(fake_img.detach())
            d_fake_loss = torch.mean(fake_pred ** 2)
            
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) * 0.5
            d_loss.backward()
            self.d_optimizer.step()
            
            # Train Generator
            self.g_optimizer.zero_grad()
            
            # Adversarial loss
            fake_pred = self.discriminator(fake_img)
            g_adv_loss = torch.mean((fake_pred - 1) ** 2)
            
            # L1 loss
            g_l1_loss = torch.mean(torch.abs(fake_img - person_img) * mask)
            
            # Total generator loss
            g_loss = g_adv_loss + self.config['training']['lambda_l1'] * g_l1_loss
            g_loss.backward()
            self.g_optimizer.step()
            
            # Update metrics
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, G_Loss: {g_loss.item():.4f}, D_Loss: {d_loss.item():.4f}')
        
        # Average losses for the epoch
        avg_g_loss = total_g_loss / len(self.train_loader)
        avg_d_loss = total_d_loss / len(self.train_loader)
        
        # Save checkpoint
        if (epoch + 1) % self.config['training']['save_interval'] == 0:
            self.save_checkpoint(epoch, avg_g_loss, avg_d_loss)
    
    def save_checkpoint(self, epoch, g_loss, d_loss):
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_loss': g_loss,
            'd_loss': d_loss
        }
        torch.save(checkpoint, f'checkpoints/model_epoch_{epoch}.pth')