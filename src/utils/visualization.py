import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        
    def save_images(self, visuals, epoch, iteration):
        """Save images to disk"""
        for label, image_batch in visuals.items():
            image_grid = vutils.make_grid(image_batch, normalize=True, scale_each=True)
            vutils.save_image(
                image_grid,
                f'{self.save_dir}/{label}_epoch_{epoch}_iter_{iteration}.png'
            )
    
    def plot_losses(self, g_losses, d_losses):
        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss")
        plt.plot(g_losses, label="G")
        plt.plot(d_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f'{self.save_dir}/losses.png')
        plt.close()