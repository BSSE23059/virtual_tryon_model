import os
import torch
from torch.utils.data import DataLoader
from data.dataset import VirtualTryOnDataset
from data.data_preprocessor import DataPreprocessor
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def test_preprocessing():
    # Initialize preprocessor
    preprocessor = DataPreprocessor(img_size=3024)
    
    # Test with sample images
    test_person_path = os.path.join('dataset', 'train', 'persons', 'person_1.jpg')
    test_bag_path = os.path.join('dataset', 'train', 'Hand bags', 'bag_1.jpg')
    
    try:
        # Process a single sample
        sample = preprocessor.preprocess_sample(test_person_path, test_bag_path)
        print("Preprocessing test passed!")
        print(f"Output shapes:")
        for key, tensor in sample.items():
            print(f"{key}: {tensor.shape}")
        return True
    except Exception as e:
        print(f"Preprocessing test failed: {str(e)}")
        return False

def test_dataset():
    # Initialize dataset
    dataset = VirtualTryOnDataset(
        root_dir='dataset',
        split='train',
        img_size=3024,
        is_training=True
    )
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    try:
        # Get a sample batch
        sample_batch = next(iter(dataloader))
        print("\nDataset test passed!")
        print(f"Batch contents:")
        for key, tensor in sample_batch.items():
            if isinstance(tensor, torch.Tensor):
                print(f"{key}: {tensor.shape}")
            else:
                print(f"{key}: {type(tensor)}")
        return True
    except Exception as e:
        print(f"Dataset test failed: {str(e)}")
        return False

def visualize_sample(sample):
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Convert tensors to images and display
    images = {
        'Person': sample['person'],
        'Bag': sample['bag'],
        'Pose Map': sample['pose'],
        'Mask': sample['mask']
    }
    
    for idx, (title, img) in enumerate(images.items()):
        # Denormalize
        img = img.squeeze().cpu()
        if img.shape[0] == 3:  # RGB image
            img = (img + 1) / 2.0
        
        if len(img.shape) == 3:
            img = img.permute(1, 2, 0)  # Change from CxHxW to HxWxC
        
        axes[idx].imshow(img)
        axes[idx].set_title(title)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_visualization.png')
    plt.close()

def main():
    print("Testing Virtual Try-On Pipeline...")
    print("-" * 50)
    
    # Test preprocessing
    print("1. Testing Preprocessor...")
    preprocess_ok = test_preprocessing()
    
    # Test dataset
    print("\n2. Testing Dataset...")
    dataset_ok = test_dataset()
    
    if preprocess_ok and dataset_ok:
        print("\nAll tests passed! Testing visualization...")
        # Initialize dataset and get a sample
        dataset = VirtualTryOnDataset(
            root_dir='dataset',
            split='train',
            img_size=3024,
            is_training=True
        )
        sample = dataset[0]
        visualize_sample(sample)
        print("\nVisualization saved as 'sample_visualization.png'")
    else:
        print("\nSome tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()