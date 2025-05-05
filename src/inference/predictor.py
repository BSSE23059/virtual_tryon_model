import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from ..models.generator import Generator
from ..models.geometric_matching import GeometricMatchingModule
from ...data.data_preprocessor import DataPreprocessor

class VirtualTryOnPredictor:
    def __init__(self, checkpoint_path, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocessor = DataPreprocessor()
        
        # Load models
        self.generator = Generator().to(self.device)
        self.geometric_matcher = GeometricMatchingModule().to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.generator.eval()
        
    def predict(self, person_path, bag_path):
        """Generate virtual try-on result"""
        # Preprocess inputs
        data = self.preprocessor.preprocess_sample(person_path, bag_path)
        
        with torch.no_grad():
            # Move to device
            person = data['person'].unsqueeze(0).to(self.device)
            bag = data['bag'].unsqueeze(0).to(self.device)
            pose = data['pose'].unsqueeze(0).to(self.device)
            
            # Get geometric transformation
            flow_field = self.geometric_matcher(bag, person)
            
            # Generate try-on result
            inputs = torch.cat([person, bag, pose], dim=1)
            result = self.generator(inputs)
            
        return self.postprocess_output(result)
    
    def postprocess_output(self, tensor):
        """Convert output tensor to image"""
        image = tensor.cpu().squeeze(0)
        image = (image + 1) / 2.0  # denormalize
        image = image.clamp(0, 1)
        image = (image * 255).numpy().transpose(1, 2, 0).astype('uint8')
        return Image.fromarray(image)