#libraries
import streamlit as st
import torch
from torchvision import transforms , models
from torch.nn.functional import softmax
import timm



# Define a custom model class that uses a ResNet50 architecture from the timm library
class YourModelClass(torch.nn.Module):  # class inheriting from torch.nn.Module
    def __init__(self, num_classes):    # number of classes that model defines 
        super(YourModelClass, self).__init__()
        self.model = timm.create_model('resnet50', pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
    
 
# Function to preprocess an input image
def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image


# Function to load a TorchScript model from a file
def load_torchscript_model(model_path, device):
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model
 

# Function to load a model from a state dictionary
def load_state_dict_model(model_path, num_classes, device):
    model = YourModelClass(num_classes)  # Update with your model class
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# Function to predict the class of an image using a model
def predict(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
    probabilities = softmax(outputs, dim=1)
    _, predicted = torch.max(outputs, 1)
    return predicted.item(), probabilities.squeeze().cpu().numpy()


# Function to predict the class of an image using the model
def predict_image(image, model, device):
    image_tensor = preprocess_image(image).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()


