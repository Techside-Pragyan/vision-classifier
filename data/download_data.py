import os
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def save_dataset_as_images():
    print("Loading CIFAR-10 dataset via torchvision...")
    
    # Minimal transform just to get the images as PIL
    transform = transforms.Compose([transforms.ToPILImage()])
    
    trainset = torchvision.datasets.CIFAR10(root='./data_cache', train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root='./data_cache', train=False, download=True)
    
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    
    # Create directories
    for dir_path in [train_dir, test_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            for cls in classes:
                os.makedirs(os.path.join(dir_path, cls))

    print("Saving training images...")
    samples_per_class = 200 
    class_counts = {cls: 0 for cls in classes}
    
    for i, (img, label) in enumerate(trainset):
        cls_name = classes[label]
        if class_counts[cls_name] < samples_per_class:
            img.save(os.path.join(train_dir, cls_name, f"train_{i}.png"))
            class_counts[cls_name] += 1
        
        # Stop if all classes have enough samples
        if all(count >= samples_per_class for count in class_counts.values()):
            break
            
    print("Saving test images...")
    class_counts_test = {cls: 0 for cls in classes}
    for i, (img, label) in enumerate(testset):
        cls_name = classes[label]
        if class_counts_test[cls_name] < 50:
            img.save(os.path.join(test_dir, cls_name, f"test_{i}.png"))
            class_counts_test[cls_name] += 1
        
        if all(count >= 50 for count in class_counts_test.values()):
            break

    print(f"Data prepared in {base_dir}")

if __name__ == "__main__":
    save_dataset_as_images()
