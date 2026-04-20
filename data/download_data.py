import os
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import numpy as np
from PIL import Image

def save_dataset_as_images():
    print("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
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
    # Limit to 500 images per class for speed in this demo, but CIFAR has thousands
    # We'll save a reasonable amount for a quick training session
    samples_per_class = 200 
    
    class_counts = {cls: 0 for cls in classes}
    for i in range(len(x_train)):
        cls_idx = y_train[i][0]
        cls_name = classes[cls_idx]
        if class_counts[cls_name] < samples_per_class:
            img = Image.fromarray(x_train[i])
            img.save(os.path.join(train_dir, cls_name, f"train_{i}.png"))
            class_counts[cls_name] += 1
            
    print("Saving test images...")
    class_counts_test = {cls: 0 for cls in classes}
    for i in range(len(x_test)):
        cls_idx = y_test[i][0]
        cls_name = classes[cls_idx]
        if class_counts_test[cls_name] < 50: # 50 per class for testing
            img = Image.fromarray(x_test[i])
            img.save(os.path.join(test_dir, cls_name, f"test_{i}.png"))
            class_counts_test[cls_name] += 1

    print(f"Data prepared in {base_dir}")

if __name__ == "__main__":
    save_dataset_as_images()
