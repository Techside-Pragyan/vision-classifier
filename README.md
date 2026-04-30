# Vision Classifier

A simple yet effective image classification pipeline using PyTorch, featuring transfer learning with MobileNetV2 on the CIFAR-10 dataset. 

This project demonstrates how to set up a vision-based deep learning model, from data preparation to training and inference preparation.

## Project Structure

```
vision-classifier/
├── data/
│   └── download_data.py   # Script to download and prepare CIFAR-10 dataset
├── train_model.py         # Training script using MobileNetV2 transfer learning
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Setup & Installation

1. Clone the repository and navigate to the project directory.
2. Ensure you have Python 3.8+ installed.
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

The project uses a subset of the CIFAR-10 dataset for fast training. The `download_data.py` script automatically downloads the dataset and saves it as individual `.png` images, organized by class.

To prepare the dataset:

```bash
python data/download_data.py
```

This script will:
- Download CIFAR-10 to `data_cache/`
- Extract 200 training images and 50 test images per class (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- Save them in the `data/train/` and `data/test/` directories.

## Training the Model

The training script (`train_model.py`) performs transfer learning using a pre-trained **MobileNetV2** backbone. The backbone is frozen, and only the final classifier head is trained for the 10 CIFAR-10 classes.

To start training:

```bash
python train_model.py
```

### What happens during training?
- **Data Augmentation:** The training data undergoes random resized crops and horizontal flips.
- **Hardware Acceleration:** The script will automatically use a GPU (`cuda:0`) if available, otherwise it falls back to the CPU.
- **Artifacts:**
  - `models/classes.json`: Contains the names of the 10 classes.
  - `models/vision_classifier.pth`: The saved PyTorch model with the best validation accuracy.
  - `models/history.json`: The training history (loss and accuracy for train and validation phases).

## Future Extensions

With `fastapi`, `uvicorn`, and `python-multipart` included in the `requirements.txt`, the logical next step is to wrap the trained model into a REST API for real-time inference.