# Fashion-MNIST Classification with PyTorch

Deep learning classification of clothing items using PyTorch neural networks.

## Overview

This project implements two neural network architectures to classify Fashion-MNIST images:
- A simple feedforward neural network
- A Convolutional Neural Network (CNN)

The CNN achieved **89.90% test accuracy**, demonstrating the effectiveness of convolutional layers for image classification tasks.

## Dataset

**Fashion-MNIST** contains 70,000 grayscale images of clothing items:
- **Training set:** 60,000 images
- **Test set:** 10,000 images  
- **Image size:** 28×28 pixels
- **Classes (10):** T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

## Models

### Model 1: Feedforward Neural Network
**Architecture:**
- Flatten layer (28×28 → 784)
- Linear layer (784 → 128)
- ReLU activation
- Linear layer (128 → 10)

**Results:**
- Training Accuracy: 92.08% 
- Test Accuracy: 88.46%

### Model 2: Convolutional Neural Network (CNN)
**Architecture:**
- 2 Convolutional blocks (Conv2d → ReLU → Conv2d → ReLU → MaxPool2d)
- Flatten layer
- Linear classifier (hidden_units * 7 * 7 = 10 * 7 * 7 = 490)

**Results:**
- Training Accuracy: 92.06%
- Test Accuracy: 90.34%

## Performance Comparison

| Model | Train Accuracy | Test Accuracy |
|-------|----------------|---------------|
| Feedforward | 92.08% | 88.46% |
| CNN | 92.06% | 90.34% |

The CNN outperformed the simple feedforward network by **2.34%**, showing the advantage of convolutional layers for spatial pattern recognition in images.

## Technologies Used

- **PyTorch** - Deep learning framework
- **torchvision** - Dataset loading and transforms
- **matplotlib** - Data visualization

## Training Details

- **Loss Function:** CrossEntropyLoss
- **Optimizer:** SGD (learning rate: 0.1)
- **Batch Size:** 32
- **Epochs:** 10
- **Device:** GPU (CUDA) if available, else CPU

## Key Learning Outcomes

- Implemented PyTorch training loops with forward pass, loss calculation, and backpropagation
- Built and compared feedforward vs convolutional architectures
- Handled image data with PyTorch DataLoaders
- Evaluated model performance on multi-class classification (10 classes)

## Usage
Open `Fashion_MNIST.ipynb` in Google Colab or Jupyter Notebook and run all cells.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ERVVSuGP6BtNchSt5njxTzBqGhHmaCFJ?usp=sharing)

## Results Visualization

The model correctly classifies various clothing items with ~90% accuracy, effectively distinguishing between similar categories like different types of shirts and shoes.

## Future Improvements

- Experiment with deeper CNN architectures
- Add data augmentation (rotation, flipping)
- Implement learning rate scheduling
- Try different optimizers (Adam, AdamW)
