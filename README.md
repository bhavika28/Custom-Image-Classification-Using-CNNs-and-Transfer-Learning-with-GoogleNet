# Custom-Image-Classification-Using-CNNs-and-Transfer-Learning-with-GoogleNet


## Overview
This project focuses on creating a custom image classification model using two different approaches: a Convolutional Neural Network (CNN) built from scratch and transfer learning utilizing GoogleNet (InceptionNet). The main goal is to classify a dataset of custom images into three distinct categories and compare the performance between a custom-built CNN model and a transfer-learned GoogleNet model.

## Project Structure
Data Collection and Preparation:

Custom Dataset: A dataset is created with at least 100 images for each of three custom categories (minimum of 300 images in total).
Data Splitting: The dataset is split into training (80%) and testing (20%) subsets to allow for model training and evaluation.
Data Preprocessing:

The images are preprocessed as needed to ensure they are ready for model training. This may include resizing, normalization, or augmentation to enhance the model's generalization.
Model 1 - Custom Convolutional Neural Network (CNN):

A CNN model is created and trained on the custom training dataset to learn the features of the three image categories.
Training: The model is trained on the training data and its performance is validated using the test set.
Model 2 - Transfer Learning with GoogleNet (InceptionNet):

GoogleNet Initialization: The pre-trained GoogleNet model is imported, and a custom linear layer is added on top to adapt it to the classification task of the custom dataset.
Training and Fine-Tuning: The modified GoogleNet is trained on the custom dataset, and its performance is compared with that of the custom CNN model.
Evaluation and Comparison:

Prediction and Comparison: Both models make predictions on the test dataset, and their accuracies are compared to determine the effectiveness of transfer learning versus a custom-built CNN.
## Tools Used
To reproduce this project, ensure that you have the following libraries installed:

PyTorch and torchvision: For deep learning model building and transfer learning.
NumPy and Pandas: For data manipulation and processing.
OpenCV or PIL: For image loading and preprocessing.
Matplotlib: For visualizing results and accuracy plots.
