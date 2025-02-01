# custom-CNNs-for-image-classification

Project Overview:

This project implements custom Convolutional Neural Networks (CNNs) using PyTorch for image classification. 

The workflow includes:

Creating a custom dataset with three different categories (each containing at least 100 images).
Splitting the dataset into 80% training and 20% testing.
Preprocessing the dataset for optimal CNN training.
Building and training a CNN model for image classification.
Evaluating the model by making predictions on test data.
Using GoogleNet (InceptionNet) as a second model, modifying it by adding a Linear layer.
Comparing accuracy between the custom CNN and GoogleNet models.

Dataset:
The dataset consists of three custom categories (e.g., Cats, Dogs, and Birds).
Each category contains at least 100 images.
Images are preprocessed and resized before being fed into the CNN.

Model Training Workflow

1.Data Preprocessing
Convert images to tensors.
Apply data augmentation techniques (e.g., resizing, normalization).
Create PyTorch DataLoaders for efficient batch processing.

2.Custom CNN Model (Baseline Model)
Build a CNN model with multiple convolutional layers.
Use ReLU activation, max-pooling, and fully connected layers for classification.

3.GoogleNet (InceptionNet) Modification
Use GoogleNet as a pre-trained model.
Modify the final layer by adding a Linear layer to adjust output classes.

4.Model Training
Train both the Custom CNN and GoogleNet models using:
CrossEntropy Loss,
Adam optimizer,
Learning rate scheduling

5.Model Evaluation
Predict and compare results on the test set.
Compute accuracy, precision, recall, and F1-score.
Compare baseline CNN vs. GoogleNet performance.

Results:
Custom CNN and GoogleNet model accuracy are compared side by side.
A visual representation of training loss and accuracy is plotted.
