# Deep-Learning-Study
## CNN Architecture Series

- **1. CNN for MNIST Digit Recognition**
  - **Notebook:** 1_CNN_MNIST_Digit_Recognition.ipynb
  - **Explanation:** 1_CNN_MNIST_Digit_Recognition.txt
  - **Summary:** The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems. The MNIST database of handwritten digits has a training set of 60,000 28x28 grayscale images of the 10 digits and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image. 
    - **Test Set:** Average loss: 0.0296, Accuracy 9793/10000 (98%)

- **2. CNN for Fashion MNIST Classification**
  - **Notebook:** 2_CNN_Fashion_MNIST.ipynb
  - **Explanation:** 2_CNN_Fashion_MNIST.txt
  - **Summary:** Fashion MNIST is a popular dataset having thousands of pictures of clothes. It has been widely used to develop machine learning models for computer vision. The dataset is composed of 28x28 images in grayscale. Its popularity comes from being a "sibling" of the MNIST dataset of handwritten digits. 
    - **Test Set:** Average loss: 0.0331, Accuracy 8057/10000 (81%)

- **3. LeNet-50 Architecture**
  - **Notebook:** 3_CNN_Lenet_50.ipynb
  - **Explanation:** 3_CNN_Lenet_50.txt
  - **Summary:** 
    - **Model Architecture:** 
      - Input: (28, 28, 1)
      - **Conv Layers:** 3 convolutional layers with ReLU activation.
      - **Pooling:** Average pooling layers for dimensionality reduction.
      - **Dense Layers:** Two fully connected layers with 120 and 84 neurons.
      - **Compilation:** Adam optimizer, categorical cross-entropy loss, accuracy metric.
      - **Training:** 10 epochs, batch size of 128, 20% validation split.
      - **Performance:** Achieves ~99.03% test accuracy.

- **4. AlexNet for Image Classification**
  - **Notebook:** 4_CNN_Alexnet_Image_Classification.ipynb
  - **Explanation:** 4_CNN_Alexnet_Image_Classification.txt
  - **Summary:** AlexNet is a convolutional neural network designed for image classification, specifically adapted to the CIFAR-10 dataset. 
    - **Epochs:** 100 
    - **Optimizer:** Adam with a learning rate of 0.001. 
    - **Loss Function:** Categorical cross-entropy. 
    - **Results:** Achieved a training accuracy of approximately 91.07% and a validation accuracy of 53.24%.

- **5. ZFNet for CIFAR-10 Classification**
  - **Notebook:** 5_CNN_ZFNet_Cifar10_Classification.ipynb
  - **Explanation:** 5_CNN_ZFNet-Cifar10-Classification.txt
  - **Summary:** ZFNet-like architecture using the CIFAR-10 dataset, achieving a test accuracy of approximately 80.7%. It begins by importing necessary libraries for data manipulation and visualization, followed by loading the CIFAR-10 dataset and splitting it into training (80%) and validation (20%) sets. Class distributions are visualized through bar plots and pie charts, and the data is normalized and one-hot encoded. 
    - **Model:** Consists of five convolutional layers, three max pooling layers, and three fully connected layers, with an output layer adjusted for ten classes.

- **6. VGGNet for Image Classification**
  - **Notebook:** 6_CNN_VGGnet_Image_Classification.ipynb
  - **Explanation:** 6_CNN_VGGnet_Image_Classification.txt
  - **Summary:** VGGNet is a deep convolutional neural network architecture that was proposed by the Visual Graphics Group (VGG) at the University of Oxford. It gained popularity for its simplicity and achieved strong performance on various image classification tasks. This project implements a Convolutional Neural Network (CNN) based on the VGG16 architecture to classify images into two categories: dogs and cats. The VGG16 model, pre-trained on the ImageNet dataset, is utilized to leverage transfer learning, enhancing classification accuracy on the smaller Dogs vs. Cats dataset.

- **7. ResNet for Image Classification**
  - **Notebook:** 7_CNN_Resnet_image_classification.ipynb
  - **Explanation:** 7_CNN_Resnet_image_classification.pdf
  - **Summary:** The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. It is split into 50,000 training images and 10,000 test images. The model uses Stochastic Gradient Descent (SGD) with a learning rate of 0.1, momentum of 0.9, and weight decay of 0.0001. A learning rate scheduler is employed to adjust the learning rate during training. Training runs for 50 epochs with a mini-batch size of 128.

- **8. Transfer Learning: Feature Extraction**
  - **Notebook:** 8_Transfer_learning_Feature_extraction.ipynb
  - **Explanation:** 8_Transfer_learning_Feature_extraction.txt
  - **Summary:** This project utilizes transfer learning with the VGG16 model for binary classification of dog and cat images using the CIFAR-10 dataset. The VGG16 model, pre-trained on the ImageNet dataset, serves as the convolutional base, which is fine-tuned for our specific task by appending custom fully connected layers. The input images are resized to 150x150 pixels, and data augmentation techniques such as shear, zoom, and horizontal flip are applied to the training dataset to enhance model generalization. The model is compiled with the Adam optimizer and binary cross-entropy loss, tracking accuracy as the performance metric. After training for 10 epochs, the model demonstrates steady improvement in accuracy, achieving training accuracy of approximately 93.98% and validation accuracy peaking at 91.94%. The results highlight the effectiveness of using pre-trained models for image classification tasks, making this approach suitable for applications requiring rapid and accurate image recognition.

- **9. Transfer Learning: Fine Tuning**
  - **Notebook:** 9_Transfer_learning_Fine_Tuning.ipynb
  - **Explanation:** 9_Transfer_learning_Fine_Tuning.txt
  - **Summary:** This project utilizes transfer learning with the VGG16 model to classify images of dogs and cats. The VGG16 model, pre-trained on the ImageNet dataset, is employed as the base model without its top classification layers to extract features from the images. The input images are resized to 150x150 pixels and augmented using techniques such as rescaling, shearing, zooming, and horizontal flipping to enhance the dataset. A custom Sequential model is created by adding the VGG16 base model followed by a Flatten layer and two Dense layers for classification—one with 256 neurons using ReLU activation and a final output layer with sigmoid activation for binary classification. The model is compiled using the RMSprop optimizer and trained for 10 epochs, achieving an accuracy improvement from approximately 82.92% in the first epoch to about 97.95% in the last epoch, indicating effective learning and generalization to the validation dataset. The training process is visualized using Matplotlib to plot the accuracy over epochs.
