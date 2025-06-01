
# Fashion MNIST Classification Using a Convolutional Neural Network

## Overview

This script builds and trains a Convolutional Neural Network (CNN) to classify images from the Fashion MNIST dataset. The dataset contains grayscale images of clothing items like shirts, trousers, and shoes.

The network is trained to recognize these items based on visual patterns and textures using a layered architecture that mimics how the human eye might scan an image.

---

## Dataset Used

The dataset used is **Fashion MNIST**, which comes built-in with TensorFlow. It contains:

* 60,000 training images
* 10,000 test images
* Each image is 28x28 pixels, grayscale
* 10 classes of clothing items (T-shirt, trouser, sneaker, etc.)

---

## What the Code Does

### 1. Load and Prepare the Data

The dataset is loaded and reshaped so that each image has a shape of (28, 28, 1). This third dimension is required for TensorFlow’s convolutional layers.

The pixel values are scaled to a range of 0 to 1 to help the model train faster and more efficiently.

```python
training_images = training_images.reshape(60000, 28, 28, 1) / 255.0
```

### 2. Build the CNN Model

The model is built using TensorFlow’s `Sequential` API. Here's what each layer does:

* **Conv2D (32 filters)**: Looks at small patches of the image to detect features.
* **MaxPooling2D**: Shrinks the image while keeping important features.
* **Conv2D (64 filters)**: Extracts deeper, more complex patterns.
* **MaxPooling2D**: Again shrinks down the image to reduce computation.
* **Flatten**: Converts the 2D data into a 1D vector.
* **Dense (128 units)**: Fully connected layer that learns from the extracted features.
* **Dense (10 units, softmax)**: Outputs probabilities for each of the 10 clothing categories.

### 3. Compile and Train the Model

The model uses:

* **Adam** optimizer for training
* **Sparse Categorical Crossentropy** as the loss function
* **Accuracy** as the performance metric

It is trained for 5 epochs on the training set.

```python
model.fit(training_images, training_labels, epochs=5)
```

### 4. Evaluate the Model

After training, the model is evaluated on the test set to measure how well it generalizes.

```python
test_loss, test_acc = model.evaluate(test_images, test_labels)
```

The result prints the test loss and accuracy.

---

## How to Run This

### Prerequisites

Make sure you have Python and TensorFlow installed.

Install TensorFlow using:

```
pip install tensorflow
```

### Running the Script

Save the script as `fashion_cnn.py` and run:

```
python fashion_cnn.py
```

You’ll see the training process in the console and finally the test accuracy.

---

## Notes for Practice

* Try increasing the number of epochs to see if accuracy improves.
* Add `Dropout` layers to prevent overfitting.
* Try changing the filter sizes or number of filters.
* Use different optimizers like `RMSprop` or `SGD`.

---

## Why This is Useful

This is a fundamental example of using CNNs for image classification. It's a common starting point for learning deep learning in vision tasks. Once you're confident with this, you can move on to more complex datasets and architectures like ResNet or MobileNet.

