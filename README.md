# DL_BNN
## Deep Learning Project: Bayesian Neural Network

In this project, a **Bayesian Neural Network (BNN)** is used to solve an image classification task. Specifically, a **Bayesian Convolutional Neural Network (CNN)** is implemented to handle image classification. 

At the first stage of the project, the Bayesian CNN implementation is trained on the **CIFAR-10** dataset. After achieving some success with CIFAR-10, a more challenging dataset will be used. The goal with CIFAR-10 is to classify images into one of 10 categories, such as "cat," "dog," "airplane," etc., using a network that not only makes predictions but also measures how uncertain it is about those predictions.

## Key Elements of the Model

### CIFAR-10 Dataset
The **CIFAR-10** dataset consists of 60,000 images, divided into 10 different classes, with 6,000 images per class. The images are small (32x32 pixels) and are in color (RGB). The task is to train a model that can correctly classify these images into their respective classes.

### What I Have Done in the BayesianCNN Implementation
In this implementation, I created a **Convolutional Neural Network (CNN)** that uses **Bayesian methods** to handle uncertainty in its predictions. Here's a breakdown of the approach:

#### 1. Bayesian Convolutional Layers (BayesianConv2dFlipout)
- I implemented custom convolutional layers where the **weights** (and **biases**) are treated as **random variables** with a distribution (using a **Normal distribution**).
- These layers use a technique called **Flipout**, which helps in efficiently approximating the uncertainty in the model's weights during training.
- For each weight and bias, I defined the **mean** and **log-standard deviation**. The mean represents the most likely value, and the log-standard deviation tells us how much uncertainty is associated with that weight.

#### 2. Bayesian Fully Connected Layers (BayesianDenseFlipout)
- Similar to the convolutional layers, I created fully connected layers where the **weights** and **biases** are probabilistic.
- These layers follow the same principle as the convolutional layers: weights and biases are learned as **distributions**, not fixed values.

#### 3. Bayesian CNN Model (BayesianCNN)
- The `BayesianCNN` class connects the Bayesian convolutional layers and fully connected layers into a complete neural network.
- The architecture is similar to a regular CNN with several **convolutional layers** for feature extraction followed by **fully connected layers** for classification.
- I used the **Flipout technique** to efficiently estimate uncertainty during the forward pass of the model.

#### 4. Forward Pass with Uncertainty
- In the forward pass, I used both **stochastic** (uncertain) and **deterministic** (fixed) modes for sampling the weights and biases.
- This allows the model to make predictions while also estimating how uncertain it is about those predictions.
- The final output is a **log-softmax** result for classification, which gives the model's predicted class probabilities.

### Uncertainty in Predictions
By using these **Bayesian methods**, the model can not only make predictions but also tell us how confident it is about those predictions. This is particularly helpful in scenarios where we need to understand the model's confidence, such as tasks involving **out-of-distribution detection**.

However, the model only achieved around **10% accuracy**, which is quite low. This suggests that the model architecture may not yet be fully learning the patterns in the data.

### Next Steps
In the future, I plan to continue training the model for more than **100 epochs** to see if it can improve its performance. Extending the training time will give the model more opportunity to learn from the data, and I will be able to derive conclusions about the architecture’s ability to learn.

Along with this, I’ll keep monitoring the model's **uncertainty estimates** to better understand how confident it is in its predictions. This will help in determining whether the model is improving its understanding over time and making better decisions.

## Conclusion
This project demonstrates how a **Bayesian Neural Network** can be applied to image classification tasks, using techniques like **Flipout** to handle uncertainty in the predictions. The initial results are promising, and further training and experimentation will help refine the model's performance.
