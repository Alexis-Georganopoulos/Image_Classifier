# Image Classification using CNNs in PyTorch

This source code is an implementation of a Convolutional Neural Network (CNN) for image classification using PyTorch. The dataset used for this implementation is the CIFAR-10 dataset, which contains 60,000 32x32 color images in 10 classes. The code includes data preprocessing, data splitting, and data loading, as well as the implementation of the CNN.
To meet the computation needs, the source code is run in Google Colab(using Cuda).<br>
[Run/view the code!](#running-the-code)

## Libraries
The following libraries are imported:

- `numpy` is imported for numerical computation.
- `torch` is imported for the implementation of the CNN.
- `torch.nn` is imported for building the CNN model.
- `torch.nn.functional` is imported for activation functions.
- `torchvision` is imported for loading the CIFAR-10 dataset.
- `torchvision.transforms` is imported for data preprocessing.
- `torch.utils.data` is imported for creating dataloaders.
- `matplotlib.pyplot` is imported for visualizing images.
- `IPython.display` is imported for setting the format of matplotlib figures.

## Data
The CIFAR-10 dataset is loaded using `torchvision.datasets.CIFAR10`. Data preprocessing is applied to the dataset using `torchvision.transforms.Compose`, which converts the data into PyTorch tensors and normalizes the pixel values. The transformed data is then loaded into three separate datasets: trainset, devset(validation set), and testset, using `torch.utils.data.Subset`. The datasets are transformed into dataloaders using `torch.utils.data.DataLoader`.

## Model
The CNN model is defined using `torch.nn.Module`. The model consists of three convolutional layers and three linear decision layers. The `forward` function is defined to execute the forward pass through the model. The output of the model is a probability distribution over the 10 possible classes. We use cross-entropy loss as the loss function for each epoch, with a batch size of 32.<br>
The exact architecture is as follows: <br>

1. Convolutional Layer: The first layer is a 2D convolutional layer that uses ReLU activation and applies 32 filters of size 3x3 to the input image.

1. Max Pooling Layer: The output of the convolutional layer is then passed through a max pooling layer that performs a downsampling operation with a pool size of 2x2.

1. Convolutional Layer: The second layer is another 2D convolutional layer with 64 filters of size 3x3, again using ReLU activation.

1. Max Pooling Layer: Another max pooling layer with a pool size of 2x2 is applied to the output of the second convolutional layer.

1. Flatten Layer: The output of the second max pooling layer is then flattened into a 1D vector.

1. Dense Layer: A fully connected layer with 128 units and ReLU activation is added on top of the flattened output.

1. Dropout Layer: A dropout layer is added to the model to prevent overfitting.

1. Output Layer: The final layer is a fully connected layer with a softmax activation function that outputs the probability distribution over the possible classes.

The model is trained on the training set and validated on the dev set to tune the hyperparameters and prevent overfitting. The test set is then used to evaluate the performance of the model on unseen data.<br>
A brief visualisation, along with the batch size and kernel sizes is shown below:

```python
#[batches, channels, kernel_width, kernel_height]
Input: [32, 3, 32, 32]
First CPBR block: [32, 64, 16, 16]
Second CPR block: [32, 128, 7, 7]
Third CPR block: [32, 256, 2, 2]
Vectorized: [32, 1024]
Final output: [32, 10]

Output size:
torch.Size([32, 10])#10 classes, softmaxed
```

## Running the code
To use this source code, you need to run the entire [`CNN_CIFAR10`](CNN_CIFAR10.ipynb) script. <br> 
Alternatively, you can view my [notebook](CNN_CIFAR10.ipynb) as-is, or run it directly in [Google Colab](https://colab.research.google.com/github/Alexis-Georganopoulos/CNN_CIFAR10/blob/main/CNN_CIFAR10.ipynb). <br>
The source code is self-contained and does not require any additional files. Once you run the script, the output will display the loss and accuracy of the model, as well as a visualization of some sample images. The CNN model will be defined and ready to train.