# MNIST Neural Network in Rust

This project features a **minimal** neural network implementation in Rust for classifying handwritten digits from the [MNIST dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset?resource=download). The implementation utilizes only the standard Rust library and `fastrand` for random number generation.

It is inspired from [miniMNIST-c](https://github.com/konrad-gajdus/miniMNIST-c/)

## Features

- Two-layer neural network architecture (input → hidden → output)
- ReLU activation function for the hidden layer
- Softmax activation function for the output layer
- Cross-entropy loss function
- Stochastic Gradient Descent (SGD) optimizer

## Configuration

You can customize the following parameters in `main.rs`:

- `HIDDEN_SIZE`: Number of neurons in the hidden layer
- `LEARNING_RATE`: Learning rate for SGD
- `EPOCHS`: Number of training epochs
- `BATCH_SIZE`: Size of mini-batches for training
- `TRAIN_SPLIT`: Proportion of data allocated for training (the remainder is used for testing)

## License

This project is open-source and available under the [MIT License](LICENSE).
