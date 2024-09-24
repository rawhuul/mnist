use std::io::{Cursor, Read};

const INPUT_SIZE: usize = 784;
const HIDDEN_SIZE: usize = 256;
const OUTPUT_SIZE: usize = 10;
const BATCH_SIZE: usize = 64;
const IMAGE_SIZE: usize = 28;

const EPOCHS: u8 = 20;

const LEARNING_RATE: f32 = 0.001;
const TRAIN_SPLIT: f32 = 0.8;

#[derive(Debug)]
struct Layer {
    weights: Vec<f32>,
    biases: Vec<f32>,
    input_size: usize,
    output_size: usize,
}

impl Layer {
    fn new(input_size: usize, output_size: usize) -> Self {
        let n = input_size * output_size;
        let scale = f32::sqrt(2.0 / input_size as f32);

        let biases = vec![0.0; n];
        let mut weights = vec![0.0; n];

        for i in 0..n {
            weights[i] = (fastrand::f32() - 0.5) * 2.0 * scale;
        }

        Self { weights, biases, input_size, output_size }
    }

    fn forward(&self, input: &[f32], output: &mut [f32]) {
        for i in 0..self.output_size {
            output[i] = self.biases[i];
            for j in 0..self.input_size {
                output[i] += input[j] * self.weights[j * self.output_size + i];
            }
        }
    }

    fn backward(
        &mut self,
        input: &[f32],
        output_grad: &[f32],
        input_grad: &mut [f32],
        lr: f32,
    ) {
        for i in 0..self.output_size {
            for j in 0..self.input_size {
                let idx = j * self.output_size + i;

                let grad = output_grad.get(i).unwrap_or(&0.0)
                    * input.get(j).unwrap_or(&0.0);

                self.weights[idx] -= lr * grad;

                if let Some(ig) = input_grad.get_mut(j) {
                    *ig +=
                        output_grad.get(i).unwrap_or(&0.0) * self.weights[idx];
                }
            }

            self.biases[i] -= lr * output_grad.get(i).unwrap_or(&0.0);
        }
    }
}

#[derive(Debug)]
struct Network {
    hidden: Layer,
    output: Layer,
}

impl Network {
    fn new() -> Self {
        let hidden = Layer::new(INPUT_SIZE, HIDDEN_SIZE);
        let output = Layer::new(HIDDEN_SIZE, OUTPUT_SIZE);

        Self { hidden, output }
    }

    fn train(&mut self, input: &[f32], label: u32, lr: f32) {
        let mut hidden_output = [0.0f32; HIDDEN_SIZE];
        let mut final_output = [0.0f32; OUTPUT_SIZE];

        let mut hidden_grad = [0.0f32; HIDDEN_SIZE];
        let mut output_grad = [0.0f32; OUTPUT_SIZE];

        self.hidden.forward(input, &mut hidden_output);

        // ReLU
        for val in hidden_output.iter_mut() {
            *val = if *val > 0.0 { *val } else { 0.0 }
        }

        self.output.forward(&hidden_output, &mut final_output);
        softmax(&mut final_output);

        for i in 0..OUTPUT_SIZE {
            let v = if i == label as usize { 1.0 } else { 0.0 };
            output_grad[i] = final_output[i] - v;
        }

        self.hidden.backward(
            &hidden_output,
            &output_grad,
            &mut hidden_grad,
            lr,
        );

        // ReLU derivative
        for i in 0..HIDDEN_SIZE {
            hidden_grad[i] *= if hidden_output[i] > 0.0 { 1.0 } else { 0.0 }
        }

        self.hidden.backward(input, &hidden_grad, &mut [], lr)
    }

    fn predict(&self, input: &[f32]) -> usize {
        let mut hidden_output = [0f32; HIDDEN_SIZE];
        let mut final_output = [0f32; OUTPUT_SIZE];

        self.hidden.forward(input, &mut hidden_output);

        // ReLU
        for val in hidden_output.iter_mut() {
            *val = if *val > 0.0 { *val } else { 0.0 }
        }

        self.output.forward(&hidden_output, &mut final_output);
        softmax(&mut final_output);

        let mut max_index = 0;

        for i in 0..OUTPUT_SIZE {
            if final_output[i] > final_output[max_index] {
                max_index = i;
            }
        }

        max_index
    }
}

#[derive(Debug)]
struct InputData {
    images: Vec<u8>,
    labels: Vec<u8>,
    n_images: usize,
}

impl InputData {
    fn new() -> Self {
        let images = read_mnist_images();
        let labels = read_mnist_labels();

        Self { images, labels, n_images: 60000 }
    }

    fn shuffle(&mut self) {
        fastrand::shuffle(&mut self.images);
        fastrand::shuffle(&mut self.labels);
    }
}

fn softmax(input: &mut [f32]) {
    let mut max = input[0];

    for i in 1..input.len() {
        if input[i] > max {
            max = input[i];
        }
    }

    let mut sum = 0f32;

    for val in input.iter_mut() {
        *val = f32::exp(*val - max);
        sum += *val;
    }

    for val in input.iter_mut() {
        *val /= sum;
    }
}

fn read_mnist_images() -> Vec<u8> {
    let data = include_bytes!("./data/train-images.idx3-ubyte");

    let mut cursor = Cursor::new(data);

    let mut bytes = [0u8; 4];

    // Ignoring first 4 bytes
    cursor.read_exact(&mut bytes).unwrap();

    cursor.read_exact(&mut bytes).unwrap();
    let num_images = u32::from_be_bytes(bytes);
    assert_eq!(
        num_images, 60000,
        "Number of images must be 60000, got {num_images}"
    );

    cursor.read_exact(&mut bytes).unwrap();
    let rows = u32::from_be_bytes(bytes);
    assert_eq!(rows, 28, "Number of rows must be 28, got {num_images}");

    cursor.read_exact(&mut bytes).unwrap();
    let cols = u32::from_be_bytes(bytes);
    assert_eq!(cols, 28, "Number of cols must be 28, got {num_images}");

    let mut images = vec![0u8; 60000 * IMAGE_SIZE * IMAGE_SIZE];
    cursor.read_exact(&mut images).unwrap();

    images
}

fn read_mnist_labels() -> Vec<u8> {
    let data = include_bytes!("./data/train-labels.idx1-ubyte");

    let mut cursor = Cursor::new(data);
    let mut bytes = [0u8; 4];

    // Ignoring first 4 bytes.
    cursor.read_exact(&mut bytes).unwrap();

    cursor.read_exact(&mut bytes).unwrap();
    let num_lables = u32::from_be_bytes(bytes);
    assert_eq!(
        num_lables, 60000,
        "Number of labels must be 60000, got {num_lables}"
    );

    let mut labels = vec![0u8; 60000];
    cursor.read_exact(&mut labels).unwrap();

    labels
}

fn main() {
    let mut network = Network::new();
    let mut data = InputData::new();
    let mut img = [0f32; INPUT_SIZE];

    data.shuffle();

    let train_size = (data.n_images as f32 * TRAIN_SPLIT) as usize;
    let test_size = data.n_images - train_size;

    for epoch in 0..EPOCHS {
        let mut total_loss = 0f32;

        let mut i = 0;

        while i < train_size {
            let mut j = 0;

            while j < BATCH_SIZE && i + j < train_size {
                let idx = i + j;

                for k in 0..INPUT_SIZE {
                    img[k] =
                        (data.images[idx * INPUT_SIZE + k] / u8::MAX) as f32;
                }

                network.train(&img, data.labels[idx] as u32, LEARNING_RATE);
                let mut hidden_output = [0f32; HIDDEN_SIZE];
                let mut final_output = [0f32; OUTPUT_SIZE];

                network.hidden.forward(&img, &mut hidden_output);

                // ReLU
                for k in 0..HIDDEN_SIZE {
                    hidden_output[k] = if hidden_output[k] > 0.0 {
                        hidden_output[k]
                    } else {
                        0.0
                    };
                }

                network.output.forward(&hidden_output, &mut final_output);
                softmax(&mut final_output);

                total_loss += -std::f32::consts::E.log(
                    final_output[data.labels[idx as usize] as usize] + 1e-10,
                );
                j += 1;
            }
            i += BATCH_SIZE;
        }

        let mut correct = 0;

        for i in train_size..data.n_images {
            for k in 0..INPUT_SIZE {
                img[k] = (data.images[i * INPUT_SIZE + k] / 255) as f32;
            }

            if network.predict(&img) == data.labels[i] as usize {
                correct += 1;
            }
        }

        println!(
            "Epoch {}, Accuracy: {}%, Avg Loss: {}",
            epoch + 1,
            correct / test_size * 100,
            total_loss / train_size as f32
        );
    }
}
