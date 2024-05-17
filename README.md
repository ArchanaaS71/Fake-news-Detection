# Fake News Detection with Deep Learning

## Overview

The proliferation of fake news presents a significant challenge in the digital age, necessitating robust solutions for its detection and mitigation. This repository presents a comprehensive fake news detection project using a dataset sourced from Kaggle. The dataset consists of two CSV files: `real.csv` and `fake.csv`, containing 21,417 and 23,481 articles respectively.

## Dataset

The dataset used in this project includes:
- `real.csv`: Contains 21,417 real news articles.
- `fake.csv`: Contains 23,481 fake news articles.

## Models

We explore the efficacy of various deep learning architectures to distinguish between real and fake news articles, including:
- Long Short-Term Memory (LSTM)
- Convolutional Neural Network (CNN)
- Bidirectional LSTM
- ResNet
- COBRA
- Gated Recurrent Unit (GRU)

## Performance

Our experimental results demonstrate promising performance across the models, with accuracy metrics ranging from 0.76 to 0.97. Notably, the LSTM, Bidirectional LSTM, COBRA, and GRU models exhibit particularly high accuracy scores of 0.97 each, showcasing their effectiveness in discriminating between genuine and fabricated news articles.

## Contributions

Through this comparative analysis, we provide insights into the strengths and limitations of different machine learning approaches for fake news detection. Our findings contribute to the ongoing efforts in developing robust tools for combating misinformation and safeguarding the integrity of online information ecosystems.

## Usage

Clone the repository and follow the instructions to run the models and reproduce the results.

```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
```

## Requirements

Ensure you have the following dependencies installed:

- Python 3.x
- TensorFlow
- Keras
- Pandas
- NumPy
- Scikit-learn

## Installation

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Running the Models

Follow the provided notebooks and scripts to train and evaluate the models on the fake news detection dataset.

## Results

The results of the models will be displayed, showcasing accuracy and other relevant metrics.

## Conclusion

This repository offers a comprehensive approach to fake news detection using various deep learning models, highlighting the potential of these models in identifying misinformation with high accuracy.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

Feel free to contribute to this project by submitting issues or pull requests. For any questions or feedback, please contact [17archanaas.@gmail.com].
