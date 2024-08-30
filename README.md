# Sentiment Analysis of IMDb Movie Reviews using LSTM

This project demonstrates sentiment analysis on IMDb movie reviews using a Long Short-Term Memory (LSTM) neural network implemented in TensorFlow/Keras.

## Table of Contents

- [Dataset](#dataset)
- [Steps](#steps)
- [Requirements](#requirements)
- [Running the Code](#running-the-code)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The project utilizes the IMDb dataset, which contains 50,000 movie reviews labeled as positive or negative. The dataset is readily available through the Keras library.

## Steps

1. **Load and Preprocess Data:** The IMDb dataset is loaded and preprocessed by:
   - Limiting the vocabulary to the top 10,000 most frequent words.
   - Padding sequences to a fixed length of 200 words.
   - Converting integer sequences back to text for tokenization.

2. **Tokenization:** A Tokenizer is created and fitted to the training text data to convert text into numerical sequences. This enables the model to process textual input.

3. **Model Building:** An LSTM model is constructed with the following layers:
   - Embedding layer to map words to dense vectors.
   - LSTM layer to capture long-term dependencies in the text.
   - Dense output layer with sigmoid activation for binary sentiment classification (positive or negative).

4. **Model Training:** The model is trained on the training data using:
   - Binary crossentropy loss function, suitable for binary classification.
   - Adam optimizer for efficient weight updates.
   - Accuracy metric to evaluate model performance.
   - TensorBoard callback for visualizing training progress and metrics.

5. **Model Evaluation:** The trained model is evaluated on the test data to assess its performance in terms of loss and accuracy.

6. **Prediction:** The model predicts the sentiment of new movie reviews by:
   - Tokenizing the new text.
   - Padding the sequences.
   - Feeding the sequences to the model for prediction.

7. **Visualization:** Training and validation loss and accuracy are plotted using Matplotlib to visualize the model's learning process and performance over epochs.

## Requirements

- Python 3
- TensorFlow
- Keras
- NumPy
- Scikit-learn
- Matplotlib

## Running the Code

1. **Install the required libraries:**
- bash pip install tensorflow keras numpy scikit-learn matplotlib

2. **Execute the provided Python code in a Google Colab environment.**

## Results

After training for 10 epochs, the model achieves an accuracy of around 85% on the test data. The visualization plots show the training and validation loss decreasing and accuracy increasing over epochs, indicating that the model is learning effectively.

## Future Work

- Experiment with different hyperparameters (e.g., number of LSTM units, learning rate) to potentially improve model performance.
- Explore more advanced text preprocessing techniques (e.g., stemming, lemmatization) to further enhance the model's ability to generalize.
- Incorporate pre-trained word embeddings (e.g., Word2Vec, GloVe) to leverage external knowledge and potentially improve accuracy.

## Contributing

Contributions to this project are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
