# Kinyarwanda Translation Model

This project implements a Kinyarwanda translation model using TensorFlow and Keras. It aims to translate English sentences into Kinyarwanda using a sequence-to-sequence architecture with LSTM and attention mechanisms.

## Table of Contents

- [Dataset Creation and Preprocessing](#dataset-creation-and-preprocessing)
- [Model Architecture and Design Choices](#model-architecture-and-design-choices)
- [Training Process and Hyperparameters Used](#training-process-and-hyperparameters-used)
- [Evaluation Metrics and Results](#evaluation-metrics-and-results)
- [Insights and Potential Improvements](#insights-and-potential-improvements)

## Dataset Creation and Preprocessing

The dataset consists of pairs of English and Kinyarwanda sentences. Here are the steps taken to preprocess the data:

Data Collection: The dataset is hardcoded within the script, consisting of simple sentence pairs for demonstration purposes.

```data = [
    ("Hello", "Muraho"),
    ("mwiriwe", "Good evening"),
    ("How are you?", "Mumeze mute?"),
    ("Good morning", "Mwaramutse"),
    ("Thank you", "Murakoze"),
    ("Goodbye", "Murabeho"),
]
```
- Tokenization: The English and Kinyarwanda sentences are tokenized using Tokenizer from Keras. Each word is assigned a unique index.
- Padding: The sequences are padded to ensure uniform input sizes using pad_sequences. This allows the model to process batches of sentences efficiently.
- Data Preparation: The processed data consists of padded sequences ready for training, along with their corresponding tokenizers for inverse transformations.

## Model Architecture and Design Choices
The model is designed using a sequence-to-sequence architecture with the following components:

- Encoder:

An embedding layer converts input word indices into dense vectors.
An LSTM layer processes the embedded vectors and outputs hidden states.
- Decoder:

Similar to the encoder, it uses an embedding layer followed by an LSTM.
An attention mechanism (AdditiveAttention) is implemented to focus on relevant parts of the input sequence when predicting the output sequence.
- Output Layer:

A dense layer with softmax activation predicts the next word in the output sequence.

## Design Choices

- Embedding Dimension: Set to 256 for a good balance between performance and computational efficiency.
- LSTM Units: Each LSTM layer has 256 units, providing sufficient capacity to capture relationships in the data.

## Training Process and Hyperparameters Used

The model is trained using the following settings:

Loss Function: Sparse categorical crossentropy is used since the target variable is a sequence of words.
Optimizer: Adam optimizer is chosen for its efficiency and ability to adapt learning rates.
Batch Size: 32, allowing for effective gradient updates while balancing memory usage.
Epochs: 50, sufficient for the model to learn from the dataset.
Training is performed with a validation split of 0.2 to monitor the model's performance on unseen data.

## Evaluation Metrics and Results

The model's performance is evaluated using the BLEU score, which measures the similarity between predicted translations and reference translations.

Average BLEU Score: Obtained from evaluating test sentences against true translations.

Example Evaluation
```python
test_sentences = ["Hello", "How are you?"]
test_sentences = ["Hello", "How are you?"]
true_translations = ["Muraho", "Mumeze mute?"]
bleu_score, predictions = evaluate_model(test_sentences, true_translations, encoder_model, decoder_model, tokenizer_eng, tokenizer_kin, max_eng_len, max_kin_len)
```

## Insights and Potential Improvements

## Insights
The model currently produces limited translations, with the BLEU score indicating room for improvement.
The dataset is small, which restricts the model's learning capability.

## Potential Improvements
- Larger Dataset: Expanding the dataset with more diverse and complex sentences could enhance model performance.
- Data Augmentation: Techniques like synonym replacement or back-translation can generate additional training samples.
- Fine-Tuning Hyperparameters: Exploring different LSTM units, embedding dimensions, and other hyperparameters could improve model accuracy.
- Experiment with Transformers: Considering modern architectures like transformers could lead to better performance in translation tasks.
- Regularization: Implementing dropout layers may help mitigate overfitting.

## Conclusion
This project serves as a foundational model for translating English to Kinyarwanda using neural networks. Future work should focus on improving data quality and model complexity to enhance translation accuracy.

## Additional Notes
Ensure that all relevant libraries (numpy, tensorflow, nltk) are installed in your environment.
