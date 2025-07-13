# Text Classification with Neural Networks

A comprehensive text classification project comparing the performance of different recurrent neural network architectures: RNN, LSTM, and BiLSTM using sentence embeddings.

##  Overview

This project implements and compares three different recurrent neural network architectures for text classification tasks:

- **Simple RNN (Recurrent Neural Network)**
- **LSTM (Long Short-Term Memory)**
- **BiLSTM (Bidirectional LSTM)**

The models use pre-trained sentence embeddings from SBERT (Sentence-BERT) to convert text into numerical representations, which are then fed into the neural networks for classification.

##  Architecture

### Model Architectures

#### 1. Simple RNN
```python
def build_rnn(input_shape, num_classes):
    model = Sequential([
        Input(shape=(input_shape,)),
        Reshape((input_shape, 1)),
        SimpleRNN(128),
        Dense(num_classes, activation='softmax')
    ])
```

#### 2. LSTM
```python
def build_lstm(input_shape, num_classes):
    model = Sequential([
        Input(shape=(input_shape,)),
        Reshape((input_shape, 1)),
        LSTM(128),
        Dense(num_classes, activation='softmax')
    ])
```

#### 3. BiLSTM
```python
def build_bilstm(input_shape, num_classes):
    model = Sequential([
        Input(shape=(input_shape,)),
        Reshape((input_shape, 1)),
        Bidirectional(LSTM(128)),
        Dense(num_classes, activation='softmax')
    ])
```

##  Features

- **Sentence Embeddings**: Uses SBERT (`all-MiniLM-L6-v2`) for high-quality text representations
- **Multiple Architectures**: Compares RNN, LSTM, and BiLSTM performance
- **Comprehensive Evaluation**: Accuracy, F1-score, and detailed classification reports
- **Visualization**: Training history plots and performance comparisons
- **Data Preprocessing**: Automatic label encoding and categorical conversion

##  Performance Results

Based on the training results:

| Model | Training Accuracy | Validation Accuracy | Test Accuracy |
|-------|------------------|-------------------|---------------|
| RNN   | 0.340           | 0.328             | 0.319         |
| LSTM  | 0.336           | 0.327             | 0.320         |
| BiLSTM| 0.420           | 0.407             | 0.403         |

**Key Findings:**
- **BiLSTM** performs best across all metrics
- **LSTM** shows similar performance to RNN
- **BiLSTM** achieves ~20% better accuracy than the other models

##  Training History Analysis

The project includes comprehensive visualization of training progress:

1. **Training vs Validation Accuracy**: Shows learning curves for all models
2. **Training vs Validation Loss**: Displays loss convergence patterns
3. **Final Performance Comparison**: Bar charts comparing final metrics
4. **Detailed Model Analysis**: Individual performance plots for each architecture

##  Installation

### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- scikit-learn
- sentence-transformers
- matplotlib
- seaborn

### Setup
```bash
# Install required packages
pip install sentence-transformers scikit-learn tensorflow matplotlib seaborn

# For Google Colab users
!pip install sentence-transformers scikit-learn tensorflow matplotlib seaborn --quiet
```

##  Project Structure

```
├── README.md                 # This file
├── untitled52.py            # Main implementation file
├── train.txt                # Training data (to be uploaded)
└── requirements.txt         # Dependencies (recommended)
```

##  Usage

### Data Preparation
1. Upload your training data file (`train.txt`)
2. Ensure the data has `text` and `label` columns
3. The script will automatically:
   - Convert text to sentence embeddings
   - Encode labels using LabelEncoder
   - Split data into training and test sets

### Model Training
```python
# The script automatically trains all models
for name in ['RNN', 'LSTM', 'BiLSTM']:
    model = models[name](input_shape, num_classes)
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
```

### Evaluation
The script provides comprehensive evaluation including:
- Accuracy and F1-score metrics
- Confusion matrices
- Detailed classification reports
- Training history visualizations

##  Visualization Features

### 1. Training History Plots
- Training vs validation accuracy curves
- Training vs validation loss curves
- Final validation accuracy comparison
- Test accuracy comparison

### 2. Detailed Model Analysis
- Individual performance plots for each model
- Side-by-side comparison of training progress
- Performance summary with all metrics

### 3. Performance Summary
- Bar charts comparing all metrics across models
- Training vs validation vs test accuracy comparison
- Loss comparison across models

##  Key Insights

1. **BiLSTM Superiority**: Bidirectional LSTM consistently outperforms unidirectional variants
2. **Convergence Patterns**: All models show stable convergence after 5-6 epochs
3. **Overfitting**: Minimal overfitting observed with proper validation splits
4. **Embedding Quality**: SBERT embeddings provide robust text representations

##  Model Comparison

| Aspect | RNN | LSTM | BiLSTM |
|--------|-----|------|--------|
| **Architecture** | Simple recurrent | Memory cells | Bidirectional memory |
| **Performance** | Baseline | Similar to RNN | Best |
| **Training Time** | Fastest | Medium | Slowest |
| **Memory Usage** | Lowest | Medium | Highest |
| **Gradient Flow** | Limited | Better | Best |

##  Limitations

- Requires significant computational resources for BiLSTM
- Training time increases with model complexity
- Performance depends heavily on data quality and size
- Sentence embeddings may not capture domain-specific nuances

## Future Improvements

1. **Hyperparameter Tuning**: Implement grid search for optimal parameters
2. **Attention Mechanisms**: Add attention layers for better performance
3. **Transfer Learning**: Fine-tune pre-trained language models
4. **Ensemble Methods**: Combine multiple models for better results
5. **Data Augmentation**: Implement text augmentation techniques



**Note**: This project was originally developed in Google Colab and may require adjustments for local execution. Ensure all dependencies are properly installed and data files are in the correct format.
