
# BERT Semantic Similarity on SNLI Dataset

This repository hosts a PyTorch implementation designed for evaluating semantic similarity between sentence pairs using the Stanford Natural Language Inference (SNLI) dataset with a BERT model. The SNLI dataset comprises 570k human-written English sentence pairs, each manually labeled as entailment, contradiction, or neutral, providing a robust foundation for developing models capable of understanding nuanced textual relationships.

## About the SNLI Dataset

The SNLI dataset is essential for training models on natural language inference (NLI), which closely aligns with semantic similarity tasks. Each sentence pair in the dataset is annotated with labels that indicate whether the hypothesis sentence is an entailment, contradiction, or bears no relation (neutral) to the premise sentence. This setup provides a valuable framework for training models to discern semantic similarity within diverse textual contexts.

## Features

- Utilization of the SNLI dataset from Hugging Face, including predefined splits for training, validation, and testing.
- Adapting BERT for semantic similarity, leveraging sequence classification capabilities.
- Visualizing training progress through loss metrics.
- Saving the trained model for future inference or deployment in semantic similarity tasks.

## Installation

Ensure Python 3.6 or newer is installed on your system. To install the required packages, run:

```bash
pip install torch transformers tqdm matplotlib
```

## Usage

1. **Dataset Preparation**: The SNLI dataset is automatically fetched from Hugging Face, complete with training, validation, and testing splits, streamlining the setup process for immediate use.

2. **Model Training**:

   Start the training process with:

   ```bash
   python train.py
   ```

   The script trains the BERT model using the SNLI dataset, specifically focusing on learning representations that reflect semantic similarity between sentence pairs.

3. **Outputs**: The model `Bert_ft.pt` and a training loss graph `Training_Loss.png` are saved in the `results` directory, encapsulating the training outcome and performance.

## Customization

Experiment with different configurations, such as altering epochs, batch size, or learning rate adjustments, to explore how these variations impact the model's understanding of semantic similarity.

---

This version focuses directly on the project's essence, providing users with a concise guide on its purpose, usage, and customization options.

## Brief explanation of Scripts

### 1. `train.py`

This script is the main entry point for training a BERT model on the semantic similarity task using the SNLI dataset. It performs several key functions:

- **Data Preparation**: It calls a function `preprocess()` (expected to be defined in `preprocess.py`) to load and preprocess the SNLI dataset into training, validation, and testing sets.
- **Model Initialization**: Initializes a BERT model specifically for sequence classification with a predefined number of labels (three, in this case, corresponding to entailment, contradiction, and neutral).
- **Training Loop**: Implements the training loop, where the model is trained on the dataset using a DataLoader. It also calculates and stores the training loss after each epoch.
- **Loss Visualization**: Calls the `plot()` function to visualize the training loss over epochs and saves the plot to the `results` folder.
- **Model Saving**: Saves the trained model to disk for future use.

### 2. `preprocess.py`

This script is responsible for loading and preprocessing the SNLI dataset. It outlines several steps:

- **Dataset Loading**: Utilizes Hugging Face's `datasets` library to load the SNLI dataset.
- **Data Cleaning**: Removes rows from the dataset where the label is undefined or not applicable (-1).
- **Data Preparation**: Converts the dataset into lists of premises, hypotheses, and labels, and returns these lists for further processing.

The `preprocess` function is designed to be flexible enough to handle different splits of the data (train, validation, test) by accepting a `data_split` argument.

### 3. `TextDataset.py`

This file defines a custom PyTorch `Dataset` class, `TextDataset`, tailored for tokenizing text data for use with BERT models. It includes important functionality:

- **Initialization**: Takes lists of texts and labels as input, along with a maximum sequence length, and initializes a BERT tokenizer.
- **Length**: Provides the number of items in the dataset.
- **Get Item**: Tokenizes a specific item (text) from the dataset, preparing it for input to BERT. This includes generating input IDs, attention masks, and token type IDs, along with the corresponding label for the text.

The `TextDataset` class is crucial for efficiently managing and preparing the data for training and evaluation within the PyTorch framework.

Each script is designed to fulfill specific roles within the training pipeline, ensuring modular and maintainable code structure for the semantic similarity task using BERT and the SNLI dataset.

## Dealing with similarity as a classification Problem

Classifying a pair of sentences as similar or not similar using BERT involves a few steps, focusing on how to effectively represent the sentence pair and then classify them based on their similarity. Here's a high-level approach to tackle this problem:

**Preparing the Input**

1. **Tokenization**: First, tokenize both sentences using BERT's tokenizer. This involves converting each sentence into a series of tokens that BERT has been trained on.
2. **Special Tokens**: For BERT to understand that it's dealing with a pair of sentences and to maintain the context, you need to add special tokens:
    - `[CLS]`: At the very beginning of the token list.
    - `[SEP]`: To separate the two sentences and at the end of the second sentence.
    
    So, the input format looks like `[CLS] sentence 1 [SEP] sentence 2 [SEP]`.
    
3. **Attention Mask**: Since BERT inputs require a fixed length, you might need to pad shorter inputs. The attention mask allows BERT to differentiate between real tokens and padding tokens.

Processing with BERT

1. **Feeding to BERT**: Pass the prepared input through BERT. BERT will output an embedding for each token, including the special `[CLS]` token.
2. **Using the Output**: For similarity tasks, you can use the `[CLS]` token's embedding, the pooled output, as it is designed to represent the entire input's context. Alternatively, you could experiment with other strategies like averaging the embeddings of all tokens (mean pooling) to represent the sentence pair.

Classification Layer

1. **Similarity Score**: Add a classification layer on top of BERT's output. This can be a simple dense layer with a softmax activation to classify the sentence pair as similar or not similar. The choice of loss function can vary, but a common choice is binary cross-entropy for this binary classification task.
2. **Fine-tuning**: It's crucial to fine-tune BERT along with your classification layer on a dataset of sentence pairs labeled for similarity. This fine-tuning step adjusts the pre-trained BERT weights to perform better on your specific similarity classification task.

Example Workflow

- **Step 1**: Tokenize the sentence pairs with the special tokens.
- **Step 2**: Pass the tokenized pairs through BERT.
- **Step 3**: Take the `[CLS]` token's output (or apply another strategy like mean pooling across the tokens) to get a fixed-size sentence pair representation.
- **Step 4**: Pass this representation through a classification layer to predict similarity.
- **Step 5**: Train this model on a labeled dataset of sentence pairs, adjusting the BERT weights and the classification layer to optimize performance on the similarity task.

This approach allows leveraging BERT's powerful contextual embeddings to understand the nuanced similarity between sentence pairs, making it effective for tasks like semantic similarity, paraphrase detection, or other related natural language understanding tasks.
