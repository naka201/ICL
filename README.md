# ICL Project

This project performs linear regression using word embeddings for predicting relationships between words.

## Overview

This project includes:
- Loading word embedding data
- Training a linear regression model
- Predicting relationships and similarities between words

## Installation

To install the required dependencies, run:
```bash
pip install -r requirements.txt

```

## Usage

Creating DataFrames with Embeddings

To create DataFrames with word embeddings, use dataframe_openai.py
```bash
python dataframe_openai.py
```

Training the Model

・data1 in llm_embed.py is training data

To train the model, run:
```bash
python llm_embed.py --exec learn_only
```

Making Predictions

・data2 in llm_embed.py is test data

To make predictions, run:
```bash
python llm_embed.py --exec pred_only
```

Full Execution(Training and Prediction)

To train the model and make predictions, run:
```bash
python llm_embed.py --exec all
```

## Requirements

This project requires:
- Python 3.7+
- pandas
- numpy
- torch
- scikit-learn
- matplotlib


 ## Files

This project includes:
- llm_embed.py: Main script for training and prediction.
- dataframe_openai.py : Convert words to word embeddings and make dataframe.
- data : Containing various words for training and prediction.
- dataset/dataset_variousword.csv: CSV file containing word embedding for training.
- dataset/dataset_variouspred.csv: CSV file containing word for predicting .
- skipgram.py: For comparison with llm_embed.py.
