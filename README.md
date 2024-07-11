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

Training the Model

To train the model, run:
```bash
python llm_embed.py --exec learn_only
```

Making Predictions

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

Requirements
	•	Python3.7+
	•	pandas
	•	numpy
	•	torch
	•	scikit-learn
	•	matplotlib


 ## Files

Files
	•	llm_embed.py: Main script for training and prediction.
	•	dataset/dataset_variousword.csv: CSV file containing word embeddings.
	•	dataset/dataset_variouspred.csv: CSV file containing prediction embeddings.
	•	models.py: Contains the linear regression model class.
	•	test_llm_embed.py: Test script for the linear regression model.
