# ner_pipeline

This project includes code for training a Named Entity Recognition (NER) model and an API for making NER predictions on text data. The project is organized into three main components: data preprocessing, model training, and prediction.

## Components

1. **Data Preprocessing (`data_pipeline.py`):**
   - The `data_pipeline.py` module provides functions for reading data from a JSON file, extracting text and labels, and preparing the data for training.

2. **Model Training (`training.py`):**
   - The `training.py` module contains functions for tokenizing the input data, adjusting labels for tokenization, and training a NER model using the Transformers library, specifically DistilBERT (`distilbert-base-nli-stsb-mean-tokens`).

3. **Prediction (`prediction.py`):**
   - The `prediction.py` module includes functions for loading a pre-trained NER model, predicting NER tags for input text, and post-processing the results.

4. **API (`main.py`):**
   - The `main.py` script sets up a FastAPI server to expose endpoints for training the NER model (`/ner_train`) and making NER predictions (`/ner_prediction`).
## How to Use

### Data Annotation with Doccano

Before training the NER model, you may need to annotate your training data using a tool like Doccano. Follow these steps:

1. Install Doccano and set it up to annotate text data.
2. Import your text data into Doccano and label named entities using the provided labels (e.g., PERSON, ORGANIZATION, etc.).
3. Export the annotated data from Doccano in JSON format.

### Training the NER Model
Send a POST request to `/ner_train` with the following JSON payload:
`json`
{
  "data_path": "path_to_annotated_data.json",
  "model_path": "path_to_pretrained_model",
  "n_epoch": 10
}

Replace path_to_annotated_data.json with the path to your annotated training data in JSON format, which includes the labeled entities.
Replace path_to_pretrained_model with the path to a pre-trained DistilBERT model (distilbert-base-nli-stsb-mean-tokens).

### Making NER Predictions
Send a POST request to `/ner_prediction` with the following JSON payload:
`json`
{
  "data_path": "path_to_annotated_data.json",
  "model_path": "path_to_pretrained_model",
  "fine_tune_model_path": "path_to_fine_tuned_model",
  "text": "text_to_predict_NER_tags"
}

Replace path_to_annotated_data.json with the path to your annotated data in JSON format.
Replace path_to_pretrained_model with the path to a pre-trained DistilBERT model (distilbert-base-nli-stsb-mean-tokens).
Replace path_to_fine_tuned_model with the path to a fine-tuned NER model.
Replace text_to_predict_NER_tags with the text for which you want to predict NER tags.

### Installation
1. Clone the repository:
git clone https://https://github.com/Rehansolanki7/ner_pipeline

2. Install dependencies:
pip install -r requirements.txt

4. Start the FastAPI server:
uvicorn main:app --reload
