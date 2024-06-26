# Image-Caption-Generator

This repository contains code for an Image Captioning model using the Flickr8k dataset. The project utilizes deep learning techniques with TensorFlow and Keras to generate captions for images based on their content. Note: This project is ongoing and is expected to be completed by November 2024.

Dataset
The Flickr8k dataset consists of 8,000 images, each with multiple human-annotated captions. These images cover a wide range of scenes and objects, making it suitable for training and evaluating image captioning models.

Directory Structure
- Images/: Directory containing the Flickr8k dataset images.
- captions.txt: Text file containing image IDs and corresponding captions.
- features.pkl: Pickle file to store extracted image features.
- best_model.h5: Trained model saved in HDF5 format.

**Getting Started**

To run the project, follow these steps:

Prerequisites
- Python 3.x
- TensorFlow
- Keras
- numpy
- tqdm
- Installation

**Usage**

Data Preparation:

- Ensure the Flickr8k dataset is downloaded and placed in the Images/ directory.
Update BASE_DIR and IMAGES_DIR paths in the code if necessary.

Feature Extraction:

- Run extract_features.py to extract features from images using VGG16.
Extracted features are stored in features.pkl.

Caption Preprocessing:

- Run preprocess_captions.py to clean and tokenize captions.
Tokenizer and cleaned captions are saved for training.

Model Training:

- Run train.py to define, compile, and train the image captioning model.
Adjust hyperparameters (epochs, batch size) as needed.

Inference:

- After training, use generate_caption.py to generate captions for new images.
Provide path to the image as command line argument.

**Files**
- extract_features.py: Script to extract image features using VGG16.
- preprocess_captions.py: Script to preprocess captions (cleaning and tokenization).
- train.py: Script to define, compile, and train the image captioning model.
- generate_caption.py: Script to generate captions for new images after training.

**Model Architecture**
The model architecture consists of an encoder-decoder structure:

- Encoder: VGG16 pretrained on ImageNet for feature extraction.
- Decoder: LSTM network with attention mechanism to generate captions.

**Results**
The model achieves competitive performance in generating accurate captions for images in the Flickr8k dataset.
Evaluation metrics (BLEU scores) and qualitative assessments (caption examples) can be included here.
