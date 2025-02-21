# PEFT_ECG_Classifiaction
Training an ECG classification model using Paramet-Efficient-Fine-Tuning approach

## ECG Image Classification with QLoRA and Hyperparameter Tuning
This repository contains Python scripts for performing parameter-efficient fine-tuning of a self-supervised learning (SSL) model (DinoV2) using the QLoRA technique for classification of ECG signal images. It leverages Hugging Face libraries and includes hyperparameter tuning using Ray Tune

### Overview
The code in this repository performs the following:

**Data Loading and Preprocessing**: Loads ECG image data from specified directories, applies transformations, and prepares it for model training.

**Model Setup**: Initializes a quantized image classification model (DinoV2) with LoRA configuration for efficient training.

**Training**: Fine-tunes the model using the QLoRA technique, incorporating class weights to address data imbalance.

**Hyperparameter Tuning**: Uses Ray Tune to optimize hyperparameters such as learning rate, LoRA rank, and dropout.

### File Descriptions
**run_hyperparameter_tuning.py**: Main script to run hyperparameter tuning. It handles data loading, model setup, and calls the hyperparameter tuning function.

**preprocess_ecg.py**: preprocess ECG signal (adapted from [ECGCAD repo]([https://pages.github.com/](https://github.com/MediaBrain-SJTU/ECGAD/tree/main))

**create_ecg_plots.py**: create plots and images of multiple ECG signals from multiple leads. Instead of training on ECG signals, we are training on images the ECG signals. This function creates the training and validationser set.

**data_setup.py**: Contains functions for loading image data, creating data loaders, and applying transformations for data augmentation and preprocessing.

**model_builder.py**: Includes functions for setting up the quantized image classification model (DinoV2) with LoRA configuration.

**train.py**: Contains functions for training the LoRA model and performing hyperparameter tuning using Ray Tune.

**utils.py**: Provides utility functions for calculating class weights, computing evaluation metrics (accuracy, precision, recall, F1 score), and creating the hyperparameter search space

### Usage
1. Clone the repository.
2. Install the required dependencies in requirements.txt.
3. Prepare the ECG image data: Organize the data into training and validation directories.
4. Configure the training: Modify the train_dir and val_dir variables in run.py to point to the correct data directories.
5. Run the hyperparameter tuning: to start the hyperparameter tuning process using Ray Tune

### Quick start
```
python run_hyperparameter_tuning.py # to run hyperparameter tuning
```
### Data
Data used for training of the model is orignally acquired from [Physionet](https://physionet.org/content/ptb-xl/1.0.3/).
Preprocessed image data from ECG signals will be available soon. 

### Notes
The code supports quantization using 4-bit precision to reduce memory footprint and improve training efficiency.

Class weights are calculated to handle imbalanced datasets, ensuring that underrepresented classes are given more importance during training.

The classifier of DinoV2 can be switched with any other model.

The faster_filter function is used to load a small portion of the data for testing the code
