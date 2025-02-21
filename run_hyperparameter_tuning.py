"""
The script run.py performs the following:
 1- identify input dir for image data
 2 - calls data setup, loading, and transforming functions
 3- calls model setup functions
 4- creates ray object for hyperparameter tuning
 5- runs hyperparameter tuning
"""
import os
import ray
from data_setup import get_transforms, load_data, preprocess_train,preprocess_val
from model_builder import setup_model, setup_tokenizer
from train import hyperparameter_tuning
from utils import calculate_class_weight, faster_filter

output_dir = './VIT/models'
os.makedirs(output_dir,exist_ok = True)
# identify input directory
train_dir = '/content/gdrive/MyDrive/ECG_colab/data/PTB-XL/physionet.org/files/ptb-xl/1.0.3/PTB-XL-img/train'#'/content/gdrive/MyDrive/ECG_colab/data/hf_format_data/'
val_dir = '/content/gdrive/MyDrive/ECG_colab/data/PTB-XL/physionet.org/files/ptb-xl/1.0.3/PTB-XL-img/val'
batch_size = 4
model_checkpoint = 'facebook/dinov2-base'
# other options for model_checkpoint: 'google/vit-base-patch16-224' # 'facebook/dinov2-base-imagenet1k-1-layer'#'google/vit-base-patch16-224' #'facebook/dinov2-base'

# load data and get data-label mappings
print('Loading data')
raw_dataset,label2id, id2label = load_data(train_dir,val_dir)

#setup model and tokenizer
model  = setup_model(model_checkpoint, label2id, id2label,batch_size)
image_processor = setup_tokenizer(model_checkpoint)



#get data
#load small portion of data for testing code
train_ds = faster_filter(raw_dataset["train"], fraction=0.01)
val_ds = faster_filter(raw_dataset["validation"], fraction=0.01)


## apply transformations

train_transform, val_transform = get_transforms(image_processor)
train_ds.set_transform(lambda batch: preprocess_train(batch, train_transform))
val_ds.set_transform(lambda batch: preprocess_val(batch,val_transform))
print('Loaded data and transformations')

# #setup ray parameters
ray_model= ray.put(model)
ray_train_data = ray.put(train_ds)
ray_validation_data = ray.put(val_ds)
# get class weights
class_weight_dict = calculate_class_weight(train_dir)
# run hyperparameter tuning
print('Start hyperparameter tuning with Ray')
best_result = hyperparameter_tuning((ray_model, ray_train_data, ray_validation_data, class_weight_dict))
# return best reuslt from hyperparameter tuning
print("Best trial config: {}".format(best_result.config))
print("Best trial final validation loss: {}".format(
best_result.metrics["loss"]))
return best_result