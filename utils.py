import torch
from torch.utils.data import DataLoader, Subset
from ray import tune
import numpy as np
import random
import os
import evaluate
  
# calculate weights for samples to take into account class imbalance
def calculate_class_weight(train_dir):
    """
    Calculate weights for samples to handle imbalanced dataset.
    Assigning higher weights to underrepresented classes and lower weights to overrepresented classes.
    Classes with fewer samples receive higher weights and vice versa.
    """
    frequencies_dict ={}
    for root, dir, _ in os.walk(train_dir):
        for d in dir:
          frequencies_dict[d] = len(os.listdir(os.path.join(root, d)))
    
    weights = {}
    total_samples = sum(frequencies_dict.values())
    num_classes = len(frequencies_dict.keys())
    for k in frequencies_dict.keys():
        weights[k] = total_samples / (num_classes * frequencies_dict[k]) #classes with fewer samples receive higher weights and vice versa.

    return weights
  
  
  
def print_trainable_parameters(model):
    """
    Helper function that calculates and prints the number of trainable parameters and total parameters in a given model
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
      all_param += param.numel()
      if param.requires_grad:
          trainable_params += param.numel()
    print(
      f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


def compute_metrics(eval_pred):
    """Computes accuracy, precision, recall and F1 score on a batch of predictions"""

    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric= evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    logits, labels = eval_pred # eval_pred is a tuple consisted of predictions and labels that are returned by the model
    predictions = np.argmax(logits.detach().cpu().numpy(), axis=-1)

    precision = precision_metric.compute(predictions=predictions, references=labels, average='weighted')["precision"]
    recall = recall_metric.compute(predictions=predictions, references=labels, average='weighted')["recall"]
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='weighted')["f1"]
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]

    # The trainer is expecting a dictionary where the keys are the metrics names and the values are the scores.
    return {"precision": precision, "recall": recall, "f1-score": f1, 'accuracy': accuracy}

def compute_loss(model, inputs,class_weight_dict = None,return_outputs=False, num_items_in_batch=None):
    device = model.device
    # device = torch.device("cuda")
    labels = inputs["label"].to(device)
    pixel_values = inputs["pixel_values"].to(device)
    
    # Get model's prediction
    outputs = model(pixel_values)
    logits = outputs.get("logits").to(device)

  # Compute custom loss accroding
    if class_weight_dict:
    # for training consider class_weights to account for the imbalance in the data
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor(list(class_weight_dict.values()), device=device, dtype=logits.dtype))
    else:
        loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
    return (loss, outputs) if return_outputs else loss

def create_search_space():
    return {
        "lr": tune.loguniform(1e-4, 1e-1),
        "r": tune.choice([2, 4, 6, 8, 10, 16]),
        "target_modules": tune.choice([["query"], ["value"], ["query", "value"]]),
        "lora_dropout": tune.uniform(0.1, 0.5),
        'epochs': tune.choice([3,])
    }
    

def faster_filter(dataset, fraction=0.05):
    """Selects a fraction of samples from a dataset using random indices.
    
    Args:
    dataset: The input dataset.
    fraction: The fraction of samples to select.
    
    Returns:
    A subset of the input dataset containing the selected samples.
    """
    num_samples = int(len(dataset) * fraction)
    random_indices = random.sample(range(len(dataset)), num_samples)
    subset = dataset.select(random_indices)
    return subset
  
  
def faster_filter(dataset, fraction=0.05):
  """Selects a fraction of samples from a dataset using random indices.

  Args:
    dataset: The input dataset.
    fraction: The fraction of samples to select.

  Returns:
    A subset of the input dataset containing the selected samples.
  """
  num_samples = int(len(dataset) * fraction)
  random_indices = random.sample(range(len(dataset)), num_samples)
  subset = dataset.select(random_indices)
  return subset