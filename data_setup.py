
import os
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

def load_data(train_dir,val_dir):
  data_files = {}
  data_files["train"] = os.path.join(train_dir, "**")
  if val_dir:
    data_files["validation"] = os.path.join(val_dir, "**")
  # data_files["test"] = os.path.join(test_dir, "**")
  dataset = load_dataset(
              "imagefolder",
              data_files= data_files
          )

  labels = dataset["train"].features['label'].names
  label2id, id2label = dict(), dict()
  for i, label in enumerate(labels):
      label2id[label] = i
      id2label[i] = label

  return dataset,label2id, id2label


def create_data_loaders(dataset, batch_size, sample_pct, collate_fn):
    subset_size = int(len(dataset) * sample_pct)
    indices = torch.randperm(len(dataset))[:subset_size]
    subset = Subset(dataset, indices)
    data_loader = DataLoader(subset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
    return data_loader

def collate_fn(examples):
    """
    collate_fn() will be used to batch examples together.
    Each batch consists of 2 keys, namely pixel_values and labels.
    """
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "label": labels}


def get_transforms(image_processor):
  """
  Defines and returns transform object for data augmentation and preprocessing of training and validation.

  - Normalizes images and applies resizing, cropping and flipping.
  - Converts images to tensors and normalizes them.
  """
  normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
  if "height" in image_processor.size:
      size = (image_processor.size["height"], image_processor.size["width"])
      crop_size = size
      max_size = None
  elif "shortest_edge" in image_processor.size:
      size = image_processor.size["shortest_edge"]
      crop_size = (size, size)
      max_size = image_processor.size.get("longest_edge")

  train_transforms = Compose(
          [
              RandomResizedCrop(crop_size),
              RandomHorizontalFlip(),
              ToTensor(),
              normalize,
          ]
      )

  val_transforms = Compose(
          [
              Resize(size),
              CenterCrop(crop_size),
              ToTensor(),
              normalize,
          ]
      )
  return train_transforms, val_transforms


def preprocess_train(example_batch,train_transforms):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch


def preprocess_val(example_batch, val_transforms):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch


