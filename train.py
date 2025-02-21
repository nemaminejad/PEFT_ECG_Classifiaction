

import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from utils import compute_metrics, compute_loss
from data_setup import *

from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import  with_parameters, with_resources, TuneConfig
from ray.tune.tuner import Tuner
from utils import create_search_space
from model_builder import *



def training(config, model, training_ds, validation_ds, batch_size, class_weight_dict, sample_pct):
  """
  Trains a LoRA model using the provided training and validation datasets.

  Steps:
  1. Loads the model, datasets, and optimizer, setting the device to GPU if available.
  2. Creates data loaders with batching and sampling.
  3. Restores model and optimizer state from a checkpoint if available.
  4. Iterates through epochs, performing:
    - Forward and backward passes on training data.
    - Loss computation, optimization, and logging every 10 batches.
  5. Evaluates the model on the validation set at the end of each epoch.
  6. Computes and logs evaluation metrics (loss, F1-score, precision, recall, accuracy).
  7. Reports final validation loss and saves training logs.

  Logs are recorded using TensorBoard for monitoring model performance.

  config: LoRA Config
  """

  lora_model = apply_lora_to_model(config, model)

  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  writer = SummaryWriter(f"runs at {timestamp}")
  best_eval_loss = float('inf')

  # set device
  device = torch.device("cuda") # if torch.cuda.is_available() else ("cpu")
  lora_model.to(device)
  optimizer = torch.optim.Adam(lora_model.parameters(), lr=config["lr"])
  # load data
  train_data_loader = create_data_loaders(training_ds, batch_size=batch_size, sample_pct=sample_pct, collate_fn=collate_fn)
  validation_data_loader = create_data_loaders(validation_ds, batch_size=batch_size, sample_pct=sample_pct, collate_fn=collate_fn)
  if hasattr(train, "get_checkpoint") and train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state, epoch_start = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
            lora_model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
  else:
      epoch_start = 0
  
  #start training: forward and backward passes.
  for epoch in range(epoch_start, config['epochs']):
        print(f"Epoch {epoch + 1}")
        lora_model.train()
        running_loss = 0

        for j, batch in enumerate(train_data_loader):
            optimizer.zero_grad()
            loss = compute_loss(lora_model,batch, class_weight_dict )
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            if j % 10 == 9:
                last_loss = running_loss / (j + 1)
                print(f"loss at batch {j + 1} = {last_loss}")
                tb_x = epoch * len(train_data_loader) + j + 1
                writer.add_scalar("loss", last_loss, tb_x)
                running_loss = 0


        lora_model.eval()
        running_eval_loss = 0
        running_eval_f1 = 0
        running_eval_auc = 0
        running_eval_precision = 0
        running_eval_recall = 0
        running_eval_accuracy = 0

        # to store training loss
        # epoch_train_loss = running_loss / len(train_data_loader)
        
        #validate performance
        for i, batch in enumerate(validation_data_loader):

            labels = batch["label"].to(device)
            output = lora_model(batch["pixel_values"].to(device))

            running_eval_loss += compute_loss(lora_model,batch)

            metric_dict = compute_metrics((output.logits , labels))
            f1 = metric_dict["f1-score"]
            precision = metric_dict["precision"]
            recall = metric_dict["recall"]
            accuracy = metric_dict["accuracy"]

            running_eval_f1 += f1
            running_eval_precision += precision
            running_eval_recall += recall
            running_eval_accuracy += accuracy
        # summarize valiation results
        avg_eval_loss = running_eval_loss / len(validation_data_loader)
        avg_eval_f1 = running_eval_f1 / len(validation_data_loader)
        avg_eval_precision = running_eval_precision / len(validation_data_loader)
        avg_eval_recall = running_eval_recall / len(validation_data_loader)
        avg_eval_accuracy = running_eval_accuracy / len(validation_data_loader)

        # to store validation loss
        # val_losses.append(avg_eval_loss)

        print(f"Avg validation loss ==>: {float(avg_eval_loss)}, F1 Score ==> {float(avg_eval_f1)}, Precision ==> {float(avg_eval_precision)}, Recall ==> {float(avg_eval_recall)}, Accuracy ===> {float(avg_eval_accuracy)}")
        writer.add_scalar("Eval Loss", avg_eval_loss, epoch + 1)
        writer.add_scalar("F1 Score", avg_eval_f1, epoch + 1)
        writer.add_scalar("Precision", avg_eval_precision, epoch + 1)
        writer.add_scalar("Recall", avg_eval_recall, epoch + 1)
        writer.add_scalar("Accuracy", avg_eval_accuracy, epoch + 1)
        writer.flush()

        train.report({"loss": avg_eval_loss.item()})

  print("Finished Training")
  return  


def hyperparameter_tuning(training_args, sample_pct=0.5, batch_size =  4, max_num_epochs=10, num_samples=5):
    """
    Main function to perform hyperparameter tuning and model training using Ray Tune.
    This function automates the search for optimal training parameters to improve model performance.
    Steps:
    1. Extracts model, training, and validation datasets from input arguments.
    2. Defines the hyperparameter search space.
    3. Configures the scheduler for early stopping.
    4. Initializes and runs the Ray Tune hyperparameter tuning process:
    - Allocates GPU resources.
    - Runs multiple training trials with different hyperparameter configurations.
    - Uses validation loss as the optimization metric.
    5. Identifies and prints the best trial's configuration and final validation loss.
    """
    model = training_args[0]
    train_data = training_args[1]
    val_data = training_args[2]
    class_weight_dict = training_args[3]
    
    config = create_search_space()
    
    scheduler = ASHAScheduler(
      max_t=max_num_epochs,
      grace_period=1,
      reduction_factor=2)
    print('Tuner setup')    
    tuner = Tuner(
      with_resources(
          tune.with_parameters(training, model=model, training_ds=train_data,
                          validation_ds=val_data, class_weight_dict = class_weight_dict,
                                sample_pct=sample_pct, batch_size=batch_size),
          resources={"GPU": 1}),
      tune_config=TuneConfig(
          metric="loss",
          mode="min",
          scheduler=scheduler,
          num_samples=num_samples,
      ),
      param_space=config,

    )
    results = tuner.fit()
    
    best_result = results.get_best_result("loss", "min", filter_nan_and_inf=True)
    
    
    return best_result
      