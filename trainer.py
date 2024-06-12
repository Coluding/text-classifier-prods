from transformers import Trainer, TrainingArguments
import torch
import logging
import torch.nn as nn
import numpy as np
import os
import gc
from tqdm.auto import tqdm
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# Define the training arguments
args = {
    "output_dir": "./results",
    "evaluation_strategy": "epoch",
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "num_train_epochs": 3,
    "weight_decay": 0.01,
}


training_args = TrainingArguments(
    **args
)

logging.basicConfig(level=logging.INFO)


### Custom fit function
def fit(save_dir: str, run_name: str, model: nn.Module, epochs: int, eval_steps: int,
        train_loader: torch.utils.data.dataloader.DataLoader,
        val_loader: torch.utils.data.dataloader.DataLoader, optimizer: torch.optim, learning_rate: float,
        lr_scheduler: torch.optim.lr_scheduler, **kwargs) -> list:
    """
      Fits the model with defined optimizer, learning rate scheduler, early stopper  and learning rate

      :param model: PyTorch model which should be trained, possibly on gpu
      :param epochs: Number of epochs
      :param train_loader: DataLoader for training data, possibly on gpu
      :param val_loader: DataLoader for validation data, possible on gpu
      :param optimizer: PyTorch optimizer to optimize the loss
      :param learning_rate: Learning rate for the optimizer
      :param early_stopping: True if early stopping should be checked
      :param lr_scheduler: PyTorch learning rate scheduler to adjust the learning rate
      :param kwargs: additional keyword arguments for the learning rate scheduler
      :return: history of training
      :rtype: list
  """

    hparam_dict = {
        "optimizer": optimizer.__name__,
        "learning_rate": learning_rate,
        "lr_scheduler": lr_scheduler.__name__,
        "epochs": epochs,
        "eval_steps": eval_steps,
        "batch_size": train_loader.batch_size,
        **kwargs
    }

    log_dir = f"{save_dir}/{run_name}"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Save the hyperparameters
    with open(f"{log_dir}/hparams.json", "w") as f:
        json.dump(hparam_dict, f)

    # Initialize the optimizer
    optimizer = optimizer(model.parameters(), lr=learning_rate)

    # Initialize the learning rate scheduler
    scheduler = lr_scheduler(optimizer, **kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    history = []
    best_val_loss = np.inf

    training_steps = len(train_loader) * epochs
    iterator = tqdm(range(training_steps), desc="Training")
    global_step = 0

    def eval(best_val_loss):
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        iterator = tqdm(val_loader, desc="Evaluating")
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
                total_loss += loss.item()
                all_preds.extend(outputs[1].argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                iterator.update(1)
                #iterator.refresh()
                del input_ids, attention_mask, labels, outputs
                torch.cuda.empty_cache()
                gc.collect()

        val_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        print(f"Step {global_step} / {len(iterator)}, Val Loss: {val_loss}, "
              f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
        history.append({"step": global_step, "val_loss": val_loss, "accuracy": accuracy,
                        "precision": precision, "recall": recall, "f1": f1})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{log_dir}/best_model.pt")
            print(f"New best model saved with val loss: {val_loss}")

        return best_val_loss

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            iterator.update(1)
            iterator.refresh()
            if global_step % eval_steps == 0:
                best_val_loss = eval(best_val_loss)
            del input_ids, attention_mask, labels, outputs
            torch.cuda.empty_cache()
            gc.collect()

        train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}, Train Loss: {train_loss}")
        history.append({"step": global_step, "train_loss": train_loss})
        best_val_loss = eval(best_val_loss)
    return history