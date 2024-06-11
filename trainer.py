from transformers import Trainer, TrainingArguments
import torch
import logging
import torch.nn as nn
import numpy as np
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
def fit(run_name: str, model: nn.Module, epochs: int, train_loader: torch.utils.data.dataloader.DataLoader,
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

    # Initialize the optimizer
    optimizer = optimizer(model.parameters(), lr=learning_rate)

    # Initialize the learning rate scheduler
    scheduler = lr_scheduler(optimizer, **kwargs)

    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
        train_loss = total_loss / len(train_loader)

        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                all_preds.extend(outputs.logits.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1})
