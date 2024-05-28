from transformers import Trainer, TrainingArguments

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


