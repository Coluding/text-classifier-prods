from transformers import Trainer, AutoTokenizer, TrainerCallback, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from trainer import fit
import torch
import tqdm
import json
from model import TextClassifier
from data import setup_datasets, CustomCollator



def custom_train():
    model_name = "distilbert/distilbert-base-german-cased"
    train_dataset, test_dataset, label_mapping = setup_datasets("training_data_new_wg.csv",
                                                                minimum_num_instances=1, delim="#",
                                                                remove_label="Medizinprodukt")
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}
    model = TextClassifier.from_pretrained(model_name, num_labels=len(label_mapping))
    model = model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    collator = CustomCollator(tokenizer, for_dataloader=False)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collator)

    optimizer = torch.optim.Adam
    lr_scheduler = torch.optim.lr_scheduler.StepLR

    history = fit("results_custom",
                  "old_data", model, 5000, 1000,
                  train_loader, val_loader, optimizer, 2e-5, lr_scheduler,
        step_size=1, gamma=0.1)

    # save history
    with open("results_custom/history.json", "w") as f:
        json.dump(history, f)


def main():
    CUSTOM = True

    if CUSTOM:
        custom_train()
        return

    else:
        model_name = "distilbert/distilbert-base-german-cased"
        train_dataset, test_dataset, label_mapping =setup_datasets("training_data_new_wg.csv",
                                                                minimum_num_instances=1, delim="#",
                                                                remove_label="Medizinprodukt")

        reverse_label_mapping = {v: k for k, v in label_mapping.items()}

        model = TextClassifier.from_pretrained(model_name, num_labels=len(label_mapping))
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        collator = CustomCollator(tokenizer, for_dataloader=False)

        args = {
            "output_dir": "./results",
            "learning_rate": 3e-5,
            "per_device_train_batch_size": 64,
            "per_device_eval_batch_size": 64,
            "num_train_epochs": 1000,
            "weight_decay": 0.01,
            "remove_unused_columns": False,
            "save_steps": 5000,
            #"fp16": True,

        }

        training_args = TrainingArguments(
            **args
        )

        class SaveCallback(TrainerCallback):
            def __init__(self, save_steps):
                super().__init__()
                self.save_steps = save_steps

            def on_step_end(self, args, state, control, model=None, **kwargs):
                if state.global_step % self.save_steps == 0 and state.global_step > 0:
                    model.save_pretrained(f"{args.output_dir}/checkpoint-{state.global_step}")
                    model.config.save_pretrained(f"{args.output_dir}/checkpoint-{state.global_step}")

        save_callback = SaveCallback(save_steps=5000)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=collator,
            tokenizer=tokenizer,
            #callbacks=[save_callback],
            #compute_metrics=compute_metrics
        )

        #model.config.to_json_file("./results/config.json")

        trainer.train()


if __name__ == "__main__":
    main()
