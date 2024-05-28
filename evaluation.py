from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from model import TextClassifier
from safetensors.torch import load_file
from data import setup_datasets

import warnings
warnings.filterwarnings('always')


def load_safetensors_weights(model, file_path):
    # Load the safetensors file
    state_dict = load_file(file_path)
    # Load the state_dict into the model
    model.load_state_dict(state_dict, strict=False)


def load_model_and_tokenizer(model_name, checkpoint_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TextClassifier.from_pretrained(checkpoint_path,num_labels=607)
    return model, tokenizer


def predict(sentences, model, tokenizer, reverse_label_mapping, device="cpu"):
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)
    model = model.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    outputs = outputs[0].cpu()
    predictions = np.argmax(outputs.numpy(), axis=1)
    predicted_labels = [reverse_label_mapping[p] for p in predictions]
    return predicted_labels, predictions


def compute_metrics(predictions, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def evaluate_model(model, tokenizer, dataset, batch_size, reverse_label_mapping, device="cpu"):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    model = model.to(device)
    total_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    num_batches = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        sentences = batch['text']
        true_labels = batch['label']
        predicted_labels, predictions = predict(sentences, model, tokenizer, reverse_label_mapping, device)
        metrics = compute_metrics(predictions, true_labels)

        for key in total_metrics.keys():
            total_metrics[key] += metrics[key]

        num_batches += 1

    # Calculate average metrics
    avg_metrics = {key: total / num_batches for key, total in total_metrics.items()}

    return avg_metrics


def main():
    warnings.filterwarnings('ignore')
    model_name = "distilbert/distilbert-base-german-cased"
    checkpoint_path = "./results_final/checkpoint-55000"

    train_dataset, test_dataset, label_mapping = setup_datasets("training_data.csv", seed=4, split_size=0.8)
    model, tokenizer = load_model_and_tokenizer(model_name, checkpoint_path)
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}

    random_indices = np.random.choice(len(test_dataset), 150)
    sentences = [test_dataset['text'][i] for i in random_indices]
    true_labels = [test_dataset['label'][i] for i in random_indices]
    true_mapped_labels = [reverse_label_mapping[l] for l in true_labels]

    predicted_labels, predictions = predict(sentences, model, tokenizer, reverse_label_mapping, "cuda")
    metrics = compute_metrics(predictions, true_labels)

    for sentence, true_label, predicted_label in zip(sentences, true_mapped_labels, predicted_labels):
        print(f"Produktname: {sentence}")
        print(f"Korrekte Produktgruppe: {true_label}")
        print(f"Vorhergesagte Produktgruppe: {predicted_label}")
        print("------------------------------------------------------------------------------------")

    print(f"Metrics: {metrics}")

    avg_metrics = evaluate_model(model, tokenizer, test_dataset, 128, reverse_label_mapping, "cuda")

    print(f"Average metrics: {avg_metrics}")

if __name__ == "__main__":
    main()