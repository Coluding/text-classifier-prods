from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load the custom TextClassifier, safetensors, and data setup functions
from model import TextClassifier
from safetensors.torch import load_file
from data import setup_datasets

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# Function to load safetensors weights
def load_safetensors_weights(model, file_path):
    state_dict = load_file(file_path)
    model.load_state_dict(state_dict, strict=False)


# Function to load the model and tokenizer
def load_model_and_tokenizer(model_name, checkpoint_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TextClassifier.from_pretrained(checkpoint_path, num_labels=607)
    return model, tokenizer


# Function to predict labels
def predict(sentences, model, tokenizer, reverse_label_mapping, device="cpu"):
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)
    model = model.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    outputs = outputs[0].cpu()
    predictions = np.argmax(outputs.numpy(), axis=1)
    predicted_labels = [reverse_label_mapping[p] for p in predictions]
    return predicted_labels, predictions


# Function to compute metrics
def compute_metrics(predictions, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


# Function to evaluate the model
def evaluate_model(model, tokenizer, dataset, num_samples, reverse_label_mapping, device="cpu"):
    random_indices = np.random.choice(len(dataset), num_samples)
    sentences = [dataset['text'][i] for i in random_indices]
    true_labels = [dataset['label'][i] for i in random_indices]
    predicted_labels, predictions = predict(sentences, model, tokenizer, reverse_label_mapping, device)
    metrics = compute_metrics(predictions, true_labels)

    evaluation_details = []
    for sentence, true_label, predicted_label in zip(sentences, true_labels, predicted_labels):
        evaluation_details.append({
            "sentence": sentence,
            "true_label": reverse_label_mapping[true_label],
            "predicted_label": predicted_label
        })

    return metrics, evaluation_details


# Load the model and tokenizer
model_name = "distilbert/distilbert-base-german-cased"
checkpoint_path = "./results_final/checkpoint-55000"
model, tokenizer = load_model_and_tokenizer(model_name, checkpoint_path)

# Load datasets to get label mappings
_, test_dataset, label_mapping = setup_datasets("training_data.csv", seed=4, split_size=0.8)
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Load labels from the file
with open("labels_above_100.txt", "r") as f:
    labels = f.read().splitlines()

# Define the layout of the dashboard
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Product Label Prediction Dashboard"),
            html.Label("Trainierte Labels:"),
            dcc.Dropdown(
                id="label-dropdown",
                options=[{"label": label, "value": label} for label in labels],
                multi=True,
                placeholder="Suche Labels"
            ),
            html.Hr()
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col([
            html.H1("Produktgruppen Vorhersage"),
            html.Label("Produktname eingeben:"),
            dcc.Input(id="product-name-input", type="text", style={"width": "100%"}),
            html.Button("Vorhersage", id="predict-button", n_clicks=0, style={"margin-top": "10px"}),
            dbc.Spinner(html.Div(id="prediction-output", style={"margin-top": "20px"}))
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col([
            html.H2("Modell Evaluation"),
            html.Label("Anzahl der Samples:"),
            dcc.Input(id="num-samples-input", type="number", value=10, style={"width": "100%"}),
            html.Button("Evaluation durchführen", id="evaluate-button", n_clicks=0, style={"margin-top": "10px"}),
            dbc.Spinner(html.Div(id="evaluation-output", style={"margin-top": "20px"}))
        ], width=12)
    ])
])


# Define the callback to update the prediction
@app.callback(
    Output("prediction-output", "children"),
    [Input("predict-button", "n_clicks")],
    [State("product-name-input", "value")]
)
def update_prediction(n_clicks, product_name):
    if n_clicks > 0 and product_name:
        predicted_labels, _ = predict([product_name], model, tokenizer, reverse_label_mapping, "cpu")
        return html.Div([
            html.H4(f"Predicted Label: {predicted_labels[0]}")
        ])
    return ""


# Define the callback to run the evaluation
@app.callback(
    Output("evaluation-output", "children"),
    [Input("evaluate-button", "n_clicks")],
    [State("num-samples-input", "value")]
)
def run_evaluation(n_clicks, num_samples):
    if n_clicks > 0 and num_samples:
        metrics, evaluation_details = evaluate_model(model, tokenizer, test_dataset, num_samples, reverse_label_mapping,
                                                     "cpu")

        details = []
        for detail in evaluation_details:
            details.append(html.Div([
                html.P(f"Produkt Name: {detail['sentence']}"),
                html.P(f"Korrekte Produktgruppe: {detail['true_label']}"),
                html.P(f"Vorhergesagte Produktgruppe: {detail['predicted_label']}"),
                html.Hr()
            ]))

        return html.Div([
            html.H4("Evaluation Metrics:"),
            html.P(f"Accuracy: {100*metrics['accuracy']:.2f}%"),
            html.P(f"Precision: {100*metrics['precision']:.2f}%"),
            html.P(f"Recall: {100*metrics['recall']:.2f}%"),
            html.P(f"F1 Score: {100*metrics['f1']:.2f}"),
            html.Hr(),
            html.H4("Evaluation Details:"),
            *details
        ])
    return ""


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
