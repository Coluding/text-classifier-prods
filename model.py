import os
from typing import Union

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig, PreTrainedModel


class SequenceConfig(DistilBertConfig):
    def __init__(self, num_labels=2, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels

    @classmethod
    def from_pretrained(cls, model_name_or_path, num_labels=2, **kwargs):
        config = DistilBertConfig.from_pretrained(model_name_or_path, **kwargs)
        return cls(num_labels=num_labels, **config.to_dict())

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        super().save_pretrained(save_directory, push_to_hub=push_to_hub, **kwargs)
        self.num_labels = kwargs.get("num_labels", 2)

class TextClassifier(PreTrainedModel):
    config_class = DistilBertConfig
    base_model_prefix = "distilbert"

    def __init__(self, config: SequenceConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, self.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

    def forward(self, input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None, labels=None):
        outputs = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_state = outputs[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output

    @classmethod
    def from_pretrained(cls, model_name_or_path, *model_args, **kwargs):
        config = SequenceConfig.from_pretrained(model_name_or_path, num_labels=kwargs.pop("num_labels", 2))
        return super().from_pretrained(model_name_or_path, *model_args, config=config, **kwargs)


if __name__ == "__main__":
    import time
    model_name = "distilbert/distilbert-base-german-cased"
    model = TextClassifier.from_pretrained(model_name, num_labels=2)
    model = model.cuda()
    inp = torch.randint(0, 30522, (100, 128)).cuda()
    labels = torch.randint(0, 2, (100,)).cuda()
    now = time.time()
    out = model(inp, labels=labels)
    print(f"Time taken: {time.time() - now}")
