import os
from typing import Optional, Tuple, Any
import torch.nn as nn
import torch.utils.checkpoint
from PIL import Image
from datasets import load_from_disk
from torch.nn import CrossEntropyLoss
from transformers import AutoImageProcessor, BeitModel, BeitForMaskedImageModeling
from transformers import BertModel
import numpy as np

# load dataset
dataset = load_from_disk('hateful_memes')

class text_only_model(nn.Module):
    def __init__(self):
        super(text_only_model, self).__init__()
        # choose and load baseModel
        self.bert_base = BertModel.from_pretrained("bert-base-uncased/", output_hidden_states=True)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 2)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            # token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Tuple[Any, Any]:
        outputs = self.bert_base(
            input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            # position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs.hidden_states


        # take layer 12 as the input to the classifier
        logits = self.classifier(hidden_states[12][:, 0, :])
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        output = (logits,)
        return (loss,) + output



model = text_only_model()

# Freeze the parameters of BERT
# for param in model.bert_base.parameters():
#     param.requires_grad = False

from transformers import Trainer, TrainingArguments
import evaluate

metric = evaluate.load("evaluate/metrics/roc_auc")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probabilities = 1 / (1 + np.exp(-logits))[:, 1]
    return metric.compute(prediction_scores=probabilities, references=labels)


# set the training parameters
class CustomTrainer(Trainer):
    def create_optimizer(self):
        optimizer_grouped_parameters = [
            {
                "params": self.model.classifier.parameters(),
                "lr": 1e-3,
                "weight_decay": self.args.weight_decay
            },
            {
                "params": self.model.bert_base.parameters(),
                "lr": self.args.learning_rate,
                "weight_decay": self.args.weight_decay
            },
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        return self.optimizer


training_args = TrainingArguments(
    output_dir="test_trainer", evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    # no_cuda=True,
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    compute_metrics=compute_metrics,

)

# text-train
trainer.train()
model.bert_base.save_pretrained("text_only/bert12layer_hatefulmemes")
