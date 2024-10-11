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
image_processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
model = BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k")

from transformers import AutoImageProcessor, BeitModel
import torch.nn as nn
import torch
from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


class img_only_model(nn.Module):
    def __init__(self):
        super(img_only_model, self).__init__()
        #load local model
        self.model = BeitModel.from_pretrained("beit/", output_hidden_states=True)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(768, 2)

    def forward(self, pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None, ) -> torch.Tensor:
        outputs = self.model(pixel_values)
        hidden_states = outputs.hidden_states
        # take layer X as the input to the classifier
        logits = self.classifier(hidden_states[12][:, 0, :])   #n = 12

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        output = (logits,)
        return ((loss,) + output)


m = img_only_model()

# Freeze the parameters of BEIT
# for param in m.model.parameters():
#    param.requires_grad = False

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
    evaluation_strategy="epoch",
    learning_rate = 4e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    output_dir="img_only/img1",
    #no_cuda=True,
    )

trainer = Trainer(
    model=m,
    args=training_args,
    # data_collator = data_collator,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    compute_metrics=compute_metrics,
)

#vision-train
trainer.train()
m.model.save_pretrained('img_only/beit12layers_hatefulmemes')