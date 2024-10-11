from datasets import load_dataset
from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import torch
import torch.nn as nn
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

d_model = 768
nhead = 8
dropout = 0.1
layer_norm_eps = 1e-5
dim_feedforward = 3072


# the Attention Mechanism Fusion module
class CoAttention(nn.Module):
    def __init__(self, feature_size=768):
        super(CoAttention, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)

    def forward(self, af, tf):
        x = self.norm1(af + self._sa_block(af, tf, tf))
        x = self.norm2(x + self._ff_block(x))

        y = self.norm1(tf + self._sa_block(tf, af, af))
        y = self.norm2(y + self._ff_block(y))

        x1 = self.norm1(x + self._sa_block(x, y, y))
        x1 = self.norm2(x1 + self._ff_block(x1))

        y1 = self.norm1(y + self._sa_block(y, x, x))
        y1 = self.norm2(y1 + self._ff_block(y1))

        x2 = self.norm1(x1 + self._sa_block(x1, y1, y1))
        x2 = self.norm2(x2 + self._ff_block(x2))

        y2 = self.norm1(y1 + self._sa_block(y1, x1, x1))
        y2 = self.norm2(y2 + self._ff_block(y2))

        fused_features = (x + y) / 2
        return fused_features

    def _sa_block(self, q, k, v):
        x = self.self_attn(q, k, v, need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


import torch.nn as nn
from transformers import AutoModel
from transformers import BertModel
import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from transformers import AutoImageProcessor, BeitModel

skew = -4
import torch
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()

        # load the model trained before
        self.text_model = BertModel.from_pretrained("text_only/bert12layer_hatefulmemes",
                                                    output_hidden_states=True)
        self.img_model = BeitModel.from_pretrained("img_only/beit12layers_hatefulmemes",
                                                   output_hidden_states=True)

        # Select the fusion mechanism
        self.fusion_model = CoAttention()
        self.dropout = nn.Dropout(0.1)
        self.linear1 = nn.Linear(768 * 2, 2)
        self.linear = nn.Linear(768, 2)  # output features from bert is 768 and 2 is ur number of labels

    def forward(
            self,
            pixel_values: Optional[torch.Tensor] = None,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor]:
        outputs_audio = self.img_model(pixel_values)
        outputs_text = self.text_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # Specify the matching policy for the fusion
        # layer_fusion1 = self.fusion_model(outputs_audio.hidden_states[1][:,0,:],outputs_text.hidden_states[1][:,0,:])
        # layer_fusion2 = self.fusion_model(outputs_audio.hidden_states[1][:,0,:],outputs_text.hidden_states[5][:,0,:])
        # layer_fusion3 = self.fusion_model(outputs_audio.hidden_states[2][:,0,:],outputs_text.hidden_states[6][:,0,:])
        # layer_fusion4 = self.fusion_model(outputs_audio.hidden_states[3][:,0,:],outputs_text.hidden_states[7][:,0,:])
        layer_fusion5 = self.fusion_model(outputs_audio.hidden_states[12][:, 0, :],
                                          outputs_text.hidden_states[6][:, 0, :])
        # layer_fusion6 = self.fusion_model(outputs_audio.hidden_states[9][:,0,:],outputs_text.hidden_states[10][:,0,:])
        # layer_fusion7 = self.fusion_model(outputs_audio.hidden_states[8][:,0,:],outputs_text.hidden_states[9][:,0,:])
        # layer_fusion8 = self.fusion_model(outputs_audio.hidden_states[11][:,0,:],outputs_text.hidden_states[6][:,0,:])
        # layer_fusion9 = self.fusion_model(outputs_audio.hidden_states[12][:,0,:],outputs_text.hidden_states[7][:,0,:])
        # layer_fusion10 = self.fusion_model(outputs_audio.hidden_states[1][:,0,:],outputs_text.hidden_states[10][:,0,:])
        # layer_fusion11 = self.fusion_model(outputs_audio.hidden_states[7][:,0,:],outputs_text.hidden_states[11][:,0,:])
        # layer_fusion12 = self.fusion_model(outputs_audio.hidden_states[8][:,0,:],outputs_text.hidden_states[12][:,0,:])

        # outputs_fusion = self.w1*layer_fusion1 + self.w2*layer_fusion2 + self.w3*layer_fusion3 + self.w4*layer_fusion4 + self.w5*layer_fusion5 + self.w6*layer_fusion6 + self.w7*layer_fusion7 + self.w8*layer_fusion8 + self.w9*layer_fusion9 + self.w10*layer_fusion10 + self.w11*layer_fusion11 + self.w12*layer_fusion12
        # outputs_fusion = (layer_fusion5 + layer_fusion6 + layer_fusion7)/3.0
        # + layer_fusion10 + layer_fusion11 + layer_fusion12
        # outputs_fusion = 0.0002443*layer_fusion1 + 0.0004886*layer_fusion2 + 0.0009772*layer_fusion3 + 0.0019544*layer_fusion4 + 0.0039088*layer_fusion5 + 0.0078176*layer_fusion6 + 0.0156352*layer_fusion7 + 0.0312704*layer_fusion8 + 0.0625408*layer_fusion9 + 0.1250816*layer_fusion10 + 0.2501632*layer_fusion11 + 0.5003264*layer_fusion12

        # Classify the fused features
        logits = self.linear(layer_fusion5)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        output = (logits,)
        res = ((loss,) + output)
        return res


model = FusionModel()
for param in model.img_model.parameters():
    param.requires_grad = False

for param in model.text_model.parameters():
    param.requires_grad = False

from transformers import TrainingArguments, Trainer, TrainerCallback

training_args = TrainingArguments(
    output_dir="test_trainer", evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    save_strategy='steps',
    # no_cuda = True,
)

import numpy as np
import evaluate

metric = evaluate.load("evaluate/metrics/roc_auc")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probabilities = 1 / (1 + np.exp(-logits))[:, 1]
    return metric.compute(prediction_scores=probabilities, references=labels)


class CustomTrainer(Trainer):
    def create_optimizer(self):
        optimizer_grouped_parameters = [
            {
                "params": self.model.linear.parameters(),
                "lr": 1e-3,
                "weight_decay": self.args.weight_decay
            },
            {
                "params": self.model.fusion_model.parameters(),
                "lr": self.args.learning_rate,
                "weight_decay": self.args.weight_decay
            },
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        return self.optimizer


trainer = CustomTrainer(
    model=model,
    args=training_args,
    # data_collator=data_collator,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    # tokenizer = feature_extractor,
    compute_metrics=compute_metrics,
    # callbacks=[MyCallback],
)

trainer.train()
# torch.save(model, "fusion_bert-beit/fusion_3.0_epoch.pt")
torch.cuda.empty_cache()
