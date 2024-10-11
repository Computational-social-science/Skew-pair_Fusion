import torch
import librosa
from datasets import load_dataset
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import torch.nn as nn
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
import torch
import torch.nn as nn

class Concatenation(nn.Module):
    def __init__(self, feature_size=768):
        super(Concatenation, self).__init__()
        self.classifier = nn.Linear(768*2, 768)

    def forward(self, audio_features, text_features):
        cat_features = torch.cat((audio_features, text_features), dim=1)
        cat_features = self.classifier(cat_features)

        return cat_features
    
import torch
import torch.nn as nn
d_model = 768
nhead = 8
dropout = 0.1
layer_norm_eps = 1e-5
dim_feedforward = 3072

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

        fused_features = (x+y)/2
        return fused_features

    def _sa_block(self, q, k, v):
        x = self.self_attn(q, k, v, need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
    

class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()

        #self.text_model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states = True)
        #self.audio_model = AutoModelForPreTraining.from_pretrained("facebook/wav2vec2-base", output_hidden_states = True)
        self.audio_model = Wav2Vec2ForSequenceClassification.from_pretrained("/content/drive/MyDrive/myModel/wav2vec2_superb", output_hidden_states = True)
        #self.audio_model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", output_hidden_states = True)
        self.text_model = AutoModelForSequenceClassification.from_pretrained("/content/drive/MyDrive/myModel/bert_12layers", output_hidden_states=True)

        self.fusion_model = CoAttention()
        #self.fusion_model = Concatenation()
        self.dropout = nn.Dropout(0.1)
        self.linear1 = nn.Linear(768*2, 4)
        self.linear = nn.Linear(768, 4) 

    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        attention_mask_audio: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor]:
        outputs_audio = self.audio_model(input_values, attention_mask=attention_mask_audio)
        outputs_text  = self.text_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        #layer_fusion1 = self.fusion_model(outputs_audio.hidden_states[1][:,0,:],outputs_text.hidden_states[1][:,0,:])
        #layer_fusion2 = self.fusion_model(outputs_audio.hidden_states[1][:,0,:],outputs_text.hidden_states[5][:,0,:])
        #layer_fusion3 = self.fusion_model(outputs_audio.hidden_states[2][:,0,:],outputs_text.hidden_states[6][:,0,:])
        #layer_fusion4 = self.fusion_model(outputs_audio.hidden_states[3][:,0,:],outputs_text.hidden_states[7][:,0,:])
        layer_fusion5 = self.fusion_model(outputs_audio.hidden_states[8][:,0,:],outputs_text.hidden_states[1][:,0,:])
        layer_fusion6 = self.fusion_model(outputs_audio.hidden_states[9][:,0,:],outputs_text.hidden_states[2][:,0,:])
        layer_fusion7 = self.fusion_model(outputs_audio.hidden_states[10][:,0,:],outputs_text.hidden_states[3][:,0,:])
        layer_fusion8 = self.fusion_model(outputs_audio.hidden_states[11][:,0,:],outputs_text.hidden_states[4][:,0,:])
        layer_fusion9 = self.fusion_model(outputs_audio.hidden_states[12][:,0,:],outputs_text.hidden_states[5][:,0,:])
        #layer_fusion10 = self.fusion_model(outputs_audio.hidden_states[1][:,0,:],outputs_text.hidden_states[10][:,0,:])
        #layer_fusion11 = self.fusion_model(outputs_audio.hidden_states[7][:,0,:],outputs_text.hidden_states[11][:,0,:])
        #layer_fusion12 = self.fusion_model(outputs_audio.hidden_states[8][:,0,:],outputs_text.hidden_states[12][:,0,:])

        #outputs_fusion = self.w1*layer_fusion1 + self.w2*layer_fusion2 + self.w3*layer_fusion3 + self.w4*layer_fusion4 + self.w5*layer_fusion5 + self.w6*layer_fusion6 + self.w7*layer_fusion7 + self.w8*layer_fusion8 + self.w9*layer_fusion9 + self.w10*layer_fusion10 + self.w11*layer_fusion11 + self.w12*layer_fusion12
        outputs_fusion = (layer_fusion5 + layer_fusion6 + layer_fusion7 + layer_fusion8 + layer_fusion9)/5.0
        # + layer_fusion10 + layer_fusion11 + layer_fusion12
        #outputs_fusion = 0.0002443*layer_fusion1 + 0.0004886*layer_fusion2 + 0.0009772*layer_fusion3 + 0.0019544*layer_fusion4 + 0.0039088*layer_fusion5 + 0.0078176*layer_fusion6 + 0.0156352*layer_fusion7 + 0.0312704*layer_fusion8 + 0.0625408*layer_fusion9 + 0.1250816*layer_fusion10 + 0.2501632*layer_fusion11 + 0.5003264*layer_fusion12


        logits = self.linear(outputs_fusion)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 4), labels.view(-1))

        output = (logits,)
        return ((loss,) + output)

model = FusionModel()

for param in model.audio_model.parameters():
    param.requires_grad = False

for param in model.text_model.parameters():
    param.requires_grad = False

from transformers import TrainingArguments, Trainer, TrainerCallback

training_args = TrainingArguments(
    output_dir="test_trainer", evaluation_strategy="epoch",
    learning_rate = 5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10
    )

import numpy as np
import evaluate
from sklearn.metrics import accuracy_score
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

from transformers import Trainer, TrainingArguments
import torch

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
            }
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        return self.optimizer

class MyCallback(TrainerCallback):
    "A callback that prints a message at the beginning of training"
    # epoch = 1
    def on_epoch_begin(self, args, state, control, model, **kwargs):
      model.audio_model.save_pretrained(f'/yourModel/skew4/wav2vec2_{state.epoch}_epoch')
      model.text_model.save_pretrained(f'/yourModel/skew4/bert_{state.epoch}_epoch')
      #self.epoch = self.epoch + 1
      print(f"save{state.epoch}")

trainer = CustomTrainer(
    model=model,
    args=training_args,
    # data_collator=data_collator,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    #tokenizer = feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()