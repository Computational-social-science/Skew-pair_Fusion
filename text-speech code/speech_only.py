from transformers import AutoProcessor, AutoModelForAudioClassification, AutoModelForPreTraining, AutoModelForMaskedLM, AutoModelForSequenceClassification
import torch.nn as nn
import torch
import librosa
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import torch.nn.init as init
from transformers import AutoModel
import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
class audio_only_model(nn.Module):
    def __init__(self):
        super(audio_only_model, self).__init__()

        #--------------local_model---------------
        #self.model = HubertForSequenceClassification.from_pretrained("/content/drive/MyDrive/myModel/hubert", output_hidden_states = True)
        #self.model = Wav2Vec2ForSequenceClassification.from_pretrained("/content/drive/MyDrive/myModel/wav2vec2_superb", output_hidden_states = True)
        #self.model = HubertForSequenceClassification.from_pretrained("superb/hubert-base-superb-er", output_hidden_states = True)
        #self.model = AutoModelForPreTraining.from_pretrained("facebook/wav2vec2-base-960h", output_hidden_states = True)
        self.model = AutoModelForPreTraining.from_pretrained("facebook/wav2vec2-base", output_hidden_states = True)
        #self.model = Wav2Vec2ForSequenceClassification.from_pretrained("/content/drive/MyDrive/myModel/wav2vec2_superb_12layers", output_hidden_states = True)
        #self.model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-er", output_hidden_states = True)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(768, 4)

    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        attention_mask_audio: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor]:
        #print(input_values.size())
        outputs = self.model(input_values, attention_mask=attention_mask_audio)
        hidden_states = outputs.hidden_states
        # 取出模型的1-12层输出
        hidden_M = [
            #hidden_states[0][:, 0, :],
            hidden_states[1][:, 0, :],
            hidden_states[2][:, 0, :],
            hidden_states[3][:, 0, :],
            hidden_states[4][:, 0, :],
            hidden_states[5][:, 0, :],
            hidden_states[6][:, 0, :],
            hidden_states[7][:, 0, :],
            hidden_states[8][:, 0, :],
            hidden_states[9][:, 0, :],
            hidden_states[10][:, 0, :],
            hidden_states[11][:, 0, :],
            hidden_states[12][:, 0, :],
        ]
        # 取第1层作为分类器的输入
        logits = self.classifier(hidden_states[1][:, 0, :])

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 4), labels.view(-1))

        output = (logits,)
        return ((loss,) + output)
    
model = audio_only_model()

for param in model.model.parameters():
   param.requires_grad = False

import numpy as np
import evaluate

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    evaluation_strategy="epoch",
    learning_rate = 1e-3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    fp16 = True,
    output_dir="wofeishenling/wofei",
    #push_to_hub=True,
    )

trainer = Trainer(
    model=m,
    #model = AutoModelForPreTraining.from_pretrained("facebook/wav2vec2-base-960h",
    #                        output_hidden_states = True,
    #                        num_labels = 4),
    args=training_args,
    # data_collator = data_collator,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    compute_metrics=compute_metrics,
    #callbacks=[EarlyStoppingCallback(early_stopping_patience=3), SaveBestModelCallback()]
)

trainer.train()