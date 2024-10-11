from transformers import AutoProcessor, AutoModelForAudioClassification, AutoModelForPreTraining, AutoModelForMaskedLM, AutoModelForSequenceClassification
import torch.nn as nn
from transformers import AutoTokenizer
import torch
import librosa
from transformers import AutoModel
from transformers import BertModel
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch.nn.init as init
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

class text_only_model(nn.Module):
    def __init__(self):
        super(text_only_model, self).__init__()
        # 选择并加载baseModel
        self.bert_base = BertModel.from_pretrained("bert-base-uncased", output_hidden_states = True)
        #self.bert_base = AutoModelForSequenceClassification.from_pretrained("/content/drive/MyDrive/myModel/bert_12layers", output_hidden_states=True)
        #self.bert_base = AutoModelForSequenceClassification.from_pretrained("JerryM/distilbert-base-uncased-finetuned-emotion",output_hidden_states=True)
        #self.bert_base = AutoModelForSequenceClassification.from_pretrained("wofeishenling/autotrain-iemocap_text_4-39809103601", output_hidden_states = True)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 4)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        #token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor]:
        #print(model.M_12_1.weight)
        outputs = self.bert_base(
            input_ids,
            attention_mask=attention_mask,
            #token_type_ids=token_type_ids,
            #position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs.hidden_states
        #print(hidden_states[1][:, 0, :].shape) #32*768

        logits = self.classifier(hidden_states[12][:, 0, :])
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 4), labels.view(-1))

        output = (logits,)
        return ((loss,) + output)

model = text_only_model()

for param in model.bert_base.parameters():
    param.requires_grad = False

from transformers import Trainer, TrainingArguments

# 设置训练参数
class CustomTrainer(Trainer):
    def create_optimizer(self):
        optimizer_grouped_parameters = [
            {
              "params": self.model.classifier.parameters(),
              "lr": 1e-3,
              "weight_decay": self.args.weight_decay
            },
            {
              "params": [self.model.w1, self.model.w2],
              "lr": 0.005,
              "weight_decay": self.args.weight_decay
            }
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        return self.optimizer

training_args = TrainingArguments(
    output_dir="test_trainer", evaluation_strategy="epoch",
    learning_rate = 5e-5,
    weight_decay = 0.1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    )


metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()