from datasets import load_dataset
from transformers import Wav2Vec2FeatureExtractor, TimesformerModel
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
import winsound
from datasets import Dataset
from tqdm import tqdm
from datasets import Dataset, DatasetDict, Features, Value, Array4D, concatenate_datasets, load_from_disk
def load_and_combine_batches(save_dir, partition_name):
    """加载特定分区的所有批次，然后合并为一个数据集，显示进度条。"""
    batch_folders = [os.path.join(save_dir, partition_name, d) for d in
                     os.listdir(os.path.join(save_dir, partition_name))
                     if os.path.isdir(os.path.join(save_dir, partition_name, d))]
    # 添加进度条
    datasets = [Dataset.load_from_disk(folder) for folder in
                tqdm(sorted(batch_folders), desc=f"Loading batches from {partition_name}")]
    combined_dataset = concatenate_datasets(datasets)
    return combined_dataset


def create_dataset_dict(save_dir):
    """创建包含训练集和测试集的DatasetDict，并设置为torch格式，显示进度条。"""
    print("Loading training data...")
    train_dataset = load_and_combine_batches(save_dir, 'train')
    print("Loading testing data...")
    test_dataset = load_and_combine_batches(save_dir, 'test')

    # 将训练集和测试集设置为torch格式
    train_dataset.set_format(type='torch', columns=['pixel_values','input_values','labels'])
    test_dataset.set_format(type='torch', columns=['pixel_values','input_values','labels'])

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    return dataset_dict
# input_values,attention_mask_audio,token_type_ids,position_ids,head_mask,inputs_embeds,labels,output_attentions,output_hidden_states,return_dict,label_ids,labels,label.
os.environ['http_proxy'] = 'http://127.0.0.1:10809'
os.environ['https_proxy'] = 'http://127.0.0.1:10809'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tokenizer = AutoTokenizer.from_pretrained("wav2vec2/")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("wav2vec2/")

#dataset path
save_dir = 'E:/video_batches'
dataset = create_dataset_dict(save_dir)

train_dataset = dataset['train']
test_dataset = dataset['test']

# # 简单融合
# class Concatenation(nn.Module):
#     def __init__(self, feature_size=768):
#         super(Concatenation, self).__init__()
#         self.classifier = nn.Linear(768*2, 768)
#
#     def forward(self, audio_features, text_features):
#         cat_features = torch.cat((audio_features, text_features), dim=1)
#         cat_features = self.classifier(cat_features)
#
#         return cat_features


d_model = 768
nhead = 8
dropout = 0.1
layer_norm_eps = 1e-5
dim_feedforward = 3072


# 使用注意力机制融合模块
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


from transformers import AutoProcessor, AutoModelForAudioClassification, AutoModelForPreTraining, AutoModelForMaskedLM, \
    AutoModelForSequenceClassification,ViTForImageClassification
import torch.nn as nn
from transformers import AutoModel
from transformers import BertModel
import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from transformers import AutoImageProcessor, BeitModel
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor, Wav2Vec2Model
skew = -4
import torch
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


from transformers import Wav2Vec2ForPreTraining,AutoProcessor, AutoModelForAudioClassification, AutoModelForPreTraining, AutoModelForMaskedLM, AutoModelForSequenceClassification
import torch.nn as nn
from transformers import AutoModel
from transformers import BertModel, TimesformerModel, Wav2Vec2Model
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

        # 分别选择文本basemodel以及语音的basemodel
        self.audio_model = Wav2Vec2Model.from_pretrained("E:/audio_only/wav2vec12layers/", output_hidden_states=True)
        # self.audio_model = AutoModelForPreTraining.from_pretrained("facebook/wav2vec2-base", output_hidden_states =
        # True)
        self.img_model = TimesformerModel.from_pretrained("E:/video_model/12layers/", output_hidden_states=True)#checkpoint1

        # 选择融合机制
        self.fusion_model = CoAttention()
        # self.fusion_model = Concatenation()
        self.dropout = nn.Dropout(0.1)
        self.linear1 = nn.Linear(768*2, 28)
        self.linear = nn.Linear(768, 28) # output features from bert is 768 and 2 is ur number of labels

    def forward(
        self,
            input_values: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            pixel_values: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor]:
        outputs_audio = self.audio_model(input_values)
        # outputs_audio = self.img_model(pixel_values)
        outputs_img = self.img_model(pixel_values)


       ## 指定融合的匹配策略
        layer_fusion1 = self.fusion_model(outputs_audio.hidden_states[12][:, 0, :],outputs_img.hidden_states[12][:,0,:])
        layer_fusion2 = self.fusion_model(outputs_audio.hidden_states[11][:, 0, :],outputs_img.hidden_states[11][:,0,:])
        layer_fusion3 = self.fusion_model(outputs_audio.hidden_states[10][:, 0, :],outputs_img.hidden_states[10][:,0,:])
        layer_fusion4 = self.fusion_model(outputs_audio.hidden_states[9][:, 0, :],outputs_img.hidden_states[9][:,0,:])
        layer_fusion5 = self.fusion_model(outputs_audio.hidden_states[8][:, 0, :],outputs_img.hidden_states[8][:,0,:])#checkpoint2
        layer_fusion6 = self.fusion_model(outputs_audio.hidden_states[7][:, 0, :],outputs_img.hidden_states[7][:,0,:])
        layer_fusion7 = self.fusion_model(outputs_audio.hidden_states[6][:, 0, :],outputs_img.hidden_states[6][:,0,:])
        layer_fusion8 = self.fusion_model(outputs_audio.hidden_states[5][:, 0, :],outputs_img.hidden_states[5][:,0,:])
        layer_fusion9 = self.fusion_model(outputs_audio.hidden_states[4][:, 0, :],outputs_img.hidden_states[4][:,0,:])
        layer_fusion10 = self.fusion_model(outputs_audio.hidden_states[3][:,0,:],outputs_img.hidden_states[3][:,0,:])
        layer_fusion11 = self.fusion_model(outputs_audio.hidden_states[2][:,0,:],outputs_img.hidden_states[2][:,0,:])
        layer_fusion12 = self.fusion_model(outputs_audio.hidden_states[1][:,0,:],outputs_img.hidden_states[1][:,0,:])

        # outputs_fusion = self.w1*layer_fusion1 + self.w2*layer_fusion2 + self.w3*layer_fusion3 + self.w4*layer_fusion4 + self.w5*layer_fusion5 + self.w6*layer_fusion6 + self.w7*layer_fusion7 + self.w8*layer_fusion8 + self.w9*layer_fusion9 + self.w10*layer_fusion10 + self.w11*layer_fusion11 + self.w12*layer_fusion12
        # outputs_fusion = (layer_fusion1 + layer_fusion2+ layer_fusion3)/3.0
        # outputs_fusion = outputs_img.hidden_states[12][:, 0, :]
        outputs_fusion = layer_fusion1
        # + layer_fusion10 + layer_fusion11 + layer_fusion12
        # outputs_fusion = 0.0002443*layer_fusion1 + 0.0004886*layer_fusion2 + 0.0009772*layer_fusion3 + 0.0019544*layer_fusion4 + 0.0039088*layer_fusion5 + 0.0078176*layer_fusion6 + 0.0156352*layer_fusion7 + 0.0312704*layer_fusion8 + 0.0625408*layer_fusion9 + 0.1250816*layer_fusion10 + 0.2501632*layer_fusion11 + 0.5003264*layer_fusion12

        # 将融合好的特征进行分类
        logits = self.linear(outputs_fusion)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 28), labels.view(-1))

        output = (logits,)
        res = ((loss,) + output)
        return res


model = FusionModel()
for param in model.img_model.parameters():
    param.requires_grad = False

for param in model.audio_model.parameters():
    param.requires_grad = False

from transformers import TrainingArguments, Trainer, TrainerCallback

training_args = TrainingArguments(
    output_dir="test_trainer",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    save_strategy='steps',
    # no_cuda = True,
)

import numpy as np
import evaluate

metric = evaluate.load("evaluate/metrics/accuracy")
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     # print(predictions)
#     return metric.compute(predictions=predictions, references=labels)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    exp_logits = np.exp(logits)
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # 获取概率最高的类别作为预测结果
    predictions = np.argmax(probabilities, axis=1)

    # 检查预测和标签的长度是否相同
    if len(predictions) != len(labels):
        raise ValueError("Length of predictions and labels must be the same.")

    # 检查是否存在 NaN 值
    if np.any(np.isnan(predictions)) or np.any(np.isnan(labels)):
        raise ValueError("Predictions and labels must not contain NaN.")

    # 使用评估指标计算准确度
    return metric.compute(predictions=predictions, references=labels)


class CustomTrainer(Trainer):
    def create_optimizer(self):
        optimizer_grouped_parameters = [
            {
                "params": self.model.linear.parameters(),
                "lr": 5e-5,
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
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    # tokenizer = feature_extractor,
    compute_metrics=compute_metrics,
    # callbacks=[MyCallback],
)

trainer.train()

torch.cuda.empty_cache()

duration = 1000  # 持续时间/ms
frequency = 500  # 频率/Hz
winsound.Beep(frequency, duration)

