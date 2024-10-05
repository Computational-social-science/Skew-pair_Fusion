from transformers import AutoProcessor, AutoModelForAudioClassification, AutoModelForPreTraining, AutoModelForMaskedLM, \
    AutoModelForSequenceClassification, AutoTokenizer, Wav2Vec2ForCTC, AutoModelForCTC, Wav2Vec2Processor
import torch.nn as nn
import torch
import librosa
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor, Wav2Vec2Model
import torch.nn.init as init
from transformers import AutoModel
import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from VGG_pre import read_picklefile_train,read_picklefile_test
from datasets import Dataset, DatasetDict
import numpy as np
import os
from datasets import Dataset, DatasetDict, Features, Value, Array4D, concatenate_datasets, load_from_disk
from tqdm import tqdm
import numpy as np
import torch

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
feature_extractor = Wav2Vec2Processor.from_pretrained("wav2vec2/")

# features_train,labels_train = read_picklefile_train()
# features_test,labels_test = read_picklefile_test()
# print(type(features_test))
# train_dataset = Dataset.from_dict({'input_values': features_train, 'labels': labels_train})
# test_dataset = Dataset.from_dict({'input_values': features_test, 'labels': labels_test})

# train_dataset = Dataset.load_from_disk('E:\AVE-fusion-train/')
# test_dataset = Dataset.load_from_disk('E:\AVE-fusion-test/')


#dataset path
save_dir = 'E:/video_batches'


dataset = create_dataset_dict(save_dir)

# train_dataset = Dataset.load_from_disk('E:\AVE-fusion-train/')
# test_dataset = Dataset.load_from_disk('E:\AVE-fusion-test/')
train_dataset = dataset['train']
test_dataset = dataset['test']
print(len(train_dataset))

#
class audio_only_model(nn.Module):
    def __init__(self):
        super(audio_only_model, self).__init__()
        self.model = Wav2Vec2Model.from_pretrained("wav2vec2/", output_hidden_states = True)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(768, 28)

    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
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
        # outputs = self.model(input_values, attention_mask=attention_mask_audio)
        outputs = self.model(input_values)
        hidden_states = outputs.hidden_states

        # 取第1层作为分类器的输入
        logits = self.classifier(hidden_states[1][:, 0, :])
        # print("logits shape:", logits.shape)
        # print("labels shape:", labels.shape)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 28), labels.view(-1))

        output = (logits,)
        return ((loss,) + output)

m = audio_only_model()
for param in m.model.parameters():
    # print(param)
    param.requires_grad = True

import numpy as np
import evaluate


metric = evaluate.load("evaluate/metrics/accuracy")
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

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    evaluation_strategy="epoch",
    learning_rate = 5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    fp16 = True,
    output_dir="wav2vec_only",
    #push_to_hub=True,
    )
trainer = Trainer(
    model=m,
    #model = AutoModelForPreTraining.from_pretrained("facebook/wav2vec2-base-960h",
    #                        output_hidden_states = True,
    #                        num_labels = 4),
    args=training_args,
    # data_collator = data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    #callbacks=[EarlyStoppingCallback(early_stopping_patience=3), SaveBestModelCallback()]
)

trainer.train()
m.model.save_pretrained('E:/audio_only/wav2vec01layers')

trainer.evaluate()
