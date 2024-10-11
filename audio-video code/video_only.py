import pandas as pd
import torchaudio
from transformers import Wav2Vec2Processor
import os
from datasets import Dataset, DatasetDict, Features, Value
from tqdm import tqdm
from datasets import load_from_disk, concatenate_datasets
import torch.nn as nn
import torch
from typing import Optional, Tuple
from torch.nn import CrossEntropyLoss
from transformers import TimesformerModel
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer

processor = Wav2Vec2Processor.from_pretrained("F:/dataset/wav2vec2-base-960h")


def read_and_process_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != processor.feature_extractor.sampling_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate,
                                                   new_freq=processor.feature_extractor.sampling_rate)
        waveform = resampler(waveform)

    inputs = processor(waveform, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt")
    return inputs.input_values


def create_and_save_batches(csv_file, audio_dir, save_dir, partition_name, batch_size=50):
    df = pd.read_csv(csv_file, header=None, names=['audio_name', 'label'])
    total_size = len(df)

    specific_save_dir = os.path.join(save_dir, partition_name)
    os.makedirs(specific_save_dir, exist_ok=True)

    for start in tqdm(range(0, total_size, batch_size), desc=f"Processing {partition_name} batches"):
        end = min(start + batch_size, total_size)
        df_batch = df.iloc[start:end]

        data = {
            'input_values': [],
            'label': []
        }

        for _, row in tqdm(df_batch.iterrows(), total=len(df_batch),
                           desc=f"Processing {partition_name} batch {start // batch_size}"):
            audio_path = os.path.join(audio_dir, f"{row['audio_name']}.wav")
            try:
                input_tensor = read_and_process_audio(audio_path)
                data['input_values'].append(input_tensor.squeeze().numpy())  # 确保数据为二维
                data['label'].append(int(row['label']))
            except Exception as e:
                print(f"Error processing audio {audio_path}: {e}")
                continue

        if data['input_values']:
            features = Features({
                'input_values': Array2D(dtype="float32", shape=(None, None)),  # 动态维度
                'label': Value(dtype='int64')
            })

            dataset = Dataset.from_dict(data, features=features)
            dataset.set_format(type='torch', columns=['input_values', 'label'])
            batch_save_path = os.path.join(specific_save_dir, f'batch_{start // batch_size}')
            os.makedirs(batch_save_path, exist_ok=True)
            dataset.save_to_disk(batch_save_path)


audio_dir = 'F:/dataset/AVE/audio'
train_csv = 'F:/dataset/AVE/train.csv'
test_csv = 'F:/dataset/AVE/test.csv'
save_dir = 'F:/dataset/AVE/audio_batches'

create_and_save_batches(train_csv, audio_dir, save_dir, 'train', batch_size=50)
create_and_save_batches(test_csv, audio_dir, save_dir, 'test', batch_size=50)


def load_and_combine_batches(save_dir, partition_name):
    batch_folders = [os.path.join(save_dir, partition_name, d) for d in
                     os.listdir(os.path.join(save_dir, partition_name))
                     if os.path.isdir(os.path.join(save_dir, partition_name, d))]
    datasets = [Dataset.load_from_disk(folder) for folder in
                tqdm(sorted(batch_folders), desc=f"Loading batches from {partition_name}")]
    combined_dataset = concatenate_datasets(datasets)
    return combined_dataset


def create_dataset_dict(save_dir):
    print("Loading training data...")
    train_dataset = load_and_combine_batches(save_dir, 'train')
    print("Loading testing data...")
    test_dataset = load_and_combine_batches(save_dir, 'test')

    train_dataset.set_format(type='torch', columns=['pixel_values', 'input_values', 'labels'])
    test_dataset.set_format(type='torch', columns=['pixel_values', 'input_values', 'labels'])

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    return dataset_dict


save_dir = 'E:/video_batches'

dataset = create_dataset_dict(save_dir)

dataset.save_to_disk("F:\dataset\AVE\dataset")

dataset = load_from_disk('F:\dataset\AVE\dataset')


class img_only_model(nn.Module):
    def __init__(self):
        super(img_only_model, self).__init__()
        self.model = TimesformerModel.from_pretrained("E:/timesformer/", output_hidden_states=True)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(768, 28)

    def forward(self, pixel_values: torch.Tensor,labels: Optional[torch.Tensor] = None, ) -> torch.Tensor:
        outputs = self.model(pixel_values)
        hidden_states = outputs.hidden_states
        logits = self.classifier(hidden_states[12][:, 0, :])

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 28), labels.view(-1))

        output = (logits,)
        return ((loss,) + output)


m = img_only_model()

# for param in m.model.parameters():
#    param.requires_grad = True

metric = evaluate.load("E:/evaluate/metrics/accuracy/")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    exp_logits = np.exp(logits)
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    predictions = np.argmax(probabilities, axis=1)

    if len(predictions) != len(labels):
        raise ValueError("Length of predictions and labels must be the same.")

    if np.any(np.isnan(predictions)) or np.any(np.isnan(labels)):
        raise ValueError("Predictions and labels must not contain NaN.")

    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    output_dir="wofeishenling"
    # no_cuda=True,
)

trainer = Trainer(
    model=m,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    compute_metrics=compute_metrics,
)

trainer.train()
torch.cuda.empty_cache()
m.model.save_pretrained('E:/video_model/12layers')
