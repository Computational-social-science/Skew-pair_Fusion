import os

import librosa
import pandas as pd
import av
import numpy as np
import torch
import torchaudio
from transformers import AutoImageProcessor, TimesformerModel
from datasets import Dataset, DatasetDict, Features, Value, Array4D, concatenate_datasets, load_from_disk, Sequence, \
    Features, Array3D, Array2D
from tqdm import tqdm
from torchvision import transforms
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor


def read_and_process_audio(audio_path):
    processor = Wav2Vec2Processor.from_pretrained("wav2vec2")
    audio_files_train = librosa.load(audio_path, sr=16000)[0]
    inputs = processor(audio_files_train, sampling_rate=16000, return_tensors='pt', padding=True, truncation=True,
                       max_length=16000 * 5, return_attention_mask=True)

    # print(inputs.input_values)[0]
    return inputs.input_values[0]


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def read_and_process_video(video_path, num_frames=8):
    container = av.open(video_path)
    stream = container.streams.video[0]
    total_frames = stream.frames
    frame_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=np.int64)

    frames = []
    for frame_idx in frame_indices:
        container.seek(int(frame_idx), any_frame=False)
        frame = next(container.decode(video=0))
        image = frame.to_image()
        tensor = transform(image)
        frames.append(tensor)
    return torch.stack(frames)


def create_and_save_batches(csv_file, video_dir, audio_dir,save_dir, partition_name, batch_size=50):
    df = pd.read_csv(csv_file, header=None, names=['video_name', 'labels'])
    total_size = len(df)
    image_processor = AutoImageProcessor.from_pretrained("E:/videomae-base")

    specific_save_dir = os.path.join(save_dir, partition_name)
    os.makedirs(specific_save_dir, exist_ok=True)

    for start in tqdm(range(0, total_size, batch_size), desc=f"Processing {partition_name} batches"):
        end = min(start + batch_size, total_size)
        df_batch = df.iloc[start:end]

        data = {
            'pixel_values': [],
            'input_values': [],
            'labels': []
        }

        for _, row in tqdm(df_batch.iterrows(), total=len(df_batch),
                           desc=f"Processing {partition_name} batch {start // batch_size}"):
            video_path = os.path.join(video_dir, f"{row['video_name']}.mp4")
            audio_path = os.path.join(audio_dir, f"{row['video_name']}.wav")
            try:
                audio_tensor =  read_and_process_audio(audio_path)
                video_tensor = read_and_process_video(video_path)
                data['pixel_values'].append(video_tensor)
                data['input_values'].append(audio_tensor)
                data['labels'].append(int(row['labels']))
            except Exception as e:
                print(f"Error processing video {video_path}: {e}")
                continue

        if data['pixel_values']:
            features = Features({
                'pixel_values': Array4D(dtype="float16", shape=(8, 3, 224, 224)),
                'input_values': Sequence(Value(dtype="float64")),
                'labels': Value(dtype='int64')
            })

            dataset = Dataset.from_dict(data, features=features)
            dataset = Dataset.from_dict(data)
            dataset.set_format(type='torch', columns=['pixel_values', 'input_values','labels'])
            batch_save_path = os.path.join(specific_save_dir, f'batch_{start // batch_size}')
            os.makedirs(batch_save_path, exist_ok=True)
            dataset.save_to_disk(batch_save_path)


# config params
video_dir = 'E:\AVE_Dataset\AVE'
train_csv = 'E:/train.csv'
test_csv = 'E:/test.csv'
save_dir = 'E:/video_batches'
audio_dir = 'E:\AVE'

create_and_save_batches(train_csv, video_dir,audio_dir, save_dir, 'train', batch_size=100)
create_and_save_batches(test_csv, video_dir,audio_dir, save_dir, 'test', batch_size=100)