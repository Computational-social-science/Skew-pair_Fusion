import torch
import librosa
from datasets import load_dataset
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("wofeishenling/autotrain-iemocap_text_4-39809103601")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

tokenized_train_dataset = train_dataset.map(lambda example: feature_extractor(example["speech"],
                                        sampling_rate=16000,
                                        max_length=int(feature_extractor.sampling_rate * 5),
                                        truncation='max_length',
                                        padding = 'max_length',
                                        return_tensors="pt",
                                        return_attention_mask = True,
                                        ))
tokenized_test_dataset = test_dataset.map(lambda example: feature_extractor(example["speech"],
                                        sampling_rate=16000,
                                        max_length=int(feature_extractor.sampling_rate * 5),
                                        truncation='max_length',
                                        padding = 'max_length',
                                        return_tensors="pt",
                                        return_attention_mask = True
                                        ))

tokenized_train_dataset = tokenized_train_dataset.rename_column("attention_mask", "attention_mask_audio")
tokenized_test_dataset = tokenized_test_dataset.rename_column("attention_mask", "attention_mask_audio")

tokenized_train_dataset = tokenized_train_dataset.map(lambda example: tokenizer(example["transcription"], truncation=True, padding='max_length', max_length=128))
tokenized_test_dataset = tokenized_test_dataset.map(lambda example: tokenizer(example["transcription"], truncation=True, padding='max_length', max_length=128))

tokenized_train_dataset = tokenized_train_dataset.remove_columns(["file","audio","transcription","speech"])
tokenized_test_dataset = tokenized_test_dataset.remove_columns(["file","audio","transcription","speech"])
tokenized_train_dataset = tokenized_train_dataset.rename_column("label", "labels")
tokenized_test_dataset = tokenized_test_dataset.rename_column("label", "labels")

tokenized_test_dataset.set_format("torch")
tokenized_train_dataset.set_format("torch")

def reshape_tensor(example):
    # Assume 'sample' is a tensor with shape [1, 5]
    example["input_values"] = example["input_values"].squeeze(0)
    example["attention_mask_audio"] = example["attention_mask_audio"].squeeze(0)
    return example
tokenized_train_dataset = tokenized_train_dataset.map(reshape_tensor)
tokenized_test_dataset = tokenized_test_dataset.map(reshape_tensor)

tokenized_test_dataset.save_to_disk("/content/drive/MyDrive/iemocap4_5s/test.hf")
tokenized_train_dataset.save_to_disk("/content/drive/MyDrive/iemocap4_5s/train.hf")