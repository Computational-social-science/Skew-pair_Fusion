import json
from datasets import Dataset, DatasetDict

with open('LE-dataset v2.1-150-topic-openchat- v1.2.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

latin_phrases = [entry["local sentence"] for entry in data]
english_sentences = [entry["llmed reponse"] for entry in data]
topics = [entry["topic"] for entry in data]

data_dict = {
    "latin_phrase": latin_phrases,
    "english_sentence": english_sentences,
    "topic": topics
}

dataset = Dataset.from_dict(data_dict)

dataset_dict = DatasetDict({"train": dataset})

dataset.save_to_disk("latin_english_dataset_with_topics")