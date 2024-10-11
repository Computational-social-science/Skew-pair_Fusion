from datasets import load_from_disk

dataset = load_from_disk("latin_english_dataset_with_topics")

import torch
from transformers import BertTokenizer, BertModel
from datasets import load_from_disk
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from tqdm import tqdm
import matplotlib.pyplot as plt

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据集
dataset = load_from_disk("latin_english_dataset_with_topics")
print(dataset)

# 采样1万个样本
sample_size = 10000
sampled_dataset = dataset.shuffle(seed=42).select(range(sample_size))


latin_texts = sampled_dataset["latin_phrase"]
english_texts = sampled_dataset["english_sentence"]


model_name = "huggingface/hub/models--bert-base-multilingual-cased/"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, output_hidden_states=True).to(device)

def extract_features(texts, batch_size=32):
    all_features = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting features"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        hidden_states = outputs.hidden_states
        for j in range(len(batch_texts)):
            features = [hidden_states[layer][j, 0, :].cpu().numpy() for layer in range(1, len(hidden_states))]
            all_features.append(features)
    return all_features

latin_features = extract_features(latin_texts)
english_features = extract_features(english_texts)

def combine_features(latin_features, english_features, dense, skew):
    combined_features = []
    for i in tqdm(range(len(latin_features)), desc=f"Combining features with skew {skew}"):
        combined_vector = []
        latin_start = max(1, 1 + max(0, skew))  
        english_start = max(1, 1 + max(0, -skew))  
        for d in range(dense):
            latin_layer = latin_start + d
            english_layer = english_start + d
            if 1 <= latin_layer <= 12 and 1 <= english_layer <= 12:
                latin_vector = latin_features[i][latin_layer - 1]
                english_vector = english_features[i][english_layer - 1]
                combined_vector.extend(latin_vector)
                combined_vector.extend(english_vector)
        combined_features.append(np.array(combined_vector))
    return np.array(combined_features)

def evaluate_clustering(combined_features, n_clusters=4):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(combined_features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(scaled_features)
    labels = kmeans.labels_
    if len(set(labels)) > 1:
        silhouette_avg = silhouette_score(scaled_features, labels)
        calinski_harabasz_avg = calinski_harabasz_score(scaled_features, labels)
        davies_bouldin_avg = davies_bouldin_score(scaled_features, labels)
    else:
        silhouette_avg = -1
        calinski_harabasz_avg = -1
        davies_bouldin_avg = -1
    return silhouette_avg, calinski_harabasz_avg, davies_bouldin_avg


results = []


dense = 3
for skew in range(-7, 8):  
    combined_features = combine_features(latin_features, english_features, dense, skew)
    silhouette_avg, calinski_harabasz_avg, davies_bouldin_avg = evaluate_clustering(combined_features)
    print(f"Skew {skew}: Silhouette Score = {silhouette_avg}, Calinski-Harabasz Score = {calinski_harabasz_avg}, Davies-Bouldin Score = {davies_bouldin_avg}")
    results.append((skew, silhouette_avg, calinski_harabasz_avg, davies_bouldin_avg))


df = pd.DataFrame(results, columns=["Skew", "Silhouette Score", "Calinski-Harabasz Score", "Davies-Bouldin Score"])


df.to_csv("multi_layer_combination_results_4topics.csv_3dense", index=False)


results = []


for latin_layer in range(1, 13):
    for english_layer in range(1, 13):
        combined_features = combine_features(latin_features, english_features, latin_layer, english_layer)
        silhouette_avg, calinski_harabasz_avg, davies_bouldin_avg = evaluate_clustering(combined_features)
        print(silhouette_avg)
        skewness = latin_layer - english_layer
        results.append((latin_layer, english_layer, skewness, silhouette_avg, calinski_harabasz_avg, davies_bouldin_avg))


df = pd.DataFrame(results, columns=["Latin Layer", "English Layer", "Skewness", "Silhouette Score", "Calinski-Harabasz Score", "Davies-Bouldin Score"])


df.to_csv("layer_combination_results_4topics.csv", index=False)


experiment_data = {
    "layers": layers,
    "latin_silhouette_scores": [float(score) for score in latin_silhouette_scores],
    "latin_calinski_harabasz_scores": [float(score) for score in latin_calinski_harabasz_scores],
    "latin_davies_bouldin_scores": [float(score) for score in latin_davies_bouldin_scores],
    "english_silhouette_scores": [float(score) for score in english_silhouette_scores],
    "english_calinski_harabasz_scores": [float(score) for score in english_calinski_harabasz_scores],
    "english_davies_bouldin_scores": [float(score) for score in english_davies_bouldin_scores]
}