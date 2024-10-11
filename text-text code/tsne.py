import torch
from transformers import BertTokenizer, BertModel
from datasets import load_from_disk
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = load_from_disk("latin_english_dataset_with_topics")

sample_size = 2500
sampled_dataset = dataset.shuffle(seed=42).select(range(sample_size))

latin_texts = sampled_dataset["latin_phrase"]
english_texts = sampled_dataset["english_sentence"]
topics = sampled_dataset["topic"]

model_name = "/home/pp/.cache/huggingface/hub/models--bert-base-multilingual-cased/snapshots/3f076fdb1ab68d5b2880cb87a0886f315b8146f8"
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

def combine_features(latin_features, english_features, layers):
    combined_features = []
    for i in tqdm(range(len(latin_features)), desc="Combining features"):
        combined_vector = []
        for layer in layers:
            latin_vector = latin_features[i][layer - 1]
            english_vector = english_features[i][layer - 1]
            combined_vector.extend(latin_vector)
            combined_vector.extend(english_vector)
        combined_features.append(np.array(combined_vector))
    return np.array(combined_features)

def cluster_and_plot_tsne(features, topics, title, n_clusters=4):
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    
    # Reduce dimensions with t-SNE
    tsne = TSNE(n_components=2, random_state=52)
    tsne_results = tsne.fit_transform(features)
    
    # Create a DataFrame with t-SNE results and cluster labels
    df = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
    df['Cluster'] = labels
    df['Topic'] = topics
    
    # Plot the results
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='TSNE1', y='TSNE2', hue='Cluster', palette='Set1', data=df, s=20)
    
    # Remove the legend and axis ticks
    plt.legend([],[], frameon=False)  # Remove the legend
    plt.xticks([])  # Remove x-axis ticks
    plt.yticks([])  # Remove y-axis ticks
    
    plt.title(f"{title}")
    plt.show()


latin1_eng1_features = combine_features(latin_features, english_features, layers=[1])
latin12_eng12_features = combine_features(latin_features, english_features, layers=[12])
latin1_12_eng1_12_features = combine_features(latin_features, english_features, layers=range(1, 13))

scaler = StandardScaler()
latin1_eng1_features = scaler.fit_transform(latin1_eng1_features)
latin12_eng12_features = scaler.fit_transform(latin12_eng12_features)
latin1_12_eng1_12_features = scaler.fit_transform(latin1_12_eng1_12_features)

cluster_and_plot_tsne(latin12_eng12_features, topics)
cluster_and_plot_tsne(latin1_12_eng1_12_features, topics)
cluster_and_plot_tsne(latin1_eng1_features, topics)