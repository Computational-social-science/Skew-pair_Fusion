from tqdm import tqdm
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel

model = AutoModelForSequenceClassification.from_pretrained("yourModel/skew-x/bert_8.0_epoch", output_hidden_states=True)

for i in range(1, 13):
    labels_list = []
    hidden_states_list = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for example in tqdm(tokenized_test_dataset):
        input_ids = example['input_ids']
        attention_mask = example['attention_mask']
        input_values = example['input_values']
        attention_mask_audio = example['attention_mask_audio']

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        input_values = input_values.to(device)
        attention_mask_audio = attention_mask_audio.to(device)

        outputs = model(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
        hidden_states = outputs.hidden_states

        layer_index = i
        selected_layer = hidden_states[layer_index][:, 0, :].detach().cpu().numpy()

        labels_list.append(example['labels'])
        hidden_states_list.append(selected_layer)
        del outputs, selected_layer

    labels_tensor = torch.tensor(labels_list)
    hidden_states_tensor = torch.tensor(hidden_states_list)
    hidden_states_tensor = hidden_states_tensor.squeeze(1)

    tsne = TSNE(n_components=2, perplexity=150.0, random_state=42)
    tsne_vectors = tsne.fit_transform(hidden_states_tensor)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tsne_vectors[:, 0], tsne_vectors[:, 1], c=labels_tensor, cmap='rainbow', s=8)
    plt.title("t-SNE Visualization with Labels (Layer {})".format(layer_index))
    plt.colorbar(scatter).remove()
    plt.savefig(f'{i}.png')