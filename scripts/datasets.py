import torch
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from transformers import BertConfig, BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

class NewsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        model_name="bert-base-cased",
        split='train'
    ):
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # self._config = BertConfig.from_pretrained(model_name)
        # self._bert_model = BertModel.from_pretrained(model_name, config=self._config)
        # self._bert_model.eval()
        # self._bert_tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
        self._bert_model = SentenceTransformer('all-mpnet-base-v2')
        self._data_df = pd.read_csv(f"../data/{split}_data.csv", index_col="Date")

    def __len__(self):
        return len(self._data_df.index)

    def __getitem__(self, index):
        row = self._data_df.iloc[index]
        label = row[-1]
        text_series = row[:-3]
        nan_count = text_series.isna().sum()
        day_text_matrix = np.empty((text_series.size - nan_count, 768))
        # self._bert_model = self._bert_model.to(self._device)
        for index, text in enumerate(text_series):
            if isinstance(text, str):
                # tokens = self._bert_tokenizer(text, return_tensors='pt')
                # output = self._bert_model(tokens.input_ids.to(self._device))
                # latent_matrix = output.last_hidden_state[0]
                # mean_vector = torch.mean(latent_matrix, 0)
                # mean_vector = mean_vector.to('cpu').detach().numpy()
                # mean_vector = mean_vector.reshape((1,-1))
                # day_text_matrix[index, :] = mean_vector
                sentences = [text]
                sentence_embeddings = self._bert_model.encode(sentences)
                day_text_matrix[index, :] = sentence_embeddings[0]
        return (
            torch.tensor(day_text_matrix),
            torch.tensor(label)
        )