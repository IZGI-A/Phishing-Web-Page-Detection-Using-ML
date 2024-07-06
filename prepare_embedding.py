import os
import sys
import pickle
import numpy as np
import torch
from transformers import XLMRobertaModel, XLMRobertaTokenizer
from sentence_transformers import SentenceTransformer
from translate import Translator
import trafilatura as tr


class TextExtractor:
    def __init__(self, benign_mislead_path, phishing_path):
        self.benign_mislead_path = benign_mislead_path
        self.phishing_path = phishing_path
        self.translator = Translator(to_lang="en")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def extract_text_from_files(self):
        os.makedirs('extracted_benign_misleading_text', exist_ok=True)
        os.makedirs('extracted_phishing_text', exist_ok=True)
        self._extract_text(self.benign_mislead_path, 'extracted_benign_misleading_text')
        self._extract_text(self.phishing_path, 'extracted_phishing_text')

    def _extract_text(self, source_path, target_path):
        files = os.listdir(source_path)
        for file in files:
            try:
                with open(os.path.join(source_path, file), 'r', encoding='utf-8') as f:
                    html = f.read()
            except UnicodeDecodeError:
                with open(os.path.join(source_path, file), 'r', encoding='windows-1256') as f:
                    html = f.read()

            text = tr.extract(html)
            if text:
                with open(os.path.join(target_path, f"{os.path.splitext(file)[0]}.txt"), 'w', encoding='utf-8') as f:
                    f.write(text)

    def translate_text(self):
        self._translate_folder('extracted_benign_misleading_text')
        self._translate_folder('extracted_phishing_text')

    def _translate_folder(self, path):
        os.makedirs(f'translated_text/{path}', exist_ok=True)
        files = os.listdir(path)
        for file in files:
            with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
                text = f.read()
            translated_text = self._translate_large_text(text)
            with open(f'translated_text/{path}/{file}', 'w', encoding='utf-8') as f:
                f.write(translated_text)

    def _translate_large_text(self, text, chunk_size=4000):
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        translated_chunks = [self.translator.translate(chunk) for chunk in chunks if chunk.strip()]
        return ' '.join(translated_chunks)

    def generate_embeddings(self, model_name):
        if model_name == "xlm-roberta":
            return self._generate_xlm_roberta_embeddings()
        elif model_name == "xlm-roberta-without-mean-pooling":
            return self._generate_xlm_roberta_embeddings(with_mean_pooling=False)
        elif model_name == "sbert":
            return self._generate_sbert_embeddings()
        else:
            raise ValueError("Unsupported model name")

    def _generate_xlm_roberta_embeddings(self, with_mean_pooling=True):
        model_name = 'xlm-roberta-base'
        model = XLMRobertaModel.from_pretrained(model_name).to(self.device)
        tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
        benign_embeddings = self._get_transformers_embeddings('extracted_benign_misleading_text', model, tokenizer, with_mean_pooling)
        phishing_embeddings = self._get_transformers_embeddings('extracted_phishing_text', model, tokenizer, with_mean_pooling, label=1)
        return np.vstack((benign_embeddings, phishing_embeddings))

    def _generate_sbert_embeddings(self):
        self.translate_text()
        model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens').to(self.device)
        benign_embeddings = self._get_sbert_embeddings('translated_text/extracted_benign_misleading_text', model)
        phishing_embeddings = self._get_sbert_embeddings('translated_text/extracted_phishing_text', model, label=1)
        return np.vstack((benign_embeddings, phishing_embeddings))

    def _get_transformers_embeddings(self, path, model, tokenizer, with_mean_pooling=True, label=0):
        embeddings = []
        files = os.listdir(path)
        for file in files:
            with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
                text = f.read()
            if not text.strip():
                continue
            encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.device)
            with torch.no_grad():
                model_output = model(**encoded_input)
                if with_mean_pooling:
                    embedding = self.mean_pooling(model_output, encoded_input['attention_mask']).cpu().numpy()
                else:
                    embedding = model_output.last_hidden_state[:, 0, :].cpu().numpy()  # CLS token
            embeddings.append(np.hstack((embedding, np.full((embedding.shape[0], 1), label))))
        return np.vstack(embeddings) if embeddings else np.array([]).reshape(0, 769)

    def _get_sbert_embeddings(self, path, model, label=0):
        embeddings = []
        files = os.listdir(path)
        for file in files:
            with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
                text = f.read()
            if not text.strip():
                continue
            embedding = model.encode(text).reshape(1, -1)
            embeddings.append(np.hstack((embedding, np.full((embedding.shape[0], 1), label))))
        return np.vstack(embeddings) if embeddings else np.array([]).reshape(0, 769)

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python prepare_embedding.py <path to the benign and misleading html files> <path to the phishing html files>")
        sys.exit(1)

    benign_mislead_path, phishing_path = sys.argv[1], sys.argv[2]
    extractor = TextExtractor(benign_mislead_path, phishing_path)
    extractor.extract_text_from_files()

    for model_name in ["xlm-roberta-without-mean-pooling", "xlm-roberta", "sbert"]:
        embeddings = extractor.generate_embeddings(model_name)
        with open(f'embeddings/embeddings-{model_name}.pkl', 'wb') as file:
            pickle.dump(embeddings, file)
