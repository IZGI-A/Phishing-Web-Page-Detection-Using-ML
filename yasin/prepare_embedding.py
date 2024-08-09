import codecs
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from multiprocessing import Pool

import numpy as np
import torch
import trafilatura
from googletrans import Translator
from langdetect import detect
from sentence_transformers import SentenceTransformer
from transformers import ElectraModel, ElectraTokenizer


def embed_with_electra(x):
    device = torch.device("cuda")

    model_name = "google/electra-base-discriminator"
    tokenizer = ElectraTokenizer.from_pretrained(model_name)
    model = ElectraModel.from_pretrained(model_name)
    model = model.to(device)

    if isinstance(x, str):
        x = [x]

    all_embeddings = []
    chunk_size = 100  # If your computer locked due to GPU memory error, consider reducing this value

    length = len(x)
    for i in range(0, length, chunk_size):
        inputs = tokenizer(x[i: i + chunk_size], return_tensors="pt", padding=True, truncation=True)
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        embedding_chunk = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

        all_embeddings.extend([embedding_chunk[j, :] for j in range(embedding_chunk.shape[0])])
        torch.cuda.empty_cache()

    return np.vstack(all_embeddings)


BERT_BASE_MODEL = lambda x: SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens',
                                                device='cuda').encode(x)
ELECTRA_MODEL = lambda x: embed_with_electra(x)
XLM_ROBERTA_MODEL = lambda x: SentenceTransformer('aditeyabaral/sentencetransformer-xlm-roberta-base',
                                                  device='cuda').encode(x)


MODEL_MAP = {
    "xlm-roberta": XLM_ROBERTA_MODEL,
    "sbert": BERT_BASE_MODEL,
    "electra": ELECTRA_MODEL
}

TRANSLATION_MAP = {
    "xlm-roberta": False,
    "sbert": True,
    "electra": True
}

TIMES = 0
SIZE = 1


def embed(text, model_name: str):
    return MODEL_MAP[model_name](text)


def check_file(file_path):
    if not os.path.exists(file_path):
        with open(file_path, "a") as _:
            pass
    return None


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return None


def serialize(obj, path: str):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def deserialize(path: str):
    with open(path, 'rb') as f:
        temp = pickle.load(f)
    return temp


def translate(html_content):
    html_content = html_content.strip("|\n").replace("\n", " . ")
    translator = Translator()
    translated_content = ""
    translated_content += translator.translate(html_content[:5000], 'en').text
    return translated_content


def extract_text(html_file_path, translated_path, model_name, cache:bool = True):

    if not TRANSLATION_MAP[model_name]:
        with codecs.open(html_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    if cache and os.path.exists(translated_path):
        with codecs.open(translated_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    with codecs.open(html_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        html_content = f.read()
    try:
        is_text_eng = detect(html_content) == "en"
    except Exception:
        is_text_eng = False

    if TRANSLATION_MAP[model_name] and not is_text_eng:
        html_content = translate(html_content)
        if cache:
            with codecs.open(translated_path, 'w', encoding='utf-8', errors='ignore') as f:
                f.write(html_content)

    return html_content


def translate_and_append_into(file_name: str, source_path: str, append_into: list, model_name, cache=True):
    translated_path = source_path + "_Translated"
    if cache:
        check_path(translated_path)

    global TIMES
    TIMES += 1
    print("\r" + str(TIMES) + "/" + str(SIZE) + ": %" + str(int(TIMES * 100 / SIZE)) + " - " + source_path
          + "/" + file_name, end="", flush=True)
    text = extract_text(source_path + "/" + file_name, translated_path + "/" + file_name, model_name, cache=cache)

    if text is None:
        print("This is None")
        return False

    append_into.append(text)
    return True


def translate_parallel(source_path: str, model_name):
    site_contents = []
    partial_translate_append = partial(translate_and_append_into, source_path=source_path,
                                       append_into=site_contents, model_name=model_name)

    files = os.listdir(source_path)
    global SIZE
    SIZE = len(files)

    start = time.time()
    with ThreadPoolExecutor(max_workers=10) as exe:
        result = exe.map(partial_translate_append, files)

    print(time.time() - start)

    return site_contents


def embed_legitimates(legitimate_input, legitimate_output, model_name):
    global TIMES
    TIMES = 0
    print("Started embedding legitimate...")
    start = time.time()
    site_contents = translate_parallel(source_path=legitimate_input, model_name=model_name)
    embeds = MODEL_MAP[model_name](site_contents)
    print("Finished embedding legitimate in {} second.".format(time.time() - start))

    return embeds


def embed_phishing(phishing_input, phishing_output, model_name):
    global TIMES
    TIMES = 0
    print("Started embedding phishing...")
    start_time = time.time()
    site_contents = translate_parallel(source_path=phishing_input, model_name=model_name)
    embeds = MODEL_MAP[model_name](site_contents)
    print("Finished embedding phishing in {} second.".format(time.time() - start_time))

    return embeds


def extract_and_copy_text(file_name, source_path, target_path):
    try:
        with codecs.open(source_path + "/" + file_name, 'r', encoding='utf-8', errors='ignore') as file:
            # file = open(source_path + "/" + file_name, 'r', encoding='utf-8')
            html_content = file.read()
            file.close()
            extracted_text = trafilatura.extract(html_content)
            if extracted_text is None:
                return None
            new_html = open(target_path + "/" + file_name, 'w', encoding='utf-8')
            new_html.write(extracted_text)
            new_html.close()
            return extracted_text
    except UnicodeDecodeError:
        print("{} error".format(file_name))
        return ""


def parse_htmls(source_path):
    check_path(source_path + "_Extracted_Texts")
    with Pool() as pool:
        partial_safe_extract = partial(extract_and_copy_text, source_path=source_path,
                                       target_path=source_path + "_Extracted_Texts")

        directories = os.listdir(source_path)
        size = len(directories)

        results = pool.imap_unordered(partial_safe_extract, os.listdir(source_path), chunksize=10)
        global TIMES
        TIMES = 0
        for _ in results:
            print("\r" + str(TIMES) + "/" + str(size) + ": %" + str(int(TIMES * 100 / size)), end="", flush=True)
            TIMES += 1
    return None


def main():
    transformer = sys.argv[1]  # "sbert", "xlm-roberta" or "electra"

    source_folder_name = "PreparedData"
    target_folder_name = "embeddings"

    legitimate_input = source_folder_name + "/" + "Legitimate"
    phishing_input = source_folder_name + "/" + "Phishing"

    legitimate_output = target_folder_name + "/Legitimate" + "/legitimate_out_" + transformer + ".pkl"
    phishing_output = target_folder_name + "/" + "Phishing" + "/phishing_out_" + transformer + ".pkl"

    start = time.time()
    print("Started parsing htmls...")
    parse_htmls(legitimate_input)
    parse_htmls(phishing_input)
    print("All html files were parsed in {}.".format(time.time() - start))


    legitimate_input += "_Extracted_Texts"
    phishing_input += "_Extracted_Texts"

    check_path(target_folder_name + "/Legitimate")
    check_path(target_folder_name + "/Phishing")

    check_file(legitimate_output)
    check_file(phishing_output)

    start = time.time()
    print("Started embedding ...")
    print(f"Started legitimate at {datetime.now()}")
    legitimate_data = embed_legitimates(legitimate_input, legitimate_output, transformer)
    print(f"Finished legitimate at {datetime.now()}")

    print(f"Started phishing at {datetime.now()}")
    phishing_data = embed_phishing(phishing_input, phishing_output, transformer)
    print("Embedding finished in {}.".format(time.time() - start))

    legitimates = np.zeros(shape=(legitimate_data.shape[0]))
    phishings = np.ones(shape=(phishing_data.shape[0]))

    legitimate_data = np.column_stack((legitimate_data, legitimates))
    phishing_data = np.column_stack((phishing_data, phishings))

    merged_data = np.vstack((legitimate_data, phishing_data))

    serialize(merged_data, target_folder_name + "/embeddings-" + transformer + ".pkl")


if __name__ == '__main__':
    main()
