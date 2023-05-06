import os
import re
import shutil
import jieba
import urllib.request
from pathlib import Path
from tempfile import NamedTemporaryFile

import fitz
import numpy as np
import openai
import tensorflow_hub as hub
from fastapi import UploadFile
from lcserve import serving
from sklearn.neighbors import NearestNeighbors


recommender = None


def download_pdf(url, output_path):
    urllib.request.urlretrieve(url, output_path)


def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text


def pdf_to_text(path, start_page=1, end_page=None):
    doc = fitz.open(path)
    total_pages = doc.page_count

    if end_page is None:
        end_page = total_pages

    text_list = []

    for i in range(start_page - 1, end_page):
        text = doc.load_page(i).get_text("text").encode('utf-8').decode('utf-8')
        text = preprocess(text)
        text_list.append(text)

    doc.close()
    return text_list



def text_to_chunks(texts, word_length=150, start_page=1):
    text_toks = [list(jieba.cut(t)) for t in texts]
    page_nums = []
    chunks = []

    for idx, words in enumerate(text_toks):
        for i in range(0, len(words), word_length):
            chunk = words[i : i + word_length]
            if (
                (i + word_length) > len(words)
                and (len(chunk) < word_length)
                and (len(text_toks) != (idx + 1))
            ):
                text_toks[idx + 1] = chunk + text_toks[idx + 1]
                continue
            chunk = ''.join(chunk).strip()
            chunk = f'[Page no. {idx+start_page}]' + ' ' + '"' + chunk + '"'
            chunks.append(chunk)
    return chunks

class SemanticSearch:
    def __init__(self):
        self.use = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
        self.fitted = False

    def fit(self, data, batch=1000, n_neighbors=5):
        self.data = data
        self.embeddings = self.get_text_embedding(data, batch=batch)
        n_neighbors = min(n_neighbors, len(self.embeddings))
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        self.nn.fit(self.embeddings)
        self.fitted = True

    def __call__(self, text, return_data=True):
        inp_emb = self.use([text])
        neighbors = self.nn.kneighbors(inp_emb, return_distance=False)[0]

        if return_data:
            return [self.data[i] for i in neighbors]
        else:
            return neighbors

    def get_text_embedding(self, texts, batch=1000):
        embeddings = []
        for i in range(0, len(texts), batch):
            text_batch = texts[i : (i + batch)]
            emb_batch = self.use(text_batch)
            embeddings.append(emb_batch)
        embeddings = np.vstack(embeddings)
        return embeddings


def load_recommender(path, start_page=1):
    global recommender
    if recommender is None:
        recommender = SemanticSearch()

    texts = pdf_to_text(path, start_page=start_page)
    chunks = text_to_chunks(texts, start_page=start_page)
    recommender.fit(chunks)
    return 'Corpus Loaded.'


def generate_text(openAI_key, prompt, engine="text-davinci-003"):
    openai.api_key = openAI_key
    completions = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
    )
    message = completions.choices[0].text
    return message


def generate_answer(question, openAI_key):
    topn_chunks = recommender(question)
    prompt = ""
    prompt += '搜索结果：\n\n'
    for i, c in enumerate(topn_chunks):
        # 添加页码标识，i+1是因为Python列表从0开始计数，而这里要从1开始计数
        prompt += f'[第{i+1}页]\n\"{c}\"\n\n'


    prompt += (
        "说明：请使用给定的搜索结果撰写全面的回答。"
        "使用[页码]符号引用每个参考文献（每个结果在开头都有这个编号）。 "
        "每个句子都应该在结尾处加上引用。如果搜索结果提到了多个具有相同名称的主题，请为每个主题创建单独的答案。 "
        "仅包括在结果中找到的信息，不要添加任何额外的信息。"
        "确保答案是正确的，不要输出虚假内容 "
        "如果文本与查询无关，请简单说明“未在PDF中找到文本”。"
        "忽略与问题无关的离群值搜索结果。只回答被问到的问题。"
        "答案应该简短明了。分步回答。 \n\n查询: {question}\n答案: "
    )
    # 修改prompt，使其能够正确引用中文
    prompt += f"查询: {question}\n答案:"
    answer = generate_text(openAI_key, prompt, "text-davinci-003")
    return answer


def load_openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key is None:
        raise ValueError(
            "[ERROR]: Please pass your OPENAI_API_KEY. Get your key here : https://platform.openai.com/account/api-keys"
        )
    return key


@serving
def ask_url(url: str, question: str):
    download_pdf(url, 'corpus.pdf')
    load_recommender('corpus.pdf')
    openAI_key = load_openai_key()
    return generate_answer(question, openAI_key)


@serving
async def ask_file(file: UploadFile, question: str) -> str:
    suffix = Path(file.filename).suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    load_recommender(str(tmp_path))
    openAI_key = load_openai_key()
    return generate_answer(question, openAI_key)