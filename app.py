import streamlit as st
import pandas as pd
import tqdm

import torch
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
from transformers import AutoTokenizer, AutoModel

# モデルとトークナイザーの初期化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
model = AutoModel.from_pretrained("intfloat/multilingual-e5-small").to(device).eval()

sentences_df = pd.read_csv('road_trafic_law_embeddins.csv')
for i in range(0, len(sentences_df['embedding'])):
    sentences_df['embedding'][i] = torch.tensor([float(x) for x in sentences_df['embedding'][i].split(', ')])
passage_embeddings = torch.stack(sentences_df['embedding'].tolist()).squeeze(1)

# 埋め込み計算用の関数
def get_embedding(text, type='query'):
    input_texts = ''
    if type == 'query':
        input_texts = type + ' ' + text
    elif type == 'passage':
        input_texts = type + ' ' + text
    
    batch_dict = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**batch_dict)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    embeddings = F.normalize(embeddings, p=2, dim=1).detach().cpu()
    return embeddings

# テキスト入力
st.title('道路交通法検索')
st.write('道路交通法に関する文章を入れると、関連する法律を5件検索します。')
query = st.text_input('道路交通法に関する文章を入力してください。')

# ボタン押下時の処理
# 探索中は、プログレスバーを表示
if st.button('Search'):
    query_embedding = get_embedding(query, type='query')
    similarity_scores = cosine_similarity(query_embedding, passage_embeddings)
    sentences_df['similarity'] = similarity_scores
    top_5_df = sentences_df.sort_values('similarity', ascending=False).head(5)
    st.write(top_5_df[['similarity', 'text', 'title']])
