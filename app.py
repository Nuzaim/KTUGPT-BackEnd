from flask import Flask, request
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer, util
import torch
from urllib.parse import unquote
model = SentenceTransformer("all-MiniLM-L6-v2")

app = Flask(__name__)

@app.route("/text", methods=["POST"])
def foo():
    query = request.query_string.decode()
    query = unquote(query)
    print(query)
    # text_data is the database in which semantic search is performed.
    text_data = """"""
    
    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator=".",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    text_data = str(text_data)
    text_data = text_data.lower()
    corpus = text_splitter.split_text(str(text_data))
    # Sentences are encoded by calling model.encode()
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

    # Find the closest 2 sentences of the corpus for each query sentence based on cosine similarity
    top_k = min(2, len(corpus))
    # Result dictionary
    result = {}
    # Encoding query
    query_embedding = model.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 2 scores
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    print("\n\n======================\n\n")
    print("Query:", query)
    result["query"] = query
    print("\nTop 2 most similar sentences in corpus:")

    result["answer"] = []
    for score, idx in zip(top_results[0], top_results[1]):
        print(corpus[idx], "(Score: {:.4f})".format(score))
        result["answer"].append(corpus[idx])
    return result 

@app.route("/pdf", methods=["POST"])
def hello_world():
    if not request.files['file']: return("NO")
    #read file
    pdf = request.files['file']
    # f.save('/home/user/Desktop/project/uploads/some.pdf')
    # return "File successfully uploaded"
    if pdf is None: return "<h1>No files</h1>"
    #read pdf
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
      text += page.extract_text()
    
    print(text)

    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)   
    return chunks
