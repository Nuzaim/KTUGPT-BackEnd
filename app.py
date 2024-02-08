from flask import Flask, request
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

app = Flask(__name__)

@app.route("/text", methods=["POST"])
def foo():
    text_data = request.data
    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    text_data = str(text_data)
    text_data = text_data.lower()
    chunks = text_splitter.split_text(str(text_data)) 
    # Sentences are encoded by calling model.encode()
    embeddings = model.encode(chunks)

    # Print the embeddings
    for sentence, embedding in zip(chunks, embeddings):
        print("Sentence:", sentence)
        print("Embedding:", embedding)
        print("")
    return embeddings.tolist() #return 0th element because tolist() converts to list expecting the ndarray to be 2d

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
