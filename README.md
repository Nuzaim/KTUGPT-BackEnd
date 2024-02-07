# Chunking and Embedding - Flask

This is a simple flask backend server for chunking and embedding a large text or a pdf file

### Chunking

Chunking is done by [Text Splitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/split_by_token) in Langchain.

### Embedding

Embedding is done by [Sentence Transformers](https://sbert.net/)

## Installation

Install and update using pip:

```
pip install Flask
pip install pypdf2
pip install langchain
pip install sentence-transformer 

```

## How to Run

### Run in a virtual environment
1. Open a terminal in the project root directory and run:
```
python3 -m venv .venv

```
2. To open the virtual environment:
```
. .venv/bin/activate
```

1. Start the backend server:
```
flask --app app run 
```
