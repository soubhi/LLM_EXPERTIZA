# LLM Expertiza Vectorizer

## Overview
This project is designed to read, embed, and analyze code files using CodeBERT and other machine learning tools. The embeddings and metadata are stored in MongoDB, and FAISS is used for similarity search. The project also leverages OpenAI's API for analyzing code relationships and structure.

## Features
- **Code Embedding with CodeBERT**: Tokenizes and embeds code using `microsoft/graphcodebert-base`.
- **MongoDB Integration**: Stores code metadata and embeddings for efficient querying.
- **FAISS Integration**: Uses FAISS to search for similar code embeddings.
- **OpenAI API Integration**: Generates insights about code relationships and design using GPT-4.

## Requirements
The following Python libraries are required to run the project:
- `pymongo==4.5.0`
- `numpy==1.24.3`
- `torch==2.0.1`
- `faiss-cpu==1.7.3`  *(Use `faiss-gpu` if GPU support is needed)*
- `openai==0.11.0`
- `transformers==4.33.0`
- `python-dotenv`

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/LLM_Expertiza_Vectorizer.git
   cd LLM_Expertiza_Vectorizer
2. **Requirement.txt install**:
    ```bash 
    pip install -r requirements.txt          
3. **setup environment file**:
    ```
    OMP_NUM_THREADS=1
    OPENAI_API_KEY={API Key}
    MONGODB_URL={MongoDB connection url}
    CODEBASE_PATH={path to codebase repository}
4. **run the script**:
     ```bash
     C:.../anaconda3/envs/pyml/python.exe .../LLM_EXPERTIZA/LLM_VECTOR.py