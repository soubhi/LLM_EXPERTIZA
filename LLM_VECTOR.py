from dotenv import load_dotenv
import os
from pymongo.mongo_client import MongoClient
import numpy as np
import torch
import faiss
import openai
from transformers import AutoTokenizer, AutoModel

load_dotenv()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


uri =os.getenv("MONGODB_URL") 

# Create a new client and connect to the server
client = MongoClient(uri)
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

db = client["expertiza-files"]
collection = db["expertiza-collection"]

CODEBASE_PATH = os.getenv("CODEBASE_PATH")
# Load CodeBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
model = AutoModel.from_pretrained("microsoft/graphcodebert-base")

def _read_code_files(base_path):
    code_files = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(('.rb')):  # Adjust file types as needed
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    content = f.read()
                    code_files.append((os.path.join(root, file), content))
    return code_files

# Function to embed code using CodeBERT
def _embed_code(content):
    # Tokenize and truncate the input to fit within the model's maximum length
    inputs = tokenizer(content, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        # Use the last hidden state and mean-pool over the token embeddings
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

def _extract_similar_code_embeddings(query_embedding, index, k=10):
    # Query FAISS for the top-k similar code embeddings
    distances, indices = index.search(query_embedding.reshape(1, -1), k=k)
    return indices, distances

def _create_prompt_from_code(indices):
    context = []
    for vectors in indices[0]:
        for vector in vectors:

            doc = collection.find_one({'_id': int(vector)})
            print(doc['path'])
            if doc:
                context.append(f"File Path: {doc['path']}\nCode Snippet:\n{doc['content']}\n")
        
    prompt = "\n---\n".join(context) + "\nAnalyze the relationships such as class inheritance, method calls, and imports in the provided code and how they can be improved in terms of code design, structure, and coupling. Give code snippet examples in response."
    return prompt

def _analyze_code_relationships_with_gpt4(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt + "please give code snippet examples"}]
    )
    return response['choices'][0]['message']['content']

def _analyze_code_relationships_with_code_snips(prompt_text):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt_text,
        max_tokens=150,  # Adjust as needed
        temperature=0.2  # Lower values for more deterministic outputs
    )
    return response.choices[0].text.strip()

# Main script to read, embed, and store vectors
def main():
    code_files = _read_code_files(CODEBASE_PATH)
    vectors = []
    metadata = []


    for idx, (path, content) in enumerate(code_files):
        print(f"Embedding: {path}")
        embedding = _embed_code(content)
        vectors.append(embedding)
        metadata.append({
            '_id': idx,
            'path': path,
            'content': content[:500] + '...' if len(content) > 500 else content  # Store a snippet for reference
        })

    # Convert vectors to a numpy array for FAISS
    vectors = np.array(vectors).astype('float32')

    collection.delete_many({})
 
    # Store metadata in MongoDB
    for meta in metadata:
        collection.insert_one(meta)

    # Create a FAISS index
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    
    # Save the FAISS index to a file (optional)
    faiss.write_index(index, 'expertiza_code_vectors.index')

    print(f"Stored {len(vectors)} vectors and metadata.")

    # Example query embedding
    query_text = "Locate parent-child class relationships"
    query_embedding = _embed_code(query_text)

    # Load FAISS index
    index = faiss.read_index('expertiza_code_vectors.index')

    # Extract similar code embeddings
    indices = _extract_similar_code_embeddings(query_embedding, index, k=5)

    # Create a prompt and analyze code relationships with GPT-4
    prompt = _create_prompt_from_code(indices)
    analysis_result = _analyze_code_relationships_with_gpt4(prompt)

    print("Analysis Result from GPT-4:")
    print(analysis_result)


    file_path = ".../LLM_output.txt"

    content = analysis_result

    with open(file_path, 'w') as file:
        file.write(content)
    print("wrote to file")


if __name__ == "__main__":
    main()
