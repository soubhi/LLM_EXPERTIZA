import os
import subprocess
import numpy as np
import torch
import faiss
import openai
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from transformers import AutoTokenizer, AutoModel
from urllib.parse import urlparse

load_dotenv()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

uri = os.getenv("MONGODB_URL") 
client = MongoClient(uri)
db = client["expertiza-files"]
collection = db["expertiza-collection"]

tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
model = AutoModel.from_pretrained("microsoft/graphcodebert-base")

def _get_repo_info_from_pr_url(pr_url):
    """
    Extracts the repository owner, name, and pull request number from a GitHub PR URL.
    Example URL: https://github.com/owner/repo/pull/42
    """
    parsed_url = urlparse(pr_url)
    path_parts = parsed_url.path.strip("/").split("/")
    
    if len(path_parts) < 4 or path_parts[2] != "pull":
        raise ValueError("Invalid GitHub PR URL. Expected format: https://github.com/owner/repo/pull/42")
    
    owner, repo, _, pr_number = path_parts
    return owner, repo, pr_number

def _clone_or_update_repo(pr_url, base_path="temp_codebase"):
    """
    Clones or fetches the latest code for a GitHub PR.
    """
    owner, repo, pr_number = _get_repo_info_from_pr_url(pr_url)
    repo_url = f"https://github.com/{owner}/{repo}.git"
    repo_path = os.path.join(base_path, repo)

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    if os.path.exists(repo_path):
        # If repo exists, ensure we are on a branch like 'main' before fetching
        subprocess.run(["git", "-C", repo_path, "checkout", "main"], check=True)  # Switch to main branch
        subprocess.run(["git", "-C", repo_path, "fetch", "origin", f"pull/{pr_number}/head:pr-{pr_number}"], check=True)
        subprocess.run(["git", "-C", repo_path, "checkout", f"pr-{pr_number}"], check=True)
    else:
        # Clone repo and checkout PR branch
        subprocess.run(["git", "clone", repo_url, repo_path], check=True)
        subprocess.run(["git", "-C", repo_path, "fetch", "origin", f"pull/{pr_number}/head:pr-{pr_number}"], check=True)
        subprocess.run(["git", "-C", repo_path, "checkout", f"pr-{pr_number}"], check=True)

    return repo_path, pr_number

def _get_changed_files(repo_path):
    """
    Get the list of files changed in the PR compared to the point it branched from `main`.
    """
    try:
        # Find the commit where the PR branch was created from main
        base_commit_result = subprocess.run(
            ["git", "-C", repo_path, "merge-base", "origin/main", "HEAD"],
            capture_output=True, text=True, check=True
        )
        base_commit = base_commit_result.stdout.strip()

        # Get the list of files changed in the PR since that commit
        result = subprocess.run(
            ["git", "-C", repo_path, "diff", "--name-only", base_commit, "HEAD"],
            capture_output=True, text=True, check=True
        )
        
        changed_files = result.stdout.strip().split("\n")
        return [f for f in changed_files if f.endswith(".rb")]  # Process only Ruby files
    except subprocess.CalledProcessError as e:
        print("Error fetching changed files:", e)
        return []

### Read only changed files ###
def _read_code_files(base_path):
    """
    Reads only the files that were changed in the PR.
    """
    changed_files = _get_changed_files(base_path)
    if not changed_files:
        print("No changed files detected in the PR.")
        return []

    code_files = []
    for file in changed_files:
        file_path = os.path.join(base_path, file)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                code_files.append((file_path, content))
    
    return code_files

### Embed code using CodeBERT ###
def _embed_code(content):
    inputs = tokenizer(content, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

### Search similar embeddings ###
def _extract_similar_code_embeddings(query_embedding, index, k=10):
    distances, indices = index.search(query_embedding.reshape(1, -1), k=k)
    return indices, distances

### Create prompt for GPT-4 ###
def _create_prompt_from_code(indices):
    context = []
    for vectors in indices[0]:
        for vector in vectors:
            doc = collection.find_one({'_id': int(vector)})
            if doc:
                context.append(f"File Path: {doc['path']}\nCode Snippet:\n{doc['content']}\n")

    prompt = "\n---\n".join(context) + "\nAnalyze class inheritance, method calls, and imports in the provided code and suggest improvements."
    return prompt

### Analyze code relationships using GPT-4 ###
def _analyze_code_relationships_with_gpt4(prompt):
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt + " please provide code snippet examples"}]
    )
    return response.choices[0].message.content

### Main Function ###
def main(pr_url=None):
    if pr_url:
        print(f"Processing GitHub Pull Request: {pr_url}")
        repo_path, pr_number = _clone_or_update_repo(pr_url)
    else:
        repo_path = os.getenv("CODEBASE_PATH")

    code_files = _read_code_files(repo_path)

    if not code_files:
        print("No valid changed Ruby files found in the PR.")
        return  # Exit early

    vectors = []
    metadata = []

    for idx, (path, content) in enumerate(code_files):
        print(f"Embedding: {path}")
        embedding = _embed_code(content)

        if embedding is None or embedding.shape == ():
            print(f"Skipping {path}: Failed to generate an embedding.")
            continue

        vectors.append(embedding)
        metadata.append({
            '_id': idx,
            'path': path,
            'content': content[:500] + '...' if len(content) > 500 else content
        })

    if not vectors:
        print("No valid embeddings were generated. Exiting.")
        return  # Exit early

    vectors = np.array(vectors).astype('float32')

    collection.delete_many({})
    for meta in metadata:
        collection.insert_one(meta)

    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    faiss.write_index(index, 'expertiza_code_vectors.index')
    print(f"Stored {len(vectors)} vectors and metadata.")

    query_text = "Locate parent-child class relationships"
    query_embedding = _embed_code(query_text)

    index = faiss.read_index('expertiza_code_vectors.index')
    indices = _extract_similar_code_embeddings(query_embedding, index, k=5)
    prompt = _create_prompt_from_code(indices)
    analysis_result = _analyze_code_relationships_with_gpt4(prompt)

    print("Analysis Result from GPT-4:")
    print(analysis_result)

    output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, f"LLM_output_PR_{pr_number}.txt")
    with open(file_path, 'w') as file:
        file.write(analysis_result)

    print(f"Output written to {file_path}")

if __name__ == "__main__":
    import sys
    pr_url = sys.argv[1] if len(sys.argv) > 1 else None
    main(pr_url)
