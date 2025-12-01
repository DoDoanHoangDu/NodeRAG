from sentence_transformers import SentenceTransformer
import torch
from answering.get_context import get_context
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("google/embeddinggemma-300m", device=device)

while True:
    question = input("Enter your medical question (or 'exit' to quit): ")
    if question.lower() == 'exit':
        break
    context = get_context(question, model, k_embedding=5, k_ppr=10, alpha=0.5, t=2)
    for key in context:
        print(f"Node ID: {key}\nContent: {context[key]}\n")
        print("-"*50)
    print("#"*50)