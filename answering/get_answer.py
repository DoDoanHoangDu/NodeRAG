import time
start = time.time()
import os
from sentence_transformers import SentenceTransformer
import torch
from google import genai
from answering.get_context import get_context
from answering.answer_prompt import answer_prompt
# Load embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("google/embeddinggemma-300m", device=device)

#LLM api call
dir_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(dir_path)
with open(f"{root_path}/API_KEY.txt", "r", encoding="utf-8") as f:
    API_KEY = f.read()

def call_gemini(text):
    client = genai.Client(api_key = API_KEY)
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=text
    )
    return response.text

print(f"Load time: {time.time() - start:.2f} seconds.")
#questioning loop
loop_sep = "#"*100 + "\n"
while True:
    try:
        question = input("Enter your medical question (or 'quit' to quit): ")
        print("-"*100)
        start = time.time()
        if question.lower() == 'quit':
            print(loop_sep)
            break
        context = get_context(question, model, k_embedding=8, k_ppr=None, alpha=0.5, t=2)
        #for key in context:
        #    print(f"Node ID: {key}\nContent: {context[key]}\n")
        #    print("-"*100)
        finish_retrieval_time = time.time()
        print(f"Total retrieval time: {finish_retrieval_time - start:.2f} seconds.")
        print("Number of retrieved context nodes:", len(context))
        print("-"*100)
        full_context = "\n\n".join(context.values())
        prompt = answer_prompt(full_context, question)
        answer = call_gemini(prompt)
        print("Answer:")
        print(answer)
        print("-"*100)
        print(f"Answer generation time: {time.time() - finish_retrieval_time:.2f} seconds.")
        print(f"Total time taken: {time.time() - start:.2f} seconds.")
    except Exception as e:
        print("An error occurred:", str(e))
    print(loop_sep)