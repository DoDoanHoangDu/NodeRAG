from answering.question_decompose_prompt import question_decompose_prompt
from retrieval.retrieval import retrieve_relevant_nodes
from google import genai
import os
import json
import numpy as np

#set file paths
dir_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(dir_path)

#LLM api call
with open(f"{root_path}/API_KEY.txt", "r", encoding="utf-8") as f:
    API_KEY = f.read()

def call_gemini(text):
    client = genai.Client(api_key = API_KEY)
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=text,
        config={ "response_mime_type": "application/json"}
    )
    return response.text

def get_context(query, embedding_model, k_embedding, k_ppr, alpha,t):
    #Decompose question to extract entities
    prompt = question_decompose_prompt(query)
    response_text = call_gemini(prompt)
    print("Decomposed Question Response:", response_text)
    query_entities = json.loads(response_text)
    
    #Get query embedding
    query_embedding = embedding_model.encode([query], convert_to_numpy=True).astype(np.float32)

    #Retrieve relevant nodes
    context = retrieve_relevant_nodes(query_embedding, query_entities, k_embedding, k_ppr, alpha, t)
    return context

