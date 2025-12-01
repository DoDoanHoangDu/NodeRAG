import os
import pickle
import faiss
import json
import numpy as np
import pandas as pd
from graphs.Node import Node
from retrieval.ppr_local import shallow_ppr_local


#set file paths
import sys
dir_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(dir_path)
#sys.path.append(root_path)


#Load Data
#Graph - Node dict
with open(f"{root_path}/graphs/data/graphs/G5_semantically_enriched_graph.pkl", "rb") as f:
    medical_g5 = pickle.load(f)
#Embeddings
hnsw = faiss.read_index(f"{root_path}/graphs/data/embedding/medical_index.faiss")
with open(f"{root_path}/graphs/data/embedding/medical_ids.json", "r") as f:
    medical_embedding_ids = json.load(f)
num_vectors = hnsw.ntotal
dimension = hnsw.d
embeddings = np.zeros((num_vectors, dimension), dtype='float32')
for i in range(num_vectors):
    embeddings[i] = hnsw.reconstruct(i)
#Entities
with open(f"{root_path}/graphs/data/nodes/entity/medical_entities.pkl", "rb") as f:
    medical_entities = pickle.load(f)
medical_overview = pd.read_parquet(f"{root_path}/graphs/data/nodes/community/medical_communities_overview.parquet")

def find_relevant_embeddings(query_embedding, k):
    similarity, idx = hnsw.search(query_embedding,k)
    return similarity[0], idx[0]

def find_relevant_entities(query_entities):
    if isinstance(query_entities, str):
        query_set = {query_entities}
    else:
        query_set = set(query_entities) 
    entity_ids = set()
    for e in query_set:
        e = e.upper()
        if e in medical_entities:
            entity_ids.add(medical_entities[e])
        mask = medical_overview['community_overview'].str.contains(fr'\b{e}\b',case=False,na=False,regex=True) #match whole word only
        matching_node_ids = medical_overview.loc[mask, 'node_id'].tolist()
        entity_ids.update(matching_node_ids)
    return entity_ids

def retrieve_relevant_nodes(query_embedding, query_entities,k_embedding,k_ppr, alpha, t):
    entry_points = set()
    #find relevant embeddings
    sim, idx = find_relevant_embeddings(query_embedding, k_embedding)
    embedding_node_ids = [medical_embedding_ids[i] for i in idx]

    #find relevant entities
    entity_node_ids = find_relevant_entities(query_entities)

    #combine entry node ids
    entry_node_ids = set(embedding_node_ids).union(entity_node_ids)

    #perform PPR from entry nodes to find cross nodes
    if k_ppr:
        ppr_search_results = shallow_ppr_local(medical_g5, entry_node_ids, alpha=alpha, t=t, k=k_ppr)
    else:
        ppr_search_results = shallow_ppr_local(medical_g5, entry_node_ids, alpha=alpha, t=t, k=len(entry_node_ids)*5)
    cross_node_ids = set(ppr_search_results.keys())

    #combine all relevant nodes
    relevant_node_ids = entry_node_ids.union(cross_node_ids)
    #content
    content = {}
    for node_id in relevant_node_ids:
        node = medical_g5[node_id]
        if node.node_type in ['N','O']: #remove non-informative nodes
            continue
        content[node_id] = node.content
    return content

a = Node(node_type='N', content='Test Node A')
a.print()
print(shallow_ppr_local({1:a}, [1], alpha=0.5, t=2, k=10))