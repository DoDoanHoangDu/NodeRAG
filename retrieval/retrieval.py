import os

def shallow_ppr_local(nodes_dict, entry_id, alpha = 0.5, t = 2, k = 10):
    #simulate a random walk with restarts, number of steps t, probability to stop at each node after stepping is alpha
    pi = dict()   # PPR scores: probability that the walk ends at each node
    r = {entry_id: 1.0}  # probability that the next step move to the node. at step 0, probability is 1 to move to entry node

    for _ in range(t): 
        r_next = dict()
        for node_id, residual in r.items(): #step to next node if probability residual
            pi[node_id] = pi.get(node_id, 0) + alpha * residual #increase PPR score by the probability of stopping here after step
            push_val = (1 - alpha) * residual #probability to continue walking (the remaining probability)
            node = nodes_dict[node_id]
            total_weight = node.degree
            if total_weight == 0: #stop if no neighbors (won't happen in undirected graph)
                continue
            for nbr_id, w in node.edges.items(): 
                r_next[nbr_id] = r_next.get(nbr_id, 0) + push_val * (w / total_weight) #probability to move to the neighbor using edge weight
        r = r_next
    #add remaining residual probabilities to PPR scores
    for node_id, residual in r.items():
        pi[node_id] = pi.get(node_id, 0) + residual
    top_nodes = sorted(pi.items(), key=lambda x: x[1], reverse=True)[:k]
    return dict(top_nodes)



dir_path = os.getcwd()
print("The directory of this script is:", dir_path)
root_path = os.path.dirname(dir_path)
print("The root directory is:", root_path)