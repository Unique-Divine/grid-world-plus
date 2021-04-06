
# Algorithm 1 from https://openreview.net/pdf?id=HkxjqxBYDB

import numpy as np

graph: np.ndarray = np.empty(1)


# TODO: Embedded state vector, h = φ(s) 

# TODO: graph ← Sort nodes in graph by sequential step ID, t

def get_current_sa_pair(graph):
    pass 
    # return s, a

def get_successor_sa_pair(graph):
    pass
    # return s_p, a_p

def update_graph_memory():
    """Use  Q_G(φ(s), a) ← r + γ max_{a'}( Q_G (φ(s'), a')) )
    r: reward
    γ: discount 
    φ: state vector. Each state is some s in S. 
    """
    #     pass
    # return Q_G

for m in graph.size:
    # TODO: get state-action tuple
     
    # TODO: get state embedding s' and action a' using graph
  
    # TODO: Update graph-augmented memory

    pass 


# TODO: 
