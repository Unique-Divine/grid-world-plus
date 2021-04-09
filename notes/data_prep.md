



## Data preparation from M-CURL paper

Image based RL tasks can be formulated as partially observed Markov decision processes (POMDPs), which are described by the tuple $(\mathcal{O}, \mathcal{A}, p, r, \gamma)$
- $\mathcal{O}$ represents observations, a collection of images rendereed from the environment. 
- $\mathcal{A}$ is the action space. 
Hence, $o_t \in \mathcal{O}$ represents the rendered image at time $t$ and $a_t\in \mathcal{A}$ denotes the action. 

Common practice (Mnih et al., 2015):
Stack $K (>1)$ consecutive observations as the input to capture more information. This sequence of observations makes a state, $s_t = (o_{t-K+1}, o_{t-K+2}, \cdots, o_t)$. The collection of all states $s_t$ is denoted by $\mathcal{S}$. The ConvNet encoder maps states $s\in S$ into a $d$-dimensional representation: 
$$\text{Conv}_\theta (s_t) $$


Replay Buffer 

