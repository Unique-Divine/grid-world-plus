# Experiments and Mini Projects <!-- omit in toc -->

#### Table of Contents <!-- omit in toc -->

- [Update: 2021-10-10 Discussion](#update-2021-10-10-discussion)
- [Completed](#completed)
  - [Report for Krzysztof](#report-for-krzysztof)
- [Someday/Maybe](#somedaymaybe)
  - [Someday/Maybe TODOs](#somedaymaybe-todos)

---

# Update: 2021-10-10 Discussion

Focus on 2 main areas:

1. Extend literature review with an emphasis on exploration and transfer learning
2. Implement and try out a minimum of 2 other RL algorithms to see how effective they are without transfer learning and how much TL might help. : Rainbow DQN and [other algo]
  - DDQN
  - Actor-critic

---

# Completed

## Report for Krzysztof

- Vanilla policy gradient algo REINFORCE for use on POMDP
  - Convolutional encoder for the observations
- Reward-shaping to urge exploration by punishing an agent for standing still
- Start the agent on different environments each episode to get the model to learn features relating to relative position in the environment rather than absolute position.
- Pre-train on a small, 3 by 3 environment with one goal and one hole, then transfer the agent to challenging environments

Figure 2: Plots of agent rewards per trajectory and moving average rewards
- [ ] Retrieve the data that was used to create these plots
- [ ] .

---

# Someday/Maybe

## Someday/Maybe TODOs

- [ ] (U): Implement state representations code -> with justification
- [ ] (U): Convert observation views to torch.Tensor
The goal from these two steps is to be able to pass input (representations) through our networks for testing. 
- [ ] (U): Run vanilla A-C on simple env 
- [ ] (U): Run M-CURL like  algo on simple env
- [ ] (U): Run soft A-C w/ on simple env
- [ ] (U): Write training scheme where Transformer gives argument for the Q-value calculation in a deep Q (off-policy) method such as rainbow, DDQN.

- [ ] (E): Discover/invent similarity metric for Exp. 3 (above)
- [ ] (E): Implement similiarity metric from above.
- [ ] (E): Implement the policy described by `LSTMPolicy` in the [curiosity-driven exploration](https://github.com/pathak22/noreward-rl/blob/master/src/model.py)  code in PyTorch and then use it on the env for regular policy-gradient training. Also, try to see how they prepare images as inputs. 


---

#### Experiment 1:  

Overview:
- Transformer **as** critic -> Transformer output acts as policy function.
- MLP actor

---

#### Experiment 2: 

Overview: 
- Transformer **helps** critic -> Transformer helps provide powerful representations, but a more traditional network outputs acts as policy function. 
- MLP actor 

---

#### Experiment 3:  Exp. 2 with deterministic critic (as opposed to network)

Actor-critic 
- Similarity metric computed by ConvNets and/or transformers
- State-value fn computed for policy
  - similarity_metric(discounted_rewards, memories)
  - Actor - MLP |  policy(state, similarity_metric) ->   $\pi_\psi(s) = a$.
- Critic - avg | judge (s_i) for each i in scene individually;

---

#### Actor-critic with 2 Transformers

Critic | Similar/representation stuff  w ConvNets + Transformer encoder
Actor | Use the seq predictor Transformer to add on to the current seq. | get action

Upon episode completion 

---

#### Benchmark vanilla actor-critic for image task

This is mostly just a benchmark to compare against the attention-assisted paradigms. 

---

#### Experiment: Prioritizing buffer w/ Transformer

Context vector from sequence predictions of the Transformer can say which states to focus on for a trajectory to gain the most information for predicting the reward.  

---

#### Diversify with Determinantal Point Processes



References:
- Zhang, C., Kjellstrom, H., & Mandt, S. (2017). Determinantal point processes for mini-batch diversification." *arXiv preprint arXiv:1705.00607.* [[pdf]](https://arxiv.org/pdf/1705.00607.pdf)
