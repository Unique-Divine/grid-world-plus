


## Experiment 1:  

Overview:
- Transformer **as** critic -> Transformer output acts as policy function.
- MLP actor


## Experiment 2: 

Overview: 
- Transformer **helps** critic -> Transformer helps provide powerful representations, but a more traditional network outputs acts as policy function. 
- MLP actor 

## Experiment3:  Exp. 2 with deterministic critic (as opposed to network)

Actor-critic 
- Similarity metric computed by ConvNets and/or transformers
- State-value fn computed for policy
  - similarity_metric(discounted_rewards, memories)
  - Actor - MLP |  policy(state, similarity_metric) ->   $\pi_\psi(s) = a$.
- Critic - avg | judge (s_i) for each i in scene individually;

## Experiment: actor-critic | 2 Transformers

Critic | Similar/representation stuff  w ConvNets + Transformer(s)
Actor | Use the seq predictor Transformer to add on to the current seq. | get action

Upon episode completion 

## Experiment: Vanilla actor-critic for image task

This is mostly just a benchmark to compare against the attention-assisted paradigms. 

## Experiment: Prioritizing buffer w/ Transformer

Context vector from sequence predictions of the Transformer can say which states to focus on for a trajectory to gain the most information for predicting the reward.  


----

# TODOs

- [ ] (U): Implement state representations code -> with justification
- [ ] (U): Convert observation views to torch.Tensor
The goal from these two steps is to be able to pass input (representations) through our networks for testing. 

- [ ] (E): Discover/invent similarity metric for Exp. 3 (above)
- [ ] (E): Implement similiarity metric from above.
- [ ] (U): Run vanilla A-C on simple env 
- [ ] (U): Run M-CURL like  algo on simple env
- [ ] (U): Run soft A-C w/ on simple env
- [ ] (U): Write training scheme where Transformer gives argument for the Q-value calculation in a deep Q (off-policy) method such as rainbow, DDQN.
- [ ] (E): Implement the policy described by `LSTMPolicy` in the [curiosity-driven exploration](https://github.com/pathak22/noreward-rl/blob/master/src/model.py)  code in PyTorch and then use it on the env for regular policy-gradient training. Also, try to see how they prepare images as inputs. 


