# Prioritized Experience Replay

Simple and straightforward implementation with comments

## Results

### DQN with PER

10 seeds, same hyperparameters, not tuned

<p float="left">
  <img src="/plots/cartpole.jpg" width="50%"/>
  <img src="/plots/lunarlander.jpg" width="50%"/>
</p>

### Sampling approaches

In the original publication, a particular method of prioritized sampling using SumTree is proposed, as follows: 

> To sample a minibatch of size k, the range [0, p_total] is divided equally into k ranges. Next, a value is uniformly 
> sampled from each range. Finally the transitions that correspond to each of these sampled values are retrieved from the tree.

However, there is no justification in the publication as to why this sampling method was proposed. Samples with higher 
priority correspond to a larger interval in [0, p_total], so it should be enough to sample k values from range 
[0, p_total] uniformly, which is simpler to code. 

As the graph below shows, there is no particular difference in the distribution of priorities among the two sampling methods. 

![sampling](plots/sampling_approaches.jpg)

To reproduce, run 
```bash
python plots/sampling.py
```

## TODO

- add comments and references to paper sections

# References

Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). Prioritized Experience Replay. ArXiv:1511.05952 [Cs]. http://arxiv.org/abs/1511.05952
