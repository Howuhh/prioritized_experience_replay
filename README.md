# Prioritized Experience Replay

Simple and straightforward implementation with comments. 

`SumTree` unlike other python implementations, is implemented without recursion, 
which is nearly twice faster (based on a couple of tests in ipython).   
`train.py` also implements a simple DQN algorithm to validate PER. 

## Results
10 seeds, same hyperparameters, not tuned

<p float="left">
  <img src="/plots/cartpole.jpg" width="40%"/>
  <img src="/plots/lunarlander.jpg" width="40%"/>
</p>

## TODO

- add comments and references to paper sections

# References

Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). Prioritized Experience Replay. ArXiv:1511.05952 [Cs]. http://arxiv.org/abs/1511.05952
