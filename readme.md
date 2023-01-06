Repo for reproducing [Adapting Neural Models with Sequential Monte Carlo Dropout
](https://arxiv.org/abs/2210.15779)

## TODO
- [x] Define the problem
- [x] Give naive dropout architecture
- [x] Experiment on 4.1
- [x] Reproduce 4.2
- [x] Challenge 4.3
- [x] Report
- [x] Bibliography

## Tips for usage
- Ignore the `demo` directory. It contains scratch during developpement.
- Pretraining stage takes nearly 1 hour on an GPU (RTX A2000). We recommend you to use our pretrained one.
- Adaptation is efficient thanks to vectorization, but replicate 10 times with 1000 particles on 100 tasks for 100 timesteps still takes around 40 minutes, change the settings based on your need.
- The easiest way to verify our results is to run the smcd adaptation notebook, especially the `compare_adaptaion` function, this will only run once the smcd but together with no adaptation and gradient descent.
