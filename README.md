# ['Expand Neurons, Not Parameters'](https://openreview.net/forum?id=cYxavyv2C7) - ICML 2026
Here we include the code for our paper [Expand Neurons, Not Parameters](https://openreview.net/forum?id=cYxavyv2C7).

![Neuronal expansion framework](assets/Overview.png)
*Paradigms of parameter efficiency in training and inference. ''Train-then-sparsify'' minimizes post-training inference cost, at the expense of training a large, dense model initially and of sparse fine-tuning. ''Train-then-grow'' amortizes some training cost via bootstrapping from a small, dense network, to yield a large, dense network that is more expensive to inference, though some techniques are constrained by final size. Here, we show theoretical justification and empirical feasibility of directly splitting neurons within a small, dense network into a large, sparse one.*

### Environment Setup
1. Create a conda environment:
   ```
   conda env create -f environment.yml
   ```
2. Install pip dependencies:
   ```
   conda activate expand-neurons
   pip install -r requirements.txt
   ```