**Task:** Assume you start with no labeled data. Build a classifier that achieves 90% accuracy on the test set using the smallest number of queries to a simulated human annotator database which contains labeled data.
**Dataset:** [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)  

## Resources
- [Learning with Limited Labeled Data](https://lwlld.fastforwardlabs.com/)
- https://towardsdatascience.com/active-learning-getting-the-most-out-of-limited-data-16b472f25370
- https://towardsdatascience.com/how-to-get-away-with-few-labels-label-propagation-f891782ada5c
- https://towardsdatascience.com/active-learning-on-mnist-saving-on-labeling-f3971994c7ba

# Setup
```
conda create -n jax-fashion python=3.9
conda activate jax-fashion
pip install -r requirements.txt
```

# Strategy
1. Select a subset of train data to be labeled at initialization
   1. Baseline: random sampling, n=100
2. Train classifier on labeled subset
3. Predict on remaining points in train data
4. Predict on held-out test set 
5. Choose unlabeled points which are added to labeled subset
   1. Baseline: random sampling, n=50
6. Repeat 2 -> 5 until 90% accuracy on held-out test set