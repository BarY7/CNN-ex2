r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers


import math


def part2_overfit_hp():
    wstd, lr, reg = 0.02 , 0.02, 0.004
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp(opt_name):
    wstd, lr, reg = 0.02 , 0.02, 0.004

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different hyperparameters for each optimizer.
    # ====== YOUR CODE: ======
    if opt_name == 'vanilla':
        lr = 1e-2
    if opt_name == 'momentum':
        lr = 1e-3 #steps are less affected by noise
    if opt_name == 'rmsprop':
        lr = 1e-4
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0.02 , 0.02
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    lr = 1e-3
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**
1. They mostly match what we expected to see
We trained the model without dropout until it overfits - achieving near perfect results on the training set but
only about 20% for the test set.
As expected, dropout helps us to avoid overfitting with 30 epochs - as soon as we turned on dropout (with both 
probability values) the model improves on the test set (it overfits less w.r.t the training set).
However dropout also makes training take more epochs - I think running with more epochs and dropout parameters
would be interesting to check how changing the dropout value affects convergence rate and the achieved test set accuracy.
2. Dropout with 0.4 achieved better result compared to the 0.8 one for 30 epochs. We know that when increasing the dropout, we require more epochs, but 
here both models train for the same amount of epochs so it makes sense that the 0.4 one got better results.
Based on the test loss graph, we can see that 0.8 is still downtrending while 0.4 starts to increase.
Given enough epochs I believe the 0.8 dropout model could pull ahead of the 0.4 one in terms of test set accuracy.



"""

part2_q2 = r"""
**Your answer:**
Yes, its possible. The Cross Entropy loss receives as arguments the class scores $\hat{y}$ and the real labels $y$.
In order to increase the loss, the score of the right class (denoted $x_y$) needs to decrease. Note that the sum-log-exp
term weights the same whether $x_y$ is the right class or not.
In order to increase the test accuracy, we need more samples such that $x_y$ is the maximum class score. 
Both can happen simultaneously - the score of the right class can decrease while still remaining the maximum score,
or it can decrease and then become the maximum score (because other classes decreased further).
In conclusion, this can happen - especially at the first few epochs before convergance is achieved.  
"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
