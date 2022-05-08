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
    lr = 0.0085
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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



Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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
