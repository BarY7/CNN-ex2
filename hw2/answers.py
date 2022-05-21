r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers


import math


def part2_overfit_hp():
    wstd, lr, reg = 0.15 , 0.02, 0.004
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp(opt_name):
    wstd, lr, reg = 0.1 , 0.02, 0.003

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different hyperparameters for each optimizer.
    # ====== YOUR CODE: ======
    if opt_name == 'vanilla':
        lr = 0.015
    if opt_name == 'momentum':
        lr = 0.003
    if opt_name == 'rmsprop':
        lr = 0.0002
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0.1 , 0.02
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    lr = 1e-3
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**
1. They mostly match what we expected to see.
We trained the model without dropout until it overfits - achieving near perfect results on the training set but
only about 20% for the test set.
As expected, dropout helps us to avoid overfitting with 30 epochs - as soon as we turned on dropout (with both 
probability values) the model improves on the test set (it overfits less w.r.t the training set).
Seeing as dropout aims to reach a more sparse representation of the data, I think running with more epochs and dropout parameters
would be interesting to check how changing the dropout value affects convergence rate and the achieved test set accuracy.
2. Dropout with 0.4 achieved better result compared to the 0.8 one for 30 epochs.
Intuitively, since higher dropout means we use less of our data, we should require more epochs when increasing it.
Here both models train for the same amount of epochs so it makes sense that the 0.4 one got better results.
Given enough epochs its possible that 0.8 dropout model could pull ahead of the 0.4 one in terms of test set accuracy.
We tried this expriment with 80 epochs and indeed in our case the model with 0.8 dropout got the best test set accuracy.



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
The parameters we decided on are:
batches 1000 
epochs 50 
early-stopping 2 
hidden-dims 100
learning rate = 0.001
layer 2 pool ever 1, layer 4 pool every 1, layer 8 pool ever 8, layer 16 pool ever 16.
We tried different parameters and we got the best results with those; we can see we stop approximately as the loss goes up with the early stopping, and the loss graphs seem smooth so the learning rate is appropriate. We put 50 epochs because we counted on the early stopping to regulate and stop the learning process. 
We chose batches 1000 so we can train on the entire train set which helped the accuracy.
It seems one layer of hidden dims works best, we assume the dataset is small and the task is not very complex so one hidden dim produces good results.
Pool every 1 works best, we assume because there are many irrelevant features and invariance that max-pooling helps to deal with. Many max-pools layers helped to obtain the best accuracy. 
After playing with the parameters, we saw that the accuracy we got is reasonable compared to the default and other runs, so we used it. 

The depth that produces the best accuracy is 4, we think it’s the best one because it enables the network to learn a wide range of functions meaning it is complex enough for the problem. Depth 2 is not complex enough and layers 8 and 16 couldn’t be trained explanation for this is in the next section.
1.2) 
With depths 8 and 16 the network couldn’t be trained, because of the vanishing gradient as we saw in class many layers can cause it. Two things which may be done are:
a. As we saw in class residuals connections can help, the residuals connection-skip connections enable information to propagate to deeper layers in the network by identity mapping values to the output of their blocks. This helps to guarantee that in the backpropagation the gradient wouldn’t vanish. 
b. As seen in class batch normalization can improve gradient flow and thus help the vanishing gradient problem, the idea is that the batch normalization re-scales and re-centers the input to the activation layer helping the output of the function to be not too big or too small (depends also on the activation function) and by that helping the gradient 
"""

part3_q2 = r"""
The same parameters from section 1.1 were chosen for the same reasons.
Similar to 1.1 the network couldn’t be trained with 8 layers regardless of the filter sizes and that is due to the vanishing gradient.
With 4 layers we get similar results to 1.1 in filters sizes 32 and 64, we get the best accuracy for filter with 258.  We can assume that a large number of filters extract a lot of features combined with a high number of max-pools (which help to extract the important features) cause the accuracy to increase as the number of filters increases.

With 2 layers we get similar results to 1.1 in filters sizes 32 and 64, we get the best accuracy for filter 64 . We can assume a larger number of filters extract too many features which harm the network to learn- we overfit, and a smaller number of filters don’t extract enough features-underfit. We assume the difference here between the 4 layers from the 2 layers is due to configurations in the hyper-parameters that work differently with different depths and with 2 layers we can extract fewer features because there are fewer layers from the data, therefore, we get different results from the 4 layers.

"""

part3_q3 = r"""
We chose the same parameters as 1.1 excluding layer 1 we chose pool after 1, layer 2 pool after 2, and layers 3,4 where we chose pool after-3. The pools were chosen the smallest possible regarding the dimensions same as 1.1.
We chose here batch size of 100 and a number of batches of 500, that way we have less noise in each step which gave us the best performance, for layer 1 we chose batch size 50. Smaller batch size adds more noise in each step and takes longer training process, we assume the difference between the layers to the batch size is due with different network arcthictures (more layers) require different hyperparameters (batch size).
Layers 3 and 4 couldn’t be trained because of the vanishing gradient, we increased the number of filters per layer therefore the size of each layer is increased, and we see the vanishing gradient phenomenon in lower numbers of L.
We get the best result with 2 layers, with 1 layer we underfit and with higher number of layers we get the vanishing gradient.


"""


part3_q4 = r"""
We used the same parameters as 1.1fron the same reasons as 1.1 excluding:
Lr-0.0007
Layer 1 pool every-1, early stopping-9,
Layer 2 pool every-2 early stopping-12
Layer 3 pool every-3 early stopping-15
Layer 4 pool every-3 early stopping-18
Layer 5 pool every-4 early stopping-21
Layer 6 pool every-5 early stopping-24
Batch size 100 for layers 2,3,4 and 50 for layer 1 same as the previous section 3.

We lowered the learning rate because we wanted to smooth the loss curve graph, we tried different learning rates and got the best results with the above. The early stopping increases between layers because as the depth increase it takes more time for the model to converge so more early stopping was needed.

The model:
We added skip connection, dropout, and batch normalization layers.
The main reason we thought of adding skip connections is the vanishing gradient we saw in previous sections. We saw an overfit in the previous sections, so we thought of adding dropout. The batch normalization layers were added for the advantages we saw in class on top of as described in 1.1 which may help the vanishing gradient effect.
The new architecture:
The Resnet block and Resnet pooling block have a skip connection from the start to the end.
Res Block-(CONV ->BatchNorm-> ReLU  -> Dropout)*P
Res pooling - MaxPool
[(Res Block) -> Res pooling]*(N/P) -> (Linear -> ReLU -> Dropout )*M -> Linear

The idea of this architecture is to take ideas from Resnet and implement them in our network to get the best performance.
We tried different dropout parameters and got the best results with  0.1 in the Res-block and increasing dropout in the linear layers starting from 0.3 and increasing by 0.1 on each linear layer (0.3,0.4 ad so on), we assume that in the first layers the network doesn’t tend to overfit in contrast to the Liner layers where there are more parameters which result to more overfitting thus more dropout, the deeper we go in the network more dropout is added because deeper layers tend to overfit more.
We can see we reach max accuracy of 87.89 with 5 layers, we can assume that more layers overfit and fewer layers underfit. 
We reached better results than experiment 1 in all layers, we succeeded to train with layers 3 and 4 because of the skip connections and the normalization. 
In conclusion, we created a network with better accuracy but takes longer to coverage because of the larger number of layers and the dropout.


"""
