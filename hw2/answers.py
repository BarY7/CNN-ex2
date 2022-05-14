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
    wstd, lr, reg = 0.1 , 0.02, 0.005

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different hyperparameters for each optimizer.
    # ====== YOUR CODE: ======
    if opt_name == 'vanilla':
        lr = 0.01
    if opt_name == 'momentum':
        lr = 0.001 #steps are less affected by noise
    if opt_name == 'rmsprop':
        lr = 0.001
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
Based on the test loss graph, we can see that 0.8 is still downtrending while 0.4 starts to increase.
Given enough epochs its possible that 0.8 dropout model could pull ahead of the 0.4 one in terms of test set accuracy.
We tried this expriment with 80 epochs and indeed the model with 0.8 dropout got the best test set accuracy.



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
layer 2 pool ever 2, layer 4 pool ever 2, layer 8 pool ever 8, layer 16 pool ever 16.
We tried different parameters and we got the best results with those; we can see we stop approximately as the loss goes up with the early stopping, and the loss graphs seem smooth so the learning rate is appropriate. We put 50 epochs because we counted on the early stopping to regulate and stop the learning process. After playing with the parameters, we saw that the accuracy we got is reasonable compared to the default and other runs, so we used it. 
We chose batches 1000 so we can train on the entire train set which helped the accuracy.
It seems only one layer of hidden dims works best, we assume that with more we overfit too much to the data.
Pool every 2 works best, we assume because there are many irrelevant features and invariance that max-pooling helps to deal with. Pool every 1 downscales too much we lose too much information.

The depth that produces the best accuracy is 4, we think it’s the best one because it enables the network to learn a wide range of functions meaning it is complex enough for the problem. Depth 2 is not complex enough and layers 8 and 16 couldn’t be trained explanation for this is in the next section.
1.2) 
With depths 8 and 16 the network couldn’t be trained, because of the vanishing gradient as we saw in class many layers can cause it. Two things which may be done are:
a. As we saw in class residuals connections can help, the residuals connection-skip connections enable information to propagate to deeper layers in the network by identity mapping values to the output of their blocks. This helps to guarantee that in the backpropagation the gradient wouldn’t vanish. 
b. As seen in class batch normalization can improve gradient flow and thus help the vanishing the gradient problem, the idea is that the batch normalization re-scales and re-centers the input to the activation layer helping the output of the function to be not too big or too small (depends also on the activation function) and by that helping the gradient to not be too small or too big, meaning helping to alleviate the vanishing gradient problem.


"""

part3_q2 = r"""
The same parameters from section 1.1 were chosen for the same reasons.
Similar to 1.1 the network can’t be trained with 8 layers regardless of the filter sizes and that is due to the vanishing gradient.
With 4 layers we get similar results to 1.1 in filters sizes 32 and 64, we get the best accuracy for filter with 128. 
(a) We can assume a larger number of filters extract too much information (many features) which harms the network to learn- we overfit, and a smaller number of filters don’t extract enough information-underfit.

With 2 layers we get similar results to 1.1 in filters sizes 32 and 64, we get the best accuracy for filter 64  the reason same as (a). We assume the difference here between the 4 layers from the 2 layers is due to configuration in the hyper-parameters that work differently with different depths and therefore we get different results from the 2 layers.

"""

part3_q3 = r"""
We chose the same parameters as 1.1 excluding layer 1 we chose pool after-1 layer 2,3,4 where we chose pool after-3.
We saw that with pool after 3 we get the best results, we lost too much information with lower pools, and with higher pools, we had too much information.
Layers 3 and 4 couldn’t be trained because of the vanishing gradient, we increased the number of filters per layer therefore the size of each layer is increased, and we see the vanishing gradient phenomenon in lower numbers of L.
We get the best result with 1 layer, it seems that adding more layers causes overfit we see this in the higher train accuracy and lower test accuracy with layer 2.


"""


part3_q4 = r"""
We used the same parameters as 1.1 excluding:
Lr-0.0006
Layer 1 pool every-1, early stopping-9,
Layer 2 pool every-2 early stopping-12
Layer 3 pool every-3 early stopping-15
Layer 4 pool every-4 early stopping-18

The hyperparameters were chosen after experimenting with different ones, they were chosen from the same reasons as 1.1.
We lowered the learning rate because we wanted to smooth the loss curve graph, we tried different learning rates and got the best results with the above. The early stopping increases between layers because as the depth increase it takes more time for the model to converge so more early stopping was needed.

The model:
We added skip connection, dropout, and batch normalization layers.
The main reason for adding skip connections is the vanishing gradient we saw in previous sections. We saw an overfit in the previous sections, so dropout layers were added. The batch normalization layers were added for the advantages we saw in class on top of as described in 1.1 which may help the vanishing gradient effect.
The new architecture:
The Resnet block and Resnet pooling block have a skip connection from its start to the end.
Res Block-(CONV ->BatchNorm-> ReLU)*P
Res pooling - Dropout-> MaxPool
[(Res Block)*P -> Res pooling]*(N/P) -> (Linear -> Dropout -> ReLU)*M -> Linear

The idea of this architecture is to mix the VGG with Resnet to get the best performance.
We tried different dropout parameters and got the best results adding 0.1 in the Res-block and 0.3 in the linear dropout, we assume that in the first layers the network doesn’t tend to overfit in contrast to the Liner layers where there are more parameters and because of that we needed there more dropout.
We can see we reach max accuracy of 86.5 with 3 layers, we can assume that more layers overfit and fewer layers underfit. 
We reached better results than experiment 1 in all layers, we succeeded to train with layers 3 and 4 because of the skip connections and the normalization. 
In conclusion, we created a network with better accuracy but takes longer to coverage because of the larger number of layers and the dropout.


"""
