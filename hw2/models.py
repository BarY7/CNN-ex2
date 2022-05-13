import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Block, Linear, ReLU, Sigmoid, Dropout, Sequential


class MLP(Block):
    """
    A simple multilayer perceptron model based on our custom Blocks.
    Architecture is (with ReLU activation):

        FC(in, h1) -> ReLU -> FC(h1,h2) -> ReLU -> ... -> FC(hn, num_classes)

    Where FC is a fully-connected layer and h1,...,hn are the hidden layer
    dimensions.
    If dropout is used, a dropout layer is added after every activation
    function.
    """
    def __init__(self, in_features, num_classes, hidden_features=(),
                 activation='relu', dropout=0, **kw):
        super().__init__()
        """
        Create an MLP model Block.
        :param in_features: Number of features of the input of the first layer.
        :param num_classes: Number of features of the output of the last layer.
        :param hidden_features: A sequence of hidden layer dimensions.
        :param activation: Either 'relu' or 'sigmoid', specifying which 
        activation function to use between linear layers.
        :param: Dropout probability. Zero means no dropout.
        """
        blocks = []

        # TODO: Build the MLP architecture as described.
        # ====== YOUR CODE: ======
        dims = [in_features, *hidden_features, num_classes]
        for i in range(len(dims) - 1):
            input_dim = dims[i]
            out_dim = dims[i+1]
            blocks.append(Linear(input_dim,out_dim))
            if(i != len(dims) - 2): # not last block
                blocks.append(ReLU() if activation == 'relu' else Sigmoid())
                if(dropout > 0):
                    blocks.append(Dropout(p=dropout))
        # ========================

        self.sequence = Sequential(*blocks)

    def forward(self, x, **kw):
        return self.sequence(x, **kw)

    def backward(self, dout):
        return self.sequence.backward(dout)

    def params(self):
        return self.sequence.params()

    def train(self, training_mode=True):
        self.sequence.train(training_mode)

    def __repr__(self):
        return f'MLP, {self.sequence}'


class MyResPooling(nn.Module):

    def __init__(self, chanels,drop):
        super().__init__()

        self.chanels = chanels
        layers=[]
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, dilation=1))
        layers.append(nn.Dropout(drop))

        self.down = nn.Conv2d(self.chanels, self.chanels, kernel_size=1, stride=2)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        z = self.net(x)
        x_1 = self.down(x)
        out = z + x_1
        out =nn.ReLU()(out)
        return out

class MyResBlock(nn.Module):

    def __init__(self, chanels, r=-1, pool_every=-1,filters=[]):

        super().__init__()
        self.r = r
        self.filters = filters
        layers = []
        for j in range(pool_every):
            if  self.r == -1:
                in_chan = chanels
            else:
                in_chan = self.filters[self.r]
            layers.append(nn.Conv2d(in_chan, self.filters[self.r + 1], 3, padding=1))
            layers.append(nn.BatchNorm2d(self.filters[self.r + 1]))
            layers.append(nn.ReLU())
            self.r = self.r + 1

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        z = self.net(x)
        x_1 = F.pad(input=x, pad=(0, 0, 0, 0, self.filters[self.r]-x.shape[1], 0), mode='constant', value=0)
        out = z + x_1
        out = nn.ReLU()(out)
        return out

class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(Conv -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param filters: A list of of length N containing the number of
            filters in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        self.in_size = in_size
        self.out_classes = out_classes
        self.filters = filters
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        # [(Conv -> ReLU)*P -> MaxPool]*(N/P)
        # Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        # Pooling to reduce dimensions.
        # ====== YOUR CODE: ======
        self.calc_h = in_h
        self.calc_w = in_w
        r = -1
        for i in range ((len(self.filters)//self.pool_every)):
            for j in range(self.pool_every):
                if r==-1:
                    in_chan = in_channels
                else:
                    in_chan = self.filters[r]
                layers.append(nn.Conv2d(in_chan, self.filters[r+1],3,padding=1))
                layers.append(nn.ReLU())
                r = r + 1

            layers.append(nn.MaxPool2d( kernel_size=2,stride=2,dilation=1))
            self.calc_h = (self.calc_h - 2) // 2 + 1
            self.calc_w = (self.calc_w - 2) // 2 + 1


        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        # (Linear -> ReLU)*M -> Linear
        # You'll need to calculate the number of features first.
        # The last Linear layer should have an output dimension of out_classes.
        # ====== YOUR CODE: ======
        layers.append(nn.Flatten())
        features_num = self.filters[-1]*self.calc_h*self.calc_w

        for i in range(len(self.hidden_dims)):
            in_features = features_num
            if i>0:
                in_features = self.hidden_dims[i-1]
            layers.append(nn.Linear(in_features, self.hidden_dims[i]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(self.hidden_dims[-1], self.out_classes))


        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        # Extract features from the input, run the classifier on them and
        # return class scores.
        # ====== YOUR CODE: ======
        out = None
        extract = self.feature_extractor(x)
        out = self.classifier(extract)
        # ========================
        return out

class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        super().__init__(in_size, out_classes, filters, pool_every, hidden_dims)

    # TODO: Change whatever you want about the ConvClassifier to try to
    # improve it's results on CIFAR-10.
    # For example, add batchnorm, dropout, skip connections, change conv
    # filter sizes etc.
    # ====== YOUR CODE: ======
    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        # [(Conv -> ReLU)*P -> MaxPool]*(N/P)
        # Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        # Pooling to reduce dimensions.
        # ====== YOUR CODE: ======
        self.calc_h = in_h
        self.calc_w = in_w
        r = -1
        for i in range ((len(self.filters)//self.pool_every)):
            layers.append(MyResBlock(in_channels,r,self.pool_every,self.filters))
            r = r + self.pool_every

            layers.append(MyResPooling(self.filters[r],0.1))
            self.calc_h = (self.calc_h - 2) // 2 + 1
            self.calc_w = (self.calc_w - 2) // 2 + 1


        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        # (Linear -> ReLU)*M -> Linear
        # You'll need to calculate the number of features first.
        # The last Linear layer should have an output dimension of out_classes.
        # ====== YOUR CODE: ======
        layers.append(nn.Flatten())
        features_num = self.filters[-1]*self.calc_h*self.calc_w

        for i in range(len(self.hidden_dims)):
            in_features = features_num
            if i>0:
                in_features = self.hidden_dims[i-1]
            layers.append(nn.Linear(in_features, self.hidden_dims[i]))
            layers.append(nn.Dropout(round(0.3+i/10,2)))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(self.hidden_dims[-1], self.out_classes))


        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        # Extract features from the input, run the classifier on them and
        # return class scores.
        # ====== YOUR CODE: ======
        out = None
        extract = self.feature_extractor(x)
        out = self.classifier(extract)
        # ========================
        return out
    # ========================

