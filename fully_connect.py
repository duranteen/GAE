import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init


class FullyConnected(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0., use_bias=False, activation=F.relu, share_weight=False):
        super(FullyConnected, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.share_weight = share_weight
        if share_weight:
            self.weights_u = nn.Parameter(torch.Tensor(input_dim, output_dim))
            self.weights_v = self.weights_u
            if use_bias:
                self.user_bias = nn.Parameter(torch.Tensor(output_dim))
                self.item_bias = self.user_bias
        else:
            self.weights_u = nn.Parameter(torch.Tensor(input_dim, output_dim))
            self.weights_v = nn.Parameter(torch.Tensor(input_dim, output_dim))
            if use_bias:
                self.user_bias = nn.Parameter(torch.Tensor(output_dim))
                self.item_bias = nn.Parameter(torch.Tensor(output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        if self.share_weight:
            init.xavier_uniform_(self.weights_u)
            if self.use_bias:
                init.normal_(self.user_bias, std=0.5)
        else:
            init.xavier_uniform_(self.weights_u)
            init.xavier_uniform_(self.weights_v)
            if self.use_bias:
                init.normal_(self.user_bias, std=0.5)
                init.normal_(self.item_bias, std=0.5)

    def forward(self, user_inputs, item_inputs):
        """

        :param user_inputs:
        :param item_inputs:
        :return:
        """
        x_u = self.dropout(user_inputs)
        x_u = torch.matmul(x_u, self.weights_u)

        x_v = self.dropout(item_inputs)
        x_v = torch.matmul(x_v, self.weights_v)

        x_u = self.activation(x_u)
        x_v = self.activation(x_v)
        if self.use_bias:
            x_u = x_u.clone() + self.user_bias
            x_v = x_v.clone() + self.item_bias

        return x_u, x_v
