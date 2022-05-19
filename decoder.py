import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init


class Decoder(nn.Module):
    def __init__(self, input_dim, num_weights, num_classes, dropout=0., activation=F.relu):
        """

        :param input_dim: 输入特征维度
        :param num_weights: 权重数量
        :param num_classes: 评分级别
        :param dropout:
        :param activation:
        """
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.num_weights = num_weights
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.weight = nn.ParameterList([
            nn.Parameter(torch.Tensor(input_dim, input_dim))
            for _ in range(num_weights)
        ])
        self.weight_classifier = nn.Parameter(torch.Tensor(num_weights, num_classes))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight_classifier)
        for i in range(len(self.weight)):
            init.orthogonal_(self.weight[i], gain=1.1)

    def forward(self, user_inputs, item_inputs, user_indices, item_indices):
        """

        :param user_inputs:
        :param item_inputs:
        :param user_indices:
        :param item_indices:
        :return:
        """
        user_inputs = self.dropout(user_inputs)
        item_inputs = self.dropout(item_inputs)
        user_inputs = user_inputs[user_indices]
        item_inputs = item_inputs[item_indices]

        basis_outputs = []
        for i in range(self.num_weights):
            tmp = torch.matmul(user_inputs, self.weight[i])
            out = torch.sum(tmp * item_inputs, dim=1, keepdim=True)
            basis_outputs.append(out)

        basis_outputs = torch.cat(basis_outputs, dim=1)
        outputs = torch.matmul(basis_outputs, self.weight_classifier)
        outputs = self.activation(outputs)

        return outputs
