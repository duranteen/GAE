import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init


class CatGCNEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_support, dropout=0., use_bias=False, activation=F.relu):
        """

        :param input_dim:  输入特征维度
        :param output_dim: 输出特征维度
        :param num_support: 评分类别数
        :param dropout: dropout
        :param use_bias: 是否使用偏置项
        :param activation: 激活函数
        """
        super(CatGCNEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_support = num_support
        self.dropout = nn.Dropout(dropout)
        self.use_bias = use_bias
        self.activation = activation

        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias_user = nn.Parameter(torch.Tensor(output_dim, ))
            self.bias_item = nn.Parameter(torch.Tensor(output_dim, ))

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias_user)
            init.zeros_(self.bias_item)

    def forward(self, user_supports, item_supports, user_inputs, item_inputs):
        """

        :param user_supports: 每个评分等级对应的用户与商品邻接矩阵
        :param item_supports: 每个评分等级对应的商品与用户邻接矩阵
        :param user_inputs: 用户特征的输入
        :param item_inputs: 商品特征的输入
        :return: 用户与商品隐藏特征
        """
        user_inputs = self.dropout(user_inputs)
        item_inputs = self.dropout(item_inputs)

        user_hidden = []
        item_hidden = []
        weights = torch.split(self.weight, self.output_dim // self.num_support, dim=1)
        for i in range(self.num_support):
            tmp_u = torch.matmul(user_inputs, weights[i])
            tmp_v = torch.matmul(item_inputs, weights[i])
            tmp_user_hidden = torch.sparse.mm(user_supports[i], tmp_v)
            tmp_item_hidden = torch.sparse.mm(item_supports[i], tmp_u)
            user_hidden.append(tmp_user_hidden)
            item_hidden.append(tmp_item_hidden)

        user_hidden = torch.cat(user_hidden, dim=1)
        item_hidden = torch.cat(item_hidden, dim=1)
        user_outputs = self.activation(user_hidden)
        item_outputs = self.activation(item_hidden)
        if self.use_bias:
            user_outputs = user_outputs.clone() + self.bias_user
            item_outputs = item_outputs.clone() + self.bias_item

        return user_outputs, item_outputs


class SumGCNEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_support, dropout=0., use_bias=False, activation=F.relu):
        """

        :param input_dim:  输入特征维度
        :param output_dim: 输出特征维度
        :param num_support: 评分类别数
        :param dropout: dropout
        :param use_bias: 是否使用偏置项
        :param activation: 激活函数
        """
        super(CatGCNEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_support = num_support
        self.dropout = nn.Dropout(dropout)
        self.use_bias = use_bias
        self.activation = activation

        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim * num_support))
        if self.use_bias:
            self.bias_user = nn.Parameter(torch.Tensor(output_dim, ))
            self.bias_item = nn.Parameter(torch.Tensor(output_dim, ))

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias_user)
            init.zeros_(self.bias_item)

    def forward(self, user_supports, item_supports, user_inputs, item_inputs):
        """

        :param user_supports: 每个评分等级对应的用户与商品邻接矩阵
        :param item_supports: 每个评分等级对应的商品与用户邻接矩阵
        :param user_inputs: 用户特征的输入
        :param item_inputs: 商品特征的输入
        :return: 用户与商品隐藏特征
        """
        user_inputs = self.dropout(user_inputs)
        item_inputs = self.dropout(item_inputs)

        user_hidden = []
        item_hidden = []
        weights = torch.split(self.weight, self.output_dim, dim=1)
        for i in range(self.num_support):
            w = sum(weights[: i + 1])
            tmp_u = torch.matmul(user_inputs, w)
            tmp_v = torch.matmul(item_inputs, w)
            tmp_user_hidden = torch.sparse.mm(user_supports[i], tmp_v)
            tmp_item_hidden = torch.sparse.mm(item_supports[i], tmp_u)
            user_hidden.append(tmp_user_hidden)
            item_hidden.append(tmp_item_hidden)

        user_hidden = sum(user_hidden)
        item_hidden = sum(item_hidden)
        user_outputs = self.activation(user_hidden)
        item_outputs = self.activation(item_hidden)
        if self.use_bias:
            user_outputs = user_outputs.clone() + self.bias_user
            item_outputs = item_outputs.clone() + self.bias_item

        return user_outputs, item_outputs