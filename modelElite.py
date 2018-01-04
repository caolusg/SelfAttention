import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from torch.autograd import Variable

class model(nn.Module):
    def __init__(self,parameters):                      #parameters为hyperparameters的实例
        super(model, self).__init__()
        self.parameter = parameters
        self.embed_num = self.parameter.embedding_num  #dict为字典类型 其长度比编号大一
        self.embed_dim = self.parameter.embedding_dim
        self.class_num = self.parameter.labelSize      #几分类
        self.hidden_dim = self.parameter.LSTM_hidden_dim
        self.num_layers = self.parameter.num_layers
        self.att_size = self.parameter.att_size

        print("embedding中词的数量：", self.embed_num)
        self.embedding = nn.Embedding(self.embed_num, self.embed_dim)
        #预训练 （glove）
        if self.parameter.word_Embedding:
            pretrained_weight = np.array(self.parameter.pretrained_weight)
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
            # fixed the word embedding
            self.embedding.weight.requires_grad = True

        self.bilstm = nn.LSTM(self.embed_dim, self.hidden_dim // 2, dropout=self.parameter.dropout, num_layers=self.num_layers,
                              batch_first=True, bidirectional=True)
        # linear
        self.hiddenLayer = nn.Linear(self.hidden_dim, self.class_num)
        # hidden
        self.hidden = self.init_hidden(self.num_layers, self.parameter.batch)
        # dropout
        self.dropout = nn.Dropout(self.parameter.dropout)
        self.dropout_embed = nn.Dropout(self.parameter.dropout_embed)
        self.attLinear1 = nn.Linear(self.embed_dim, self.att_size)
        self.attLinear2 = nn.Linear(self.embed_dim, self.class_num)
        self.UT = Variable(torch.rand(self.att_size, 1),requires_grad=True)  #默认生成 0~1 之间的小数

    def init_hidden(self, num_layers, batch_size):

        return (Variable(torch.zeros(2 * num_layers, batch_size, self.hidden_dim // 2)),
                Variable(torch.zeros(2 * num_layers, batch_size, self.hidden_dim // 2)))

    def forward(self, x):
        # print("*",x)
        embed = self.embedding(x)
        # print("len x", len(x))
        # print("embedding size : ", embed.size())
        # print("*"*10,x.size())
        # lstm
        # print(self.hidden[0].size())
        lstm_out, _ = self.bilstm(embed, self.hidden)
        staticOut = lstm_out
        # print("lstm_out: ", staticOut.size())
        # print("lstm_out size : ", lstm_out.size())
        # print("self.hidden size : ", self.hidden.size())
        rst = self.attLinear1(lstm_out)
        # print("attLinear1 size : ", rst.size())
        rst = F.tanh(rst)
        # print("after tanh size : ", rst.size())
        rst = rst.view(-1, self.att_size)
        # print("after view size : ", rst)
        # print("UT size : ", self.UT)
        rst = torch.mm(rst, self.UT)
        # print("after *UT size : ", rst.size())
        rst = rst.view(len(x), -1)
        # print("after second view: ", rst.size())
        rst = F.softmax(rst)
        # print("after SOFTMAX size : ", rst)

        alpha = rst

        # rst = torch.mm(staticOut.data, rst.data)
        for idx in range(rst.size(0)):
            if idx == 0:
                mul2Rst = torch.mm(torch.t(staticOut[idx]),rst[idx].unsqueeze(1))
            else :
                mul2Rst = torch.cat([mul2Rst, torch.mm(torch.t(staticOut[idx]), rst[idx].unsqueeze(1))], 1)

        mul2Rst = torch.t(mul2Rst)

        logit = self.attLinear2(mul2Rst)

        # print(logit)

        return logit, alpha