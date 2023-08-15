from torch.nn.modules import Module
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
import math
from torch.nn import init


class GRU(Module):
    def __init__(self, input_size, hidden_size, number_layer, bias=True, bidirectional=False):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.number_layer = number_layer
        self.bias = bias
        self.bidirectional = bidirectional

        self.GRU_layer = torch.nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, bias=self.bias,
                                      num_layers=self.number_layer, bidirectional=self.bidirectional, batch_first=True)
        # self.liner = torch.nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, h0):
        out, h = self.GRU_layer(x, h0)
        return out, h


class Current(Module):
    def __init__(self, input_size, hidden_size, number_layer):
        super(Current, self).__init__()
        self.rnn = GRU(input_size, hidden_size, number_layer)
        self.temporal = Temporal(1, hidden_size)

    def forward(self, x, h0):
        u = x[:, :, :50]
        v = x[:, :, 50:]
        u = self.temporal(u)
        v = self.temporal(v)
        x = torch.cat((u, v), dim=2)
        out, h = self.rnn(x, h0)
        return out, h


class Speed(Module):
    def __init__(self, input_size, hidden_size, number_layer):
        super(Speed, self).__init__()
        self.rnn = GRU(input_size, hidden_size, number_layer)
        self.speed = nn.Embedding(64, 4)

    def forward(self, x, h0):
        x = self.speed(x)
        x = self.rnn(x, h0)

        return x


class Encoder(Module):
    def __init__(self, data_length, input_size, hidden_size, num_layers, dropout=0, bidirectional=False, batch_first=True, bias=False):
        super(Encoder, self).__init__()
        self.length = data_length
        self.RNN = nn.LSTM(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)
        self.linear = nn.Linear(hidden_size * self.length, hidden_size)
        self.attention = SelfAttention(a=hidden_size, q=hidden_size, k=hidden_size, v=hidden_size, h=1)

    def forward(self, x, h0, c0):
        # x = self.temporal(x, device)
        x, (hn, cn) = self.RNN(x, (h0, c0))
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = self.attention(x)
        # x = x.unsqueeze(1)
        return x, hn, cn


class SelfAttention(Module):
    def __init__(self, a, h, k, v, q, dropout=0.1):
        """
        :param a: Output dimensionality of the previous model
        :param h: number of heads
        :param k: Dimensionality of and keys
        :param v: Dimensionality of values
        :param q: Dimensionality of queries
        """
        super(SelfAttention, self).__init__()

        self.fc_q = nn.Linear(a, h * q)
        self.fc_k = nn.Linear(a, h * k)
        self.fc_v = nn.Linear(a, h * v)
        self.fc_o = nn.Linear(h * v, a)
        self.dropout = nn.Dropout(dropout)

        self.a = a
        self.k = k
        self.v = v
        self.h = h

    def forward(self, x):
        b_s, nq = x.shape[:2]
        # x = torch.flatten(x, 1)
        q = self.fc_q(x)
        k = self.fc_k(x).view(b_s, nq).permute(1, 0)
        v = self.fc_v(x)

        att = torch.matmul(k, q) / np.sqrt(self.k)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(v, att)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class TraceLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TraceLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Softsign = nn.Softsign()
        self.sigmoid = nn.Sigmoid()

        self._flat_weights_names = []

        w_ii = nn.Parameter(torch.Tensor(input_size, 6 * hidden_size))
        w_hh = nn.Parameter(torch.Tensor(hidden_size, 6 * hidden_size))
        w_cc = nn.Parameter(torch.Tensor(hidden_size, 3 * hidden_size))
        w_ee = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
        w_rr = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
        b_ih = nn.Parameter(torch.Tensor(6 * hidden_size))
        b_hh = nn.Parameter(torch.Tensor(6 * hidden_size))
        b_k = nn.Parameter(torch.Tensor(hidden_size))

        layer_param = (w_ii, w_hh, w_cc, w_ee, w_rr, b_k, b_ih, b_hh)

        param_names = ['weight_ii', 'weight_hh', 'weight_cc', 'weight_ee',
                       'weight_rr', 'weight_aa', 'bias_ih', 'bias_hh']

        for name, param in zip(param_names, layer_param):
            setattr(self, name, param)
        self._flat_weights_names.extend(param_names)

        self.param_length = len(param_names)
        self._flat_weights = [(lambda wn: getattr(self, wn) if hasattr(self, wn) else None)(wn) for wn in
                              self._flat_weights_names]
        self.flatten_parameters()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def flatten_parameters(self) -> None:
        if len(self._flat_weights) != len(self._flat_weights_names):
            return

        for w in self._flat_weights:
            if not isinstance(w, Tensor):
                return

        first_fw = self._flat_weights[0]
        dtype = first_fw.dtype
        for fw in self._flat_weights:
            if (not isinstance(fw.data, Tensor) or not (fw.data.dtype == dtype) or
                    not fw.data.is_cuda or
                    not torch.backends.cudnn.is_acceptable(fw.data)):
                return

        unique_data_ptrs = set(p.data_ptr() for p in self._flat_weights)
        if len(unique_data_ptrs) != len(self._flat_weights):
            return

        with torch.cuda.device_of(first_fw):
            import torch.backends.cudnn.rnn as rnn

            with torch.no_grad():
                if torch._use_cudnn_rnn_flatten_weight():
                    num_weights = 4 if self.bias else 2
                    if self.proj_size > 0:
                        num_weights += 1
                    torch._cudnn_rnn_flatten_weight(
                        self._flat_weights, num_weights,
                        self.input_size, rnn.get_cudnn_mode(self.mode),
                        self.hidden_size, self.proj_size, self.num_layers,  # type: ignore
                        self.batch_first, bool(self.bidirectional))  # type: ignore

    def forward(self, x, h, c, e, r):
        """
        :param x: input
        :param h: hidden
        :param c: cell
        :param e: eddy
        :param r: current
        :return:
        """
        # the weight of input and hidden, bias
        # x = x.squeeze(1)
        bias = self._flat_weights[-1] + self._flat_weights[-2]
        weight_bias = torch.matmul(x, self._flat_weights[0]) + torch.matmul(h, self._flat_weights[1]) + bias
        weight_bias = torch.split(weight_bias, self.hidden_size, dim=1)

        # split other weight
        weight_cell = torch.split(self._flat_weights[2], self.hidden_size, dim=1)
        weight_eddy = torch.split(self._flat_weights[3], self.hidden_size, dim=1)
        weight_current = torch.split(self._flat_weights[4], self.hidden_size, dim=1)

        # the weight of eddy feature
        forget_gate = self.sigmoid(weight_bias[0] + torch.matmul(c, weight_cell[0]) + torch.matmul(e, weight_eddy[0]) + torch.matmul(r, weight_current[0]))
        input_gate = self.sigmoid(weight_bias[1] + torch.matmul(c, weight_cell[1]) + torch.matmul(e, weight_eddy[1]) + torch.matmul(r, weight_current[1]))
        c = forget_gate * c + input_gate * self.Softsign(weight_bias[2])
        e = self.Softsign(weight_bias[3] + torch.matmul(e, weight_eddy[2]))
        r = self.Softsign(weight_bias[3] + torch.matmul(r, weight_current[2]))
        output_gate = self.sigmoid(weight_bias[4] + torch.matmul(e, weight_cell[2]) + torch.matmul(e, weight_eddy[3]) + torch.matmul(r, weight_current[3]))
        h = output_gate * self.Softsign(c)

        return h, c, e, r
        # return h, c, e


class TraceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, number_layers, output_size):
        super(TraceLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.number_layers = number_layers
        self.output_size = output_size

        deep_list = []
        for i in range(number_layers):
            temp_input_size = self.input_size if i == 0 else self.hidden_size
            deep_list.append(TraceLSTMCell(
                input_size=temp_input_size, hidden_size=self.hidden_size
            ))

        self.deep_list = nn.ModuleList(deep_list)
        self.linear = torch.nn.Linear(hidden_size*number_layers, output_size)

    def forward(self, x, h, c, e, r):
        temp_input = x

        for layer in range(self.number_layers):
            hn = h[layer, :, :]
            cn = c[layer, :, :]
            en = e[layer, :, :]
            rn = r[layer, :, :]
            # hn, cn, en = self.deep_list[layer](temp_input, hn, cn, en, rn)
            hn, cn, en, rn = self.deep_list[layer](temp_input, hn, cn, en, rn)
            temp_input = hn
            out_hn = hn.unsqueeze(0) if layer == 0 else torch.cat((out_hn, hn.unsqueeze(0)))
            out_cn = cn.unsqueeze(0) if layer == 0 else torch.cat((out_cn, cn.unsqueeze(0)))

        hn = out_hn.permute(1, 0, 2)
        hn = torch.flatten(hn, 1)
        output = self.linear(hn)

        return output, out_hn, out_cn


class Temporal(Module):
    """
        Temporal Attention
    """
    def __init__(self, input_size, hidden_size):
        super(Temporal, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.fusion = nn.Linear(in_features=2 * hidden_size, out_features=input_size)

    def forward(self, x):
        b, z, m = x.shape

        x1 = x.clone()
        for i in range(z):
            output, _ = self.rnn(x[:, i, ...].unsqueeze(-1))
            output = self.fusion(output)
            output = torch.sigmoid(output)
            x1[:, i, :] = x[:, i, :] * output.squeeze(-1)

        return x1
