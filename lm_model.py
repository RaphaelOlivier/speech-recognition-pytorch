import numpy as np
import torch
import os
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init


def to_tensor(narray):
    return torch.from_numpy(narray).float()


def to_variable(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor.float())


class LanguageModel(torch.nn.Module):

    def __init__(self, vocab_size, emb_size, h_size, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.act = torch.nn.ReLU()
        self.isCuda = False
        self.h_size = h_size
        self.emb_size = emb_size
        self.drop1 = torch.nn.Dropout(0.4)
        self.drop2 = LockedDropout(0.1)
        self.rnn_sizes = [self.emb_size] + \
            [h_size for i in range(self.n_layers-1)] + [self.emb_size]
        self.rnns = torch.nn.ModuleList([torch.nn.LSTM(
            input_size=self.rnn_sizes[i], hidden_size=self.rnn_sizes[i+1],
            batch_first=True) for i in range(self.n_layers)])
        self.softmaxLayer = torch.nn.Linear(
            self.emb_size, vocab_size)

        for param in self.softmaxLayer.parameters():
            init.uniform(param, -0.1, 0.1)
        born = 1. / np.sqrt(h_size)
        for k in range(self.n_layers):
            for param in self.rnns[k].parameters():
                init.uniform(param, -born, born)

    def embed(self, inp):
        return self.act(F.embedding(inp, self.softmaxLayer.weight))

    def forward(self, input_sequence, gen=0):
        ar = to_variable(torch.zeros(1))
        tar = to_variable(
            torch.zeros(1))
        if(not self.isCuda):
            ar = ar.cpu()
            tar = tar.cpu()
        embedded = self.drop1(self.embed(input_sequence))
        outputs = []
        batch_size = input_sequence.size()[
            0]
        projected = None
        outLSTM, new_state = None, None
        prevOutLSTM = None
        old_states = [None for i in range(self.n_layers)]

        x = embedded[:, :, :]
        for k in range(self.n_layers):
            outLSTM, new_state = self.rnns[k](x, old_states[k])
            old_states[k] = new_state
            outLSTM = self.drop2(outLSTM)
            x = outLSTM

#           add tar
        output = outLSTM
        ar += l2_penalty(output)
        projected = self.softmaxLayer(self.drop2(output))
        outputs.append(projected)
#        print(torch.max(projected, dim=2)[1].size())

        if(gen > 0):

            new_input = torch.max(
                projected, dim=2)[1][:, -1:]
            for i in range(gen):
                x = self.embed(new_input)
                for k in range(self.n_layers):
                    outLSTM, new_state = self.rnns[k](
                        x, old_states[k])
                    if(k == self.n_layers-1):
                        tar += l2_penalty(
                            new_state[0] - old_states[self.n_layers-1][0])
                    old_states[k] = new_state
                    x = outLSTM
                output = self.drop1(outLSTM)
                ar += l2_penalty(output)

                projected = self.softmaxLayer(output)
                outputs.append(projected)
                new_input = torch.max(projected, dim=2)[1]

        logits = torch.cat(outputs, dim=1)
        return logits, ar, tar

    def set_cpu(self):
        self.isCuda = False
        return self._apply(lambda t: t.cpu())

    def set_cuda(self):
        self.isCuda = True
        return self._apply(lambda t: t.cuda(None))


def l1_penalty(var):
    return torch.abs(var).sum()


def l2_penalty(var):
    return torch.sqrt(torch.pow(var, 2).sum())


def write_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(path):
    model = spec_model()
    model.load_state_dict(path)
    return model


def spec_model():
    net = LanguageModel(
        vocab_size=17053, emb_size=256, h_size=256, n_layers=3)
    return net


def init_unigrams(net, data, vocab_size):
    ucounts = np.zeros(vocab_size)
    for s in data:
        for w in s:
            ucounts[w] += 1
    ucounts /= ucounts.sum()
    smooth = 0.1
    for w in range(vocab_size):
        ucounts[w] = ucounts[w]*(1 - smooth) + smooth/vocab_size
    ucounts = np.log(ucounts)
    net.softmaxLayer.bias.data = to_tensor(ucounts)


class LockedDropout(torch.nn.Module):
    def __init__(self, p=None):
        super(LockedDropout, self).__init__()
        self.dropout = p

    def forward(self, x):
        if not self.training or not self.dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.dropout)
        mask = Variable(m, requires_grad=False) / (1 - self.dropout)
        mask = mask.expand_as(x)
        return mask * x
