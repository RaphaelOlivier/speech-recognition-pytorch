import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from lm_model import *
import datatools


def to_tensor(narray):
    return torch.from_numpy(narray).float()


def to_variable(tensor, cuda=True):
    if cuda and torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor.float())


def random_seq_length(seq):
    p = 0.90
    s = 2
    seq = np.random.choice(
        [seq, seq/2], p=[p, 1-p])
    return max(int(np.random.normal(seq, s)), 1)


def make_a_batch(seqs, iseq, seq_length, batch_size):
    new_seq_length = random_seq_length(
        seq_length)

    cutSeqs = list()
    shiftSeqs = list()
    i = iseq
    while i < len(seqs):
        seq = seqs[i]
        trunc_length = len(seq)-1
        for j in range(0, trunc_length-new_seq_length, new_seq_length):
            cutSeqs.append(seq[j:j+new_seq_length].reshape(1, new_seq_length))
            shiftSeqs.append(seq[j+1:j+new_seq_length +
                                 1].reshape(1, new_seq_length))

            if len(cutSeqs) == batch_size:
                break
        i += 1
        if len(cutSeqs) == batch_size:
            break

    iseq = i
    a = np.arange(len(cutSeqs))
    np.random.shuffle(a)
    batch = np.concatenate(cutSeqs, axis=0)[a]
    labels = np.concatenate(shiftSeqs, axis=0)[a]
    return batch, labels, iseq, new_seq_length


def training(net, train_data, num_epochs, batch_size, learning_rate, vocab_size, cuda=True):
    alpha = 0.001
    beta = 0.001
    decay = 0.00001
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    seq_length = 25
    nll_loss = torch.nn.NLLLoss(
        size_average=True)
    ce_loss = torch.nn.CrossEntropyLoss(
        size_average=True)
    optim = torch.optim.Adam(net.parameters(), weight_decay=decay)
#    optim = torch.optim.SGD(
#        net.parameters(), lr=learning_rate, weight_decay=decay)
    if cuda and torch.cuda.is_available():
        net = net.set_cuda()
        nll_loss = nll_loss.cuda()
        ce_loss = ce_loss.cuda()
    for i in range(num_epochs):
        wordCounter = 0
        step = 20000
        threshold = step
        full_nll = 0
        net.train()
        print("epoch "+str(i))
        a = np.arange(len(train_data))
        np.random.shuffle(a)
        seqs = train_data[a]
        iseq = 0
        while(iseq < len(seqs)):
            input_val, label, iseq, new_length = make_a_batch(
                seqs, iseq, seq_length, batch_size)

            len_batch = len(input_val)
            wordCounter += len_batch*new_length
            lr_factor = float(
                new_length)/seq_length
#            optim = torch.optim.SGD(
#                net.parameters(), lr=learning_rate*lr_factor, weight_decay=decay)

            input_val, label = to_tensor(
                input_val), to_tensor(label)
            prediction, ar, tar = net(
                to_variable(input_val, cuda).long())
            probs = logsoftmax(
                prediction.view(-1, vocab_size))
            local_nll = nll_loss(probs, to_variable(
                label.view(-1), cuda).long())
            full_nll += local_nll.data.cpu().numpy()[
                0]*new_length*len_batch
            loss = ce_loss(probs, to_variable(
                label.view(-1), cuda).long())
            loss += alpha*ar + beta*tar
            loss.backward()

            optim.step()
            optim.zero_grad()

            if(wordCounter > threshold):
                print(str(threshold)[
                      :-3] + "k words, local loss : "+str(loss[0].data.cpu().numpy()[0]))
                threshold += step
        print(
            "Training loss per word : ", full_nll/wordCounter)
    net.eval()


if __name__ == "__main__":

    vocab = datatools.init_vocab()

    X, Y = datatools.load_train_data(vocab)
    #Xd, Yd = datatools.load_dev_data(vocab)
    #Xt = datatools.load_test_data()

    data, labels = X, Y
    # print(len(vocab))
    net = spec_model()
    init_unigrams(net, labels, len(vocab))
    # net = load_model("hw3/mynet")
    training(net, labels, 8, 32,
             15, len(vocab), cuda=True)
    write_model(net, "../lm")
