import datatools
import models
import os
import torch
from torch.autograd import Variable
import torch.utils.data
import numpy as np
from wsj_loader import WSJ
os.environ['WSJ_PATH'] = 'data'


def sort_sequence(sequences):
    lengths = np.array([len(seq) for seq in sequences])
    sort = np.flip(np.argsort(lengths), axis=0)
    return sequences[sort], lengths[sort], sort


def pad_sequence(sequences, batch_first=False, padding_value=0):
    """

    Arguments:
        sequences (list[Variable]): list of variable length sequences.
        batch_first (bool, optional): output will be in BxTx* if True, or in
            TxBx* otherwise
        padding_value (float, optional): value for padded elements.

    Returns:
        Variable of size ``T x B x * `` if batch_first is False
        Variable of size ``B x T x * `` otherwise
    """

    # assuming trailing dimensions and type of all the Variables
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    max_len, trailing_dims = max_size[0], max_size[1:]
    prev_l = max_len
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_variable = Variable(sequences[0].data.new(*out_dims).fill_(padding_value))
    for i, variable in enumerate(sequences):
        length = variable.size(0)
        # temporary sort check, can be removed when we handle sorting internally
        if prev_l < length:
            raise ValueError("lengths array has to be sorted in decreasing order")
        prev_l = length
        # use index notation to prevent duplicate references to the variable
        if batch_first:
            out_variable[i, :length, ...] = variable
        else:
            out_variable[:length, i, ...] = variable

    return out_variable


def predict(net, X, vocab=None):

    n = len(X)

    Y = list()
    for s in X:
        input_val = s
        input_val = datatools.to_tensor(input_val)
        pred = net.forward(datatools.to_variable(input_val),
                           mode="eval")
        Y.append(pred)
        if(vocab is not None):
            print(datatools.to_string_char(np.array([pred]), vocab))
    return np.array(Y)


def score(pred, Y, vocab):
    pred = datatools.to_string_char(pred, vocab)
    Y = datatools.to_string_char(Y, vocab)
    c = 0
    for i in range(len(Y)):
        c += levenshtein(pred[i], Y[i])
    return c/len(Y)


def levenshtein(s, t):

    rows = len(s)+1
    cols = len(t)+1
    dist = [[0 for x in range(cols)] for x in range(rows)]
    # source prefixes can be transformed into empty strings
    # by deletions:
    for i in range(1, rows):
        dist[i][0] = i
    # target prefixes can be created from an empty source string
    # by inserting the characters
    for i in range(1, cols):
        dist[0][i] = i

    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0
            else:
                cost = 1
            dist[row][col] = min(dist[row-1][col] + 1,      # deletion
                                 dist[row][col-1] + 1,      # insertion
                                 dist[row-1][col-1] + cost)  # substitution
    return dist[rows-1][cols-1]


def training(net, train_data, train_labels, dev_data, dev_labels, num_epochs, minibatch_size, learn_rate, vocab):
    # data = datatools.normalize(data)
    print(torch.cuda.is_available())
    dev1_data, dev1_labels = dev_data[:-20], dev_labels[:-20]
    dev2_data, dev2_labels = dev_data[-20:], dev_labels[-20:]
    # data, labels, lookup = datatools.pad(data, labels, k)

    def save_path(x): return "../net"+str(x)+".pt"

    def sub_path(x): return "../sub"+str(x)+".csv"

    # my_net = TwoLayerClassifier(nFreqs,nStates)
    # loss_fn = torch.nn.BCELoss()
    loss_fn = torch.nn.CrossEntropyLoss(size_average=False, reduce=False)
    # optim = torch.optim.SGD(net.parameters(), lr=learn_rate, momentum=0.9)
    optim = torch.optim.Adam(net.parameters(), lr=learn_rate, weight_decay=0.00001)

    for k in range(num_epochs):
        net.train()
        if torch.cuda.is_available():
            net = net.cuda()
            loss_fn = loss_fn.cuda()

        counter = 0
        print("epoch "+str(k))
        a = np.arange(len(train_labels))
        np.random.shuffle(a)
        full_loss = 0
        for j in range(0, len(a)-minibatch_size+1, minibatch_size):

            input_val, label = train_data[a[j:j+minibatch_size]
                                          ], train_labels[a[j:j+minibatch_size]]

            input_val, lengths, perm = sort_sequence(input_val)
            assert((np.array([len(seq) for seq in input_val]) == lengths).all())
            label = label[perm]
            input_val = [datatools.to_variable(datatools.to_tensor(seq))
                         for seq in input_val]
            input_val = pad_sequence(input_val, batch_first=False)
            # print(lengths)

            # label_lengths = np.array([1])
            label_lengths = np.array([len(l) for l in label])
            max_label_lengths = label_lengths.max()
            # print(max_label_lengths)
            padded_label = np.zeros((len(label), max_label_lengths))
            for i in range(len(label)):
                padded_label[i, :label_lengths[i]] = label[i]
            label = datatools.to_variable(datatools.to_tensor(
                padded_label), cuda=True).long().t()
            # print(label)
            # print(padded_label)
            label_to_mask = torch.Tensor((np.arange(label.size()[0]-1).reshape(
                -1, 1) >= (label_lengths-1).reshape((-1, label.size(1)))).astype(int)).cuda().byte()
            # print(label_to_mask)

            prediction = net(input_val, lengths, label[:-1, :], mode="train")

            # print(label.data.cpu().numpy())
            # print(np.unique(label.data.cpu().numpy())) # [1-46] as expected
            # print(type(prediction.data), type(label.data), type(
            #     output_lengths.data), type(label_lengths.data), type(loss_fn))
            #print(prediction.size(), label[:, 1:].size())
            # print(prediction.size(), label[1:, :].size())
            loss = loss_fn(prediction.contiguous().view(label.size(
                1)*(label.size(0)-1), -1), label[1:, :].contiguous().view(-1))

            loss = loss.view(label.size(0)-1, label.size(1))
            # print(loss.size(), label_to_mask.size())
            loss[label_to_mask] = 0
            # print(loss)
            loss = (torch.sum(loss))/minibatch_size
            # loss = loss_fn(prediction[:1, :1],
            #               label[:1], output_lengths[:1]/output_lengths[0], label_lengths)

            # print(loss/minibatch_size)
            # print(prediction.size(), output_lengths)
            counter += 1
            full_loss += loss.data.cpu().numpy()[0]
            loss.backward()
            optim.step()
            optim.zero_grad()
            if j/minibatch_size % 50 == 0:
                # print(prediction[0, 0])
                print(j, "sentences ; local loss "+str(loss.data.cpu().numpy()
                                                       [0]))
                # print(prediction, label)
                # for param in net.parameters():
                #    for k in range(param.size(0)):
                #        print(k, param[0])
        print("Average training loss :", full_loss/counter)
        net.eval()
        print("Saving net")
        save_net(net, save_path(k))
    # dev loss:
        dev_loss = 0
        counter = 0
        prev_dev_loss = float("Inf")
        for j in range(0, len(dev1_data)-minibatch_size+1, minibatch_size):

            input_val, label = dev1_data[j:j+minibatch_size
                                         ], dev1_labels[j:j+minibatch_size]

            input_val, lengths, perm = sort_sequence(input_val)
            assert((np.array([len(seq) for seq in input_val]) == lengths).all())
            label = label[perm]
            input_val = [datatools.to_variable(datatools.to_tensor(seq))
                         for seq in input_val]
            input_val = pad_sequence(input_val, batch_first=False)
            # print(lengths)

            # label_lengths = np.array([1])
            label_lengths = np.array([len(l) for l in label])
            max_label_lengths = label_lengths.max()
            # print(max_label_lengths)
            padded_label = np.zeros((len(label), max_label_lengths))
            for i in range(len(label)):
                padded_label[i, :label_lengths[i]] = label[i]
            label = datatools.to_variable(datatools.to_tensor(
                padded_label), cuda=True).long().t()
            # print(label)
            # print(padded_label)
            label_to_mask = torch.Tensor((np.arange(label.size()[0]-1).reshape(
                -1, 1) >= (label_lengths-1).reshape((-1, label.size(1)))).astype(int)).cuda().byte()
            # print(label_to_mask)

            prediction = net(input_val, lengths, label[:-1, :], mode="train")

            # print(label.data.cpu().numpy())
            # print(np.unique(label.data.cpu().numpy())) # [1-46] as expected
            # print(type(prediction.data), type(label.data), type(
            #     output_lengths.data), type(label_lengths.data), type(loss_fn))
            #print(prediction.size(), label[:, 1:].size())
            # print(prediction.size(), label[1:, :].size())
            loss = loss_fn(prediction.contiguous().view(label.size(
                1)*(label.size(0)-1), -1), label[1:, :].contiguous().view(-1))

            loss = loss.view(label.size(0)-1, label.size(1))
            # print(loss.size(), label_to_mask.size())
            loss[label_to_mask] = 0
            # print(loss)
            loss = (torch.sum(loss))/minibatch_size
            # loss = loss_fn(prediction[:1, :1],
            #               label[:1], output_lengths[:1]/output_lengths[0], label_lengths)

            # print(loss/minibatch_size)
            # print(prediction.size(), output_lengths)
            counter += 1
            dev_loss += loss.data.cpu().numpy()[0]

        if j/minibatch_size % 50 == 0:
            # print(prediction[0, 0])
            print(j, "sentences ; local loss "+str(loss.data.cpu().numpy()
                                                   [0]))
            # print(prediction, label)
            # for param in net.parameters():
            #    for k in range(param.size(0)):
            #        print(k, param[0])
        print("Average dev loss :", dev_loss/counter)
        if(dev_loss > prev_dev_loss):
            print("Stopping training")
            break
        prev_dev_loss = dev_loss
        val_pred = predict(net, dev2_data, vocab)
        print("heldout score : "+str(score(val_pred, dev2_labels, vocab)))


def save_net(net, path):
    torch.save(net, path)


def load_net(path):
    return torch.load(path)


def write_sub(net, Xt, sub_path, vocab):
    #testX = datatools.normalize(testX)
    testY = predict(net, Xt, vocab)
    print("prediction shape : "+str(testY.shape))
    test_sub = np.array([np.arange(testY.shape[0]), datatools.to_string_char(testY, vocab)]).T
    print(test_sub)
    np.savetxt(sub_path, test_sub, delimiter=",",
               header="id,label", comments="", fmt="%1s")


def stack_nets(path1, path2, path3):
    net1 = load_net(path1)
    net2 = load_net(path2)
    net3 = load_net(path3)

    net = net1

    d = net.state_dict()
    d1 = net1.state_dict()
    d2 = net2.state_dict()
    d3 = net3.state_dict()
    for key in d.keys():
        d[key] = (d1[key]+d2[key]+d3[key])/3
    net.load_state_dict(d)

    return net
