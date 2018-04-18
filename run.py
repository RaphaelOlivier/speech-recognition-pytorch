import datatools
import models
import routine

vocab = datatools.init_vocab()

X, Y = datatools.load_train_data_char(vocab)
Xd, Yd = datatools.load_dev_data_char(vocab)
Xt = datatools.load_test_data()

data, labels = X, Y
#data, labels = X[:200], Y[:200]

print(len(vocab))


def save_path(x): return "../net"+str(x)+".pt"


def sub_path(x): return "../sub"+str(x)+".csv"


net = models.Baseline(nLabels=len(vocab))
net = routine.load_net("../net0.pt").cuda()
net.max_sentence = 275

net.train()
routine.training(net, data, labels, Xd, Yd, 20, 16, 0.00001, vocab)

print("Writing submission")
net.eval()
routine.write_sub(net, Xt, sub_path(""), vocab, random=True, random_number=100)
