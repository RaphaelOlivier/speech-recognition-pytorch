import datatools
import models
import routine
import wsj_loader

vocab = datatools.init_vocab()

X, Y = datatools.load_train_data_char(vocab)
Xd, Yd = datatools.load_dev_data_char(vocab)
Xt = datatools.load_test_data()


net = routine.stack_nets("../net33.pt", "../net34.pt", "../net35.pt").cuda()
net.max_sentence = 300
print("Saving net")
routine.save_net(net, "../stackednet.pt")
print("Writing submission")
net.eval()
routine.write_sub(net, Xt, "../stackedsub.csv", vocab, random=True, random_number=500)
