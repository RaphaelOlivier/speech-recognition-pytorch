import datatools
import models
import routine
import wsj_loader
Xt = datatools.load_test_data()

Xt, lt = datatools.reshape_data(Xt)


net = routine.stack_nets("../lastnetB27.pt", "../lastnetB28.pt", "../lastnetB29.pt")
print("Saving net")
routine.save_net(net, "../stackednet.pt")
print("Writing submission")
net.eval()
routine.write_sub(net, Xt, lt, "../stackedsub.csv")
