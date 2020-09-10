import torch
import data
import net


def feed_random(net):
    x = torch.rand((28, 28))
    x = x.view(1, 28 * 28)
    outputs = net.feed(x)
    print(outputs)


trainset, testset = data.read()
net = net.Net()
net.train_net(trainset)
net.test_net(testset)