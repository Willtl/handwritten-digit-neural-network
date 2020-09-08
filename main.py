import data
import net

trainset, testset = data.read()

net = net.Net()
net.print()
net.feed()