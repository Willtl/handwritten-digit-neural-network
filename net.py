import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    # Define ANN's layer architecture
    def __init__(self):
        # Initialize superclass
        super().__init__()
        # Fully conected layers (ann architecture)
        self.inputs = 28 * 28
        self.outputs = 10
        self.l1 = nn.Linear(self.inputs, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 64)
        self.l4 = nn.Linear(64, self.outputs)

    # Define how the data passes through the layers
    def foward(self, x):
        # Passes x through layer one and activate with rectified linear unit function
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        # Normalize the output of the network to a probability distribution over predicted output classes
        x = F.log_softmax(self.l4(x), dim=1)
        return x

    def feed(self):
        x = torch.rand((28, 28))
        print(x)
        x = x.view(1, 28 * 28)
        print(x)
        outputs = self.foward(x)
        print(outputs)

    def print(self):
        print(self)