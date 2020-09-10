import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    # Define ANN's layer architecture
    def __init__(self):
        # Initialize superclass
        super().__init__()
        # Fully conected layers (ann architecture)
        self.inputs = 28 * 28
        self.outputs = 10
        self.l1 = nn.Linear(self.inputs, 64)  # To disable bias use bias=False
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

    def feed(self, x):
        # x = torch.rand((28, 28))
        # x = x.view(1, 28 * 28)
        outputs = self.foward(x)
        return outputs

    def train_net(self, trainset):
        learning_rate = 0.001
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        epochs = 10
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in trainset:
                # the inputs
                x, y = batch
                # zero the parameter gradients
                self.zero_grad()
                # Foward
                outputs = self.feed(x.view(-1, 28 * 28))
                # For [0, 1, 0, 0] vectors, use mean squared error, for scalar values use nll_loss
                loss = F.nll_loss(outputs, y)
                # Back propagate the loss
                loss.backward()
                # Adjust the weights
                optimizer.step()
                # Calculate epoch loss
                epoch_loss += outputs.shape[0] * loss.item()
            print("Epoch loss: ", epoch_loss / len(trainset))

    def test_net(self, testset):
        # Deactivate Dropout and BatchNorm
        self.eval()

        correct = 0
        total = 0
        # Deactivate gradient calculations
        with torch.no_grad():
            # Check each batch in testset
            for batch in testset:
                x, y, = batch
                outputs = self.feed(x.view(-1, 28 * 28))
                # Loop through outputs and check if it is correct or not
                for idx, i in enumerate(outputs):
                    if torch.argmax(i) == y[idx]:
                        correct += 1
                    total += 1
            print("Accuracy: ", round(correct/total, 3))

        # Activate Dropout and BatchNorm again
        self.train()

    def update_weights(self):
        print(self.l1.weight)
        for i in range(64):
            print(self.l1.weight[i])
        # for j in range(28 * 28):
        #     print(self.l1.weight[i][j])
        weight = nn.Parameter(torch.ones_like(self.l1.weight))
        print(weight)

    def print(self):
        print(self)