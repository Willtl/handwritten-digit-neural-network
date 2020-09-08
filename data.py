import torch
import torchvision
from torchvision import transforms, datasets


def read():
    train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    test = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

    trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
    testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

    return trainset, testset


def plot():
    trainset, testset = read()

    # Data manipulation
    for data in trainset:
        print(data)
        break
    x, y = data[0][0], data[1][0]
    print(x, y)

    # Plotting one arbitrary image
    import matplotlib.pyplot as plt
    plt.imshow(data[0][0].view(28, 28))
    plt.show()


def stuff():
    # Create an array of 2 by 5
    x = torch.Tensor(2, 5)

    # Create a random array of 2 by 5
    y = torch.rand([2, 5])

    # Reshape (view it as) an array of 1 by 10
    y = y.view([1, 10])