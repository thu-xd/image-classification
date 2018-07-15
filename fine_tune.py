from experiment import ex
from data_fetch import build_dataset
from AlexNet.AlexNet import AlexNet

@ex.automain
def main():
    train_dataset,valid_dataset,num_examples_per_epoch=build_dataset()
    model=AlexNet(train_dataset,valid_dataset,num_examples_per_epoch)
    model.train()


