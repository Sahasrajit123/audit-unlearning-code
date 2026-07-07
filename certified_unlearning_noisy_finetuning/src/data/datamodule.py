from src.data.cifar_dataset import CIFAR10DataModule, CIFAR100DataModule
from src.data.mnist_dataset import MNISTDataModule
from src.data.uci_toy_datasets import IRISDataModule, WineDataModule, BreastCancerDataModule, RCV1DataModule

def get_datamodule(config, generator):
    if config["dataset"]["name"] == "cifar10":
        return CIFAR10DataModule(config, generator)
    elif config["dataset"]["name"] == "cifar100":
        return CIFAR100DataModule(config, generator)
    elif config["dataset"]["name"] == "mnist":
        return MNISTDataModule(config, generator)
    elif config["dataset"]["name"] == "iris":
        return IRISDataModule(config, generator)
    elif config["dataset"]["name"] == "wine":
        return WineDataModule(config, generator)
    elif config["dataset"]["name"] == "breast_cancer":
        return BreastCancerDataModule(config, generator)
    elif config["dataset"]["name"] == "rcv1":
        return RCV1DataModule(config, generator)
    elif config["dataset"]["name"] == "cifar10_feature":
        from src.data.cifar_dataset import CIFAR10FeatureDataModule
        return CIFAR10FeatureDataModule(config, generator)
    elif config["dataset"]["name"] == "cifar100_feature":
        from src.data.cifar_dataset import CIFAR100FeatureDataModule
        return CIFAR100FeatureDataModule(config, generator)
    else:
        raise ValueError("Unknown dataset")
