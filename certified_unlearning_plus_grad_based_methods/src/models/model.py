import jax.numpy as jnp

from src.models.resnet import ResNet18
from src.models.simplenet import SimpleNet
from src.models.two_layer_net import TwoLayerNN
from src.models.two_layer_net import ThreeLayerNN
from src.models.tiny_net import TinyNet
from src.models.tiny_net import TinierNet
# from src.models.tiny_net import TinierNetCIFAR10
from src.models.tiny_net import CIFAR10TinykNet, CIFAR100TinyNet
from src.models.logistic_regression import LogisticRegression

class ModelFactory:
    @staticmethod
    def create_model(model_name, num_classes):
        if model_name == "simplenet":
            model_cls = SimpleNet(num_classes=num_classes)
        elif model_name == "resnet":
            model_cls = ResNet18(num_classes=num_classes, dtype=jnp.float32)
        elif model_name == "two_layer_net":
            model_cls = TwoLayerNN(num_classes=num_classes)
        elif model_name == "three_layer_net":
            model_cls = ThreeLayerNN(num_classes=num_classes)   
        elif model_name == "tiny_net":
            model_cls = TinyNet(num_classes=num_classes)
        elif model_name == "tinier_net":
            model_cls = TinierNet(num_classes=num_classes)
        elif model_name == "tiny_net_cifar":
            # model_cls = TinierNetCIFAR10(num_classes=num_classes)
            model_cls = CIFAR10TinykNet(num_classes=num_classes)
        elif model_name == "tiny_net_cifar100":
            model_cls = CIFAR100TinyNet(num_classes=num_classes)
        elif model_name == "logistic_regression":
            model_cls = LogisticRegression(num_classes=num_classes)
        else:
            raise ValueError(f"Model {model_name} not supported")
        return model_cls
