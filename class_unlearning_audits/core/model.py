"""
Model architectures.

Only the architectures actually used by train_and_unlearn.py are kept here (MLP for
MNIST, TinyNetCIFAR100/AllCNN/ResNet18 for image datasets). `build_model` is a small
factory so the training script and the unlearning script can each build a fresh,
architecturally-identical model from the same set of arguments -- the unlearning script
needs this to construct the randomly-initialized "incompetent teacher".
"""
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, num_layer=3, num_classes=10, filters_percentage=1., hidden_size=32, input_size=784):
        super().__init__()
        self.input_size = input_size
        self.num_layer = num_layer
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.layers = self._make_layers()

    def _make_layers(self):
        layer = [nn.Linear(self.input_size, self.hidden_size), nn.ReLU()]
        for _ in range(self.num_layer - 2):
            layer += [nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU()]
        layer += [nn.Linear(self.hidden_size, self.num_classes)]
        return nn.Sequential(*layer)

    def forward(self, x):
        x = x.reshape(x.size(0), self.input_size)
        return self.layers(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):
    def forward(self, x):
        return x


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None,
                 activation_fn=nn.ReLU, batch_norm=True):
        if padding is None:
            padding = (kernel_size - 1) // 2
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                             padding=padding, bias=not batch_norm)]
        if batch_norm:
            layers += [nn.BatchNorm2d(out_channels, affine=True)]
        layers += [activation_fn()]
        super().__init__(*layers)


class AllCNN(nn.Module):
    def __init__(self, filters_percentage=1., n_channels=3, num_classes=10, dropout=False, batch_norm=True):
        super().__init__()
        n_filter1 = int(96 * filters_percentage)
        n_filter2 = int(192 * filters_percentage)
        self.features = nn.Sequential(
            Conv(n_channels, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),
            nn.Dropout(inplace=True) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),
            nn.Dropout(inplace=True) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=1, stride=1, batch_norm=batch_norm),
            nn.AvgPool2d(8),
            Flatten(),
        )
        self.classifier = nn.Sequential(nn.Linear(n_filter2, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class _ResBlock(nn.Module):
    """Pre-activation basic residual block."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + shortcut


class ResNet18(nn.Module):
    def __init__(self, filters_percentage=1.0, n_channels=3, num_classes=10, num_blocks=(2, 2, 2, 2)):
        super().__init__()
        self.in_planes = 64
        self.conv1 = conv3x3(n_channels, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(int(64 * filters_percentage), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(int(128 * filters_percentage), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(int(256 * filters_percentage), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(int(512 * filters_percentage), num_blocks[3], stride=2)
        self.linear = nn.Linear(int(512 * filters_percentage) * _ResBlock.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(_ResBlock(self.in_planes, planes, s))
            self.in_planes = planes * _ResBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return self.linear(out)


class TinyNet(nn.Module):
    """Lightweight CNN for CIFAR-10-sized (3x32x32) inputs."""

    def __init__(self, num_classes=10, filters_percentage=1.0, channels=3):
        super().__init__()
        f32 = int(32 * filters_percentage)
        f64 = int(64 * filters_percentage)
        self.conv1 = nn.Conv2d(channels, f32, kernel_size=3, padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(f32, f64, kernel_size=3, padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="relu")
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(f64, num_classes)
        nn.init.kaiming_normal_(self.fc.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.avgpool1(F.relu(self.conv1(x)))
        x = self.avgpool2(F.relu(self.conv2(x)))
        x = x.mean(dim=(2, 3))
        return self.fc(x)


class TinyNetCIFAR100(nn.Module):
    """Small CNN for CIFAR-100-sized (3x32x32) inputs, 100-way classification."""

    def __init__(self, num_classes=100, filters_percentage=1.0, channels=3):
        super().__init__()
        f64 = int(64 * filters_percentage)
        f128 = int(128 * filters_percentage)
        self.conv1 = nn.Conv2d(channels, f64, kernel_size=3, padding=1, bias=True)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=f64)
        self.conv2 = nn.Conv2d(f64, f64, kernel_size=3, padding=1, bias=True)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=f64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(f64, f128, kernel_size=3, padding=1, bias=True)
        self.gn3 = nn.GroupNorm(num_groups=8, num_channels=f128)
        self.conv4 = nn.Conv2d(f128, f128, kernel_size=3, padding=1, bias=True)
        self.gn4 = nn.GroupNorm(num_groups=8, num_channels=f128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(f128, num_classes)
        for m in [self.conv1, self.conv2, self.conv3, self.conv4, self.fc]:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.relu(self.gn2(self.conv2(x)))
        x = self.pool1(x)
        x = F.relu(self.gn3(self.conv3(x)))
        x = F.relu(self.gn4(self.conv4(x)))
        x = self.pool2(x)
        x = x.mean(dim=(2, 3))
        return self.fc(x)


_DATASET_INPUT_SIZE = {
    "mnist": 1 * 28 * 28,
    "cifar10": 3 * 32 * 32,
    "cifar": 3 * 32 * 32,
    "svhn": 3 * 32 * 32,
    "cifar100": 3 * 32 * 32,
}
_DATASET_NUM_CLASSES = {
    "mnist": 10,
    "cifar10": 10,
    "cifar": 10,
    "svhn": 10,
    "cifar100": 100,
}
_DATASET_DEFAULT_MODEL = {
    "mnist": "mlp",
    "cifar10": "tinynet",
    "cifar": "tinynet",
    "svhn": "tinynet",
    "cifar100": "tinynet_cifar100",
}


def dataset_defaults(dataset_name: str):
    """Return (input_size, num_classes, default_model_name) for a known dataset name."""
    key = dataset_name.lower()
    return (
        _DATASET_INPUT_SIZE.get(key),
        _DATASET_NUM_CLASSES.get(key, 10),
        _DATASET_DEFAULT_MODEL.get(key, "mlp"),
    )


def build_model(name: str, num_classes: int, input_size: int = None, filters_percentage: float = 1.0) -> nn.Module:
    """Factory so train.py and unlearn.py can build architecturally-identical fresh models."""
    name = name.lower()
    if name == "mlp":
        if input_size is None:
            raise ValueError("MLP requires input_size (e.g. 784 for 28x28 grayscale images).")
        return MLP(num_layer=3, num_classes=num_classes, filters_percentage=filters_percentage, input_size=input_size)
    if name == "tinynet":
        return TinyNet(num_classes=num_classes, filters_percentage=filters_percentage)
    if name in ("tinynet_cifar100", "tinynetcifar100"):
        return TinyNetCIFAR100(num_classes=num_classes, filters_percentage=filters_percentage)
    if name in ("cnn", "allcnn"):
        return AllCNN(num_classes=num_classes, filters_percentage=filters_percentage)
    if name == "resnet18":
        return ResNet18(num_classes=num_classes, filters_percentage=filters_percentage)
    raise ValueError(f"Unknown model: {name!r}. Choose from: mlp, tinynet, tinynet_cifar100, allcnn, resnet18")
