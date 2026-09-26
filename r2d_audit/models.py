import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.autograd import Variable


class Affine(nn.Module):

    def __init__(self, num_features): 
        super().__init__()
        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, x):
        return x * self.weight + self.bias
    
class StandardLinearLayer(nn.Linear): 
    
    def __init__(self, in_features, out_features, bias=True, beta=np.sqrt(0.1), w_sig = np.sqrt(2.0)):
        self.beta = beta
        self.w_sig = w_sig
        super(StandardLinearLayer, self).__init__(in_features, out_features)
        self.reset_parameters()
      
    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, mean=0, std=self.w_sig/np.sqrt(self.in_features))
        if self.bias is not None:
            torch.nn.init.normal_(self.bias, mean=0, std=self.beta)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, beta={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.beta)
    



class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self,x):
        return x.view(x.size(0), -1)
    
class ConvStandard(nn.Conv2d): 
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0, w_sig =\
                 np.sqrt(1.0)):
        super(ConvStandard, self).__init__(in_channels, out_channels,kernel_size)
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.w_sig = w_sig
        self.reset_parameters()
      
    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, mean=0, std=self.w_sig/(self.in_channels*np.prod(self.kernel_size)))
        if self.bias is not None:
            torch.nn.init.normal_(self.bias, mean=0, std=0)
            
    def forward(self, input):
        return F.conv2d(input,self.weight,self.bias,self.stride,self.padding)
            
class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0,
                 activation_fn=nn.ReLU, batch_norm=True, transpose=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        model = []
        if not transpose:
#             model += [ConvStandard(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
#                                 )]
            model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=not batch_norm)]
        else:
            model += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                         output_padding=output_padding, bias=not batch_norm)]
        if batch_norm:
            model += [nn.BatchNorm2d(out_channels, affine=True)]
        model += [activation_fn()]
        super(Conv, self).__init__(*model)


    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class _ResBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(_ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out
    
class ResNet18(nn.Module):
    def __init__(self, filters_percentage=1.0, n_channels = 3, num_classes=10, block=_ResBlock, num_blocks=[2,2,2,2]):
        super(ResNet18, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(n_channels,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, int(64*filters_percentage), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(128*filters_percentage), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(256*filters_percentage), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, int(512*filters_percentage), num_blocks[3], stride=2)
        self.linear = nn.Linear(int(512*filters_percentage)*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class SmeLU(nn.Module):
    """
    This class implements the Smooth ReLU (SmeLU) activation function proposed in:
    https://arxiv.org/pdf/2202.06499.pdf

    #https://github.com/ChristophReich1996/SmeLU/blob/master/smelu/smelu.py
    """

    def __init__(self, beta: float = 0.5) -> None:
        """
        Constructor method.
        :param beta (float): Beta value if the SmeLU activation function. Default 2.
        """
        # Call super constructor
        super(SmeLU, self).__init__()
        # Check beta
        assert beta >= 0., f"Beta must be equal or larger than zero. beta={beta} given."
        # Save parameter
        self.beta: float = beta

    def __repr__(self) -> str:
        """
        Returns a string representation.
        :return (str): String representation
        """
        return f"SmeLU(beta={self.beta})"

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        :param input (torch.Tensor): Tensor of any shape
        :return (torch.Tensor): Output activation tensor of the same shape as the input tensor
        """
        output: torch.Tensor = torch.where(input >= self.beta, input,
                                           torch.tensor([0.], device=input.device, dtype=input.dtype))
        output: torch.Tensor = torch.where(torch.abs(input) <= self.beta,
                                           ((input + self.beta) ** 2) / (4. * self.beta), output)
        return output

class _ResBlocksmooth(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(_ResBlocksmooth, self).__init__()
        #self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride=stride)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )
        self.smelu = SmeLU()
    def forward(self, x):
        out = self.smelu(x)
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.smelu(out))
        out += shortcut
        return out


class MLP(nn.Module):
    def __init__(self, input_size=467, filters_percentage=None, num_classes=2):
        super(MLP, self).__init__() #allows you to call methods from the parent class nn.module, __init__() initializes for nn.module

        self.l1 = nn.Linear(input_size, 256)
        self.l2 = nn.Linear(256, 64)
        self.l5 = nn.Linear(64, num_classes)

    def forward(self, x):    
        model = torch.nn.Sequential(
            self.l1,
            SmeLU(),
            self.l2,
            SmeLU(),
            self.l5
            #nn.Softmax(dim=-1)
        )
        return model(x)



class ResNet18smooth(nn.Module):# NO RELU
    def __init__(self, filters_percentage=1.0, n_channels = 3, num_classes=10, block=_ResBlocksmooth, num_blocks=[2,2,2,2]):
        super(ResNet18smooth, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(n_channels,64)
        #self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, int(64*filters_percentage), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(128*filters_percentage), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(256*filters_percentage), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, int(512*filters_percentage), num_blocks[3], stride=2)
        self.linear = nn.Linear(int(512*filters_percentage)*block.expansion, num_classes)

        self.smelu = SmeLU()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        #out = F.sigmoid(self.bn1(self.conv1(x)))
        out = self.smelu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    

    
_MODELS = {}

def _add_model(model_fn):
    _MODELS[model_fn.__name__] = model_fn
    return model_fn

@_add_model
def mlp(**kwargs):
    return MLP(**kwargs)


@_add_model
def resnet(**kwargs):
    return ResNet18(**kwargs)

@_add_model
def resnetsmooth(**kwargs):
    return ResNet18smooth(**kwargs)


class TinyNetArchive(nn.Module):
    """
    [Archived] Previous TinyNet - A small CNN for CIFAR-10.
    Lightweight model suitable for fast training and unlearning experiments.
    Kept for backward compatibility with old checkpoints; use TinyNet (certified-deep-unlearning definition) for new runs.
    """
    def __init__(self, filters_percentage=1.0, n_channels=3, num_classes=10):
        super(TinyNetArchive, self).__init__()
        
        # Scale channels by filters_percentage
        base_channels = 32
        c1 = int(base_channels * filters_percentage)
        c2 = int(base_channels * 2 * filters_percentage)
        c3 = int(base_channels * 4 * filters_percentage)
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(n_channels, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
            
            # Block 2
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8
            
            # Block 3
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8x8 -> 4x4
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            Flatten(),
            nn.Linear(c3, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class TinyNet(nn.Module):
    """
    Lightweight CNN for CIFAR-10 (converted from JAX/Flax).
    Definition from certified-deep-unlearning/model.py.
    Architecture:
    - Conv 32 filters, 3x3, padding=same, ReLU, AvgPool 2x2
    - Conv 64 filters, 3x3, padding=same, ReLU, AvgPool 2x2
    - Global Average Pooling
    - Dense layer to num_classes
    """
    def __init__(self, num_classes=10, filters_percentage=1.0, channels=3):
        super(TinyNet, self).__init__()
        self.num_classes = num_classes

        # Calculate actual filter counts based on percentage
        filters_32 = int(32 * filters_percentage)
        filters_64 = int(64 * filters_percentage)

        # First convolutional block
        self.conv1 = nn.Conv2d(channels, filters_32, kernel_size=3, padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Second convolutional block
        self.conv2 = nn.Conv2d(filters_32, filters_64, kernel_size=3, padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Classifier
        self.fc = nn.Linear(filters_64, num_classes)
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # First block: Conv -> ReLU -> AvgPool
        x = self.conv1(x)
        x = F.relu(x)
        x = self.avgpool1(x)

        # Second block: Conv -> ReLU -> AvgPool
        x = self.conv2(x)
        x = F.relu(x)
        x = self.avgpool2(x)

        # Global average pooling (mean over spatial dimensions)
        x = x.mean(dim=(2, 3))  # Reduces from [B, C, H, W] to [B, C]

        # Classifier
        x = self.fc(x)

        return x


@_add_model
def tinynet(**kwargs):
    # Support n_channels for callers that use it (maps to channels)
    if 'n_channels' in kwargs and 'channels' not in kwargs:
        kwargs = dict(kwargs)
        kwargs['channels'] = kwargs.pop('n_channels')
    return TinyNet(**kwargs)


@_add_model
def tinynet_archive(**kwargs):
    return TinyNetArchive(**kwargs)


class TinyNetCIFAR100(nn.Module):
    """
    CIFAR-100 TinyNet aligned with the provided JAX architecture.

    Architecture:
    - Conv(64) + GroupNorm(8) + ReLU
    - Conv(64) + GroupNorm(8) + ReLU + MaxPool(2)
    - Conv(128) + GroupNorm(8) + ReLU
    - Conv(128) + GroupNorm(8) + ReLU + MaxPool(2)
    - Global Average Pooling
    - Linear(num_classes)
    """

    def __init__(self, num_classes=100, filters_percentage=1.0, channels=3):
        super(TinyNetCIFAR100, self).__init__()
        self.num_classes = num_classes

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
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
        x = self.fc(x)
        return x


@_add_model
def tinynetcifar100(**kwargs):
    if 'n_channels' in kwargs and 'channels' not in kwargs:
        kwargs = dict(kwargs)
        kwargs['channels'] = kwargs.pop('n_channels')
    return TinyNetCIFAR100(**kwargs)


def get_model(name, **kwargs):
    return _MODELS[name](**kwargs)
