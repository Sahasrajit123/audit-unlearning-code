# code taken from https://github.com/n2cholas/jax-resnet/blob/main/jax_resnet/pretrained.py

import torchvision
from functools import partial
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from flax.core import FrozenDict, freeze
from flax.traverse_util import unflatten_dict
import jax.numpy as jnp
import src.models.resnet as resnet

ModuleDef = Callable[..., Callable]
PyTorchTensor = Any


STAGE_SIZES = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
    200: [3, 24, 36, 3],
}

def pretrained_resnet(
    size: int,
    state_dict: Optional[Mapping[str, PyTorchTensor]] = None
) -> Tuple[ModuleDef, FrozenDict]:
    """Returns pretrained variables for ResNet ported from torch.hub.

    Args:
        size: 18, 34, 50, 101 or 152.
        state_dict: If provided, this state dict will be used over the
            pretrained torch.hub model. The keys must match the torch.hub resnet.

    Returns:
        Module Class and variables dictionary for Flax ResNet.
    """
    if size not in (18, 34, 50, 101, 152):
        raise ValueError('Ensure size is one of (18, 34, 50, 101, 152)')

    if state_dict is None:
        state_dict = getattr(torchvision.models, f'resnet{size}')(pretrained=True).state_dict()

    pt2jax: Dict[str, Sequence[str]] = {}
    add_bn = _get_add_bn(pt2jax)

    pt2jax['conv1.weight'] = ('params', 'layers_0', 'ConvBlock_0', 'Conv_0', 'kernel')
    add_bn('bn1', ('layers_0', 'ConvBlock_0', 'BatchNorm_0'))

    lyr = 2  # block_ind
    for b, n_blocks in enumerate(STAGE_SIZES[size], 1):
        for i in range(n_blocks):
            for j in range(2 + (size >= 50)):
                pt2jax[f'layer{b}.{i}.conv{j+1}.weight'] = ('params', f'layers_{lyr}',
                                                            f'ConvBlock_{j}', 'Conv_0',
                                                            'kernel')
                add_bn(f'layer{b}.{i}.bn{j+1}',
                       (f'layers_{lyr}', f'ConvBlock_{j}', 'BatchNorm_0'))

            if f'layer{b}.{i}.downsample.0.weight' in state_dict:
                pt2jax[f'layer{b}.{i}.downsample.0.weight'] = ('params',
                                                               f'layers_{lyr}',
                                                               'ResNetSkipConnection_0',
                                                               'ConvBlock_0', 'Conv_0',
                                                               'kernel')
                add_bn(f'layer{b}.{i}.downsample.1',
                       (f'layers_{lyr}', 'ResNetSkipConnection_0', 'ConvBlock_0',
                        'BatchNorm_0'))

            lyr += 1

    lyr += 1
    pt2jax['fc.weight'] = ('params', f'layers_{lyr}', 'kernel')
    pt2jax['fc.bias'] = ('params', f'layers_{lyr}', 'bias')

    variables = _pytorch_to_jax_params(pt2jax, state_dict, ('fc.weight',))
    model_cls = partial(getattr(resnet, f'ResNet{size}'), n_classes=1000)
    # print(variables)
    # print shape of each variable
    for k, v in variables.items():
        print(f'{k}: {v.shape}')
    return model_cls, freeze(unflatten_dict(variables))



def _pytorch_to_jax_params(pt2jax, state_dict, fc_keys):
    variables = {}
    for pt_name, jax_key in pt2jax.items():
        w = state_dict[pt_name].numpy()
        if w.ndim == 4:
            w = w.transpose((2, 3, 1, 0))
        elif pt_name in fc_keys:
            w = w.transpose()
        variables[jax_key] = w

    return variables


def _get_add_bn(pt2jax):
    def add_bn(pname, jprefix):
        pt2jax[f'{pname}.weight'] = ('params', *jprefix, 'scale')
        pt2jax[f'{pname}.bias'] = ('params', *jprefix, 'bias')
        pt2jax[f'{pname}.running_mean'] = ('batch_stats', *jprefix, 'mean')
        pt2jax[f'{pname}.running_var'] = ('batch_stats', *jprefix, 'var')

    return add_bn