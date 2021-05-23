import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def hyperfan_out_W_init(tensor: Tensor, noise_var: float = 1., n_noise: int = 1, relu: bool = False) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)

    if relu:
        relu_term = 2.0
    else:
        relu_term = 1.0

    desired_var = relu_term / (fan_in * n_noise * noise_var)
    bound = math.sqrt(desired_var)
    
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def hyperfan_out_b_init(tensor: Tensor, noise_var: float = 1., w_j: int = 1, relu: bool = False) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)

    if relu:
        relu_term = 2.0
    else:
        relu_term = 1.0

    desired_var = relu_term * (1.0 - (w_j / fan_out)) / (fan_in * noise_var)
    desired_var = max(desired_var, 0.0)

    bound = math.sqrt(desired_var)
    
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def hyperfan_in_W_init(tensor: Tensor, noise_var: float = 1., n_noise: int = 1, bias: bool = True, relu: bool = False) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)

    if relu:
        relu_term = 2.0
    else:
        relu_term = 1.0

    if bias:
        bias_term = 2.0
    else:
        bias_term = 1.0

    desired_var = relu_term / (bias_term * fan_in * n_noise * noise_var)
    bound = math.sqrt(desired_var)
    
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def hyperfan_in_b_init(tensor: Tensor, noise_var: float = 1., relu: bool = False) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)

    if relu:
        relu_term = 2.0
    else:
        relu_term = 1.0

    desired_var = relu_term / (2.0 * fan_in * noise_var)

    bound = math.sqrt(3.0 * desired_var)
    
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def fanin_uniform(tensor: Tensor):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    bound = math.sqrt(1.0 / fan_in)

    with torch.no_grad():
        return tensor.uniform_(-bound, bound)
