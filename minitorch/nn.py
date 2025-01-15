from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    new_height = height // kh
    new_width = width // kw
    output = input.contiguous().view(batch, channel, height, new_width, kw)
    output = output.permute(0, 1, 3, 2, 4)
    output = output.contiguous().view(batch, channel, new_width, new_height, kh * kw)
    output = output.permute(0, 1, 3, 2, 4)
    return output, new_height, new_width


# Define avgpool2d
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D"""
    batch, channel, height, width = input.shape
    input, new_height, new_width = tile(input, kernel)
    output = input.mean(dim=4)
    return output.view(batch, channel, new_height, new_width)


# TODO: Implement for Task 4.4.
max_reduce = FastOps.reduce(operators.max, -1e6)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor"""
    output = max_reduce(input, dim)
    return output == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Compute the maximum values along the specified dimension of the input tensor."""
        ctx.save_for_backward(input, int(dim[0]))
        return max_reduce(input, int(dim[0]))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute the gradient of the `Max` function with respect to the input tensor."""
        input, dim = ctx.saved_values
        return (argmax(input, dim)) * grad_output, 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction"""
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor"""
    exp_input = input.exp()
    return exp_input / exp_input.sum(dim=dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor"""
    max_i = max(input, dim)
    output = input - (input - max_i).exp().sum(dim=dim).log() - max_i
    return output


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D"""
    batch, channel, height, width = input.shape
    input, new_height, new_width = tile(input, kernel)
    return max(input, dim=4).view(batch, channel, new_height, new_width)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise, include an argument to turn off"""
    if ignore:
        return input
    else:
        return input * (rand(input.shape) > rate)
