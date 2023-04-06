from typing import Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn import init


class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride:int=1,
                 padding:int=0, activation:Optional[str]='relu', norm:Optional[str]=None) -> None:
        super().__init__()

        bias = False if norm == 'BN' else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if activation is not None:
            self.activation: Callable = getattr(torch, activation, 'relu')  # type: ignore
        else:
            self.activation = None  # type: ignore

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2d(x)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out  # type: ignore


class TransposedConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int=1,
                 padding: int=0, activation: Optional[str]='relu', norm: Optional[str]=None) -> None:
        super().__init__()

        bias = False if norm == 'BN' else True
        self.transposed_conv2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=2, padding=padding, output_padding=1, bias=bias)

        if activation is not None:
            self.activation: Callable = getattr(torch, activation, 'relu')  # type: ignore
        else:
            self.activation = None  # type: ignore

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.transposed_conv2d(x)
        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)
        if self.activation is not None:
            out = self.activation(out)

        return out  # type: ignore


class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int=1,
                 padding: int=0, activation: Optional[str]='relu', norm: Optional[str]=None) -> None:
        super().__init__()

        bias = False if norm == 'BN' else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        if activation is not None:
            self.activation: Callable = getattr(torch, activation, 'relu')  # type: ignore
        else:
            self.activation = None  # type: ignore

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)   # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_upsampled = f.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.conv2d(x_upsampled)
        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)
        if self.activation is not None:
            out = self.activation(out)

        return out  # type: ignore


class RecurrentConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3,
                 stride: int=1, padding: int=0, recurrent_block_type: str='convlstm',
                 activation: Optional[str]='relu', norm: Optional[str]=None) -> None:
        super().__init__()

        assert(recurrent_block_type in ['convlstm', 'convgru'])
        self.recurrent_block_type = recurrent_block_type
        if self.recurrent_block_type == 'convlstm':
            RecurrentBlock = ConvLSTM
        else:
            raise NotImplementedError("convgru is not supported!")
        self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride, padding, activation, norm)
        self.recurrent_block = RecurrentBlock(input_size=out_channels, hidden_size=out_channels, kernel_size=3)

    def forward(self, x: torch.Tensor, prev_state: Optional[torch.Tensor]) -> Tuple[torch.Tensor,
                                                                                    torch.Tensor]:
        x = self.conv(x)
        state = self.recurrent_block(x, prev_state)
        x = state[0] if self.recurrent_block_type == 'convlstm' else state
        return x, state


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int=1,
                 downsample: Optional[Callable]=None, norm: Optional[str]=None) -> None:  # TODO: typehint downsample
        super().__init__()
        bias = False if norm == 'BN' else True
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.norm = norm
        if norm == 'BN':
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.bn1 = nn.InstanceNorm2d(out_channels)   # type: ignore
            self.bn2 = nn.InstanceNorm2d(out_channels)   # type: ignore

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        if self.norm in ['BN', 'IN']:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.norm in ['BN', 'IN']:
            out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out   # type: ignore


class ConvLSTM(nn.Module):
    """Adapted from: https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py """

    def __init__(self, input_size: int, hidden_size: int, kernel_size: int) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        pad = kernel_size // 2

        # cache a tensor filled with zeros to avoid reallocating memory at each inference step if --no-recurrent is enabled
        self.zero_tensors = {}  # type: ignore

        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=pad)

    def forward(self, input_: torch.Tensor, prev_state: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor,
                                                                                              torch.Tensor]:

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:

            # create the zero tensor if it has not been created already
            state_size = tuple([batch_size, self.hidden_size] + list(spatial_size))
            if state_size not in self.zero_tensors:
                # allocate a tensor with size `spatial_size`, filled with zero (if it has not been allocated already)
                self.zero_tensors[state_size] = (
                    torch.zeros(state_size).to(input_.device),
                    torch.zeros(state_size).to(input_.device)
                )

            prev_state = self.zero_tensors[tuple(state_size)]

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell
