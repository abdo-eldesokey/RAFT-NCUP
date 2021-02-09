########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
import logging
import math


def retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=2)
    output = flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
    return output


class NConvUNet(nn.Module):
    def __init__(self, in_ch=1, channels_multiplier=2, num_downsampling=3, encoder_filter_sz=5,
                 decoder_filter_sz=3, out_filter_sz=1, pos_fn='SoftPlus', groups=1, use_bias=False,
                 data_pooling='conf_based', shared_encoder=True, use_double_conv=True):
        """
        Args:
            in_ch: Number of Input channels. Currently on 1 is supported
            channels_multiplier: Number of channels for intermediate is `in_ch` multiplied by this multiplier
            num_downsampling: Number of downsampling stages. The sparser the data, the larger this number should be.
                              In the original paper, they use 3
            encoder_filter_sz: Filter size for NConv layers in the encoder.
            decoder_filter_sz: Filter size for NConv layers in the decoder.
            out_filter_sz: Filter size for NConv layers in output layer.
            pos_fn: Non-negativity enforcement function. Options=[softplus, sigmoid, exp, softmax]
            groups: Convolution group. See `torch.nn.Conv2D` documentation
            use_bias: Convolution bias.
            data_pooling: How to perfrom pooling on the data stream. For the original paper approach, use `conf_based`
            shared_encoder: Use same layers at different scaled. For the original paper implementation. set to `True`
            use_double_conv: Either to use double convolution at each layer or just a single conv.
        """
        super().__init__()
        self.__name__ = 'NConvUNet'

        encoder_filter_sz = _pair(encoder_filter_sz)
        decoder_filter_sz = _pair(decoder_filter_sz)
        out_filter_sz = _pair(out_filter_sz)

        self.num_downsampling = num_downsampling
        self.data_pooling = data_pooling
        self.shared_encoder = shared_encoder
        self.use_double_conf = use_double_conv

        # Map from in_channels to in_channels*channels_multiplier
        self.nconv_in = NConv2d(in_channels=in_ch, out_channels=in_ch*channels_multiplier, kernel_size=encoder_filter_sz,
                                stride=(1, 1), pos_fn=pos_fn, groups=groups, bias=use_bias)

        # Two sequential nconv for intermediate layers with  in_channels*channels_multiplier channels
        if self.use_double_conf:
            self.nconv_x2 = nn.Sequential(
                NConv2d(in_channels=in_ch*channels_multiplier, out_channels=in_ch*channels_multiplier,
                        kernel_size=encoder_filter_sz, stride=(1,1), pos_fn=pos_fn, groups=groups, bias=use_bias),
                NConv2d(in_channels=in_ch*channels_multiplier, out_channels=in_ch*channels_multiplier,
                        kernel_size=encoder_filter_sz, stride=(1, 1), pos_fn=pos_fn, groups=groups, bias=use_bias)
            )
        else:
            self.nconv_x2 = nn.Sequential(
                NConv2d(in_channels=in_ch * channels_multiplier, out_channels=in_ch * channels_multiplier,
                        kernel_size=encoder_filter_sz,
                        stride=(1, 1), pos_fn=pos_fn, groups=groups, bias=use_bias),
            )

        self.encoder = nn.ModuleList([nn.Sequential(self.nconv_in, self.nconv_x2)])
        for i in range(self.num_downsampling):
            if self.shared_encoder:
                self.encoder.append(self.nconv_x2[0])  # Use only the first NConv as the sparsity decreased after downsampling
            else:  # Define a new layer
                self.encoder.append(NConv2d(in_channels=in_ch*channels_multiplier, out_channels=in_ch*channels_multiplier,
                                            kernel_size=encoder_filter_sz, stride=(1, 1), pos_fn=pos_fn, groups=groups,
                                            bias=use_bias ))

        self.decoder = nn.ModuleList([NConv2d(in_channels=2*in_ch*channels_multiplier, out_channels=in_ch*channels_multiplier,
                                              kernel_size=decoder_filter_sz, stride=(1, 1), pos_fn=pos_fn, groups=groups,
                                              bias=use_bias)
                                      for _ in range(self.num_downsampling)])

        # Map from in_channels*channels_multiplier to in_ch
        self.nconv_out = NConv2d(in_channels=in_ch*channels_multiplier, out_channels=in_ch, kernel_size=out_filter_sz,
                                 stride=(1, 1), pos_fn=pos_fn, groups=groups, bias=False)

    @staticmethod
    def downsample_data_conf(data, conf, ds_factor, pooling_type):
        conf_ds, idx = F.max_pool2d(conf, ds_factor, ds_factor, return_indices=True)
        conf_ds /= 4  # Scale the confidence the determinant of the jackobian of the transformation
        if pooling_type == 'conf_based':
            data_ds = retrieve_elements_from_indices(data, idx)
        elif pooling_type == 'max_pooling':
            data_ds = F.max_pool2d(data, ds_factor, ds_factor)
        else:
            raise NotImplementedError('Choose `self.data_pooling` from [conf_based, max_pooling]!')
        return data_ds, conf_ds

    def forward(self, inpt):
        # Initialize a list for intermediate features
        x = [None] * (self.num_downsampling * 2 + 1)
        c = [None] * (self.num_downsampling * 2 + 1)

        # Add the input to the list
        x[0] = inpt[0]
        c[0] = inpt[1]

        # The encoder
        nds = self.num_downsampling
        if nds == 0:
            x[0], c[0] = self.encoder[0]((x[0], c[0]))
        else:
            for i in range(nds+1):
                if i == 0:  # No downsampling for first scale
                    x[i + 1], c[i + 1] = self.encoder[i]((x[i], c[i]))
                else:
                    data_ds, conf_ds = self.downsample_data_conf(data=x[i], conf=c[i], ds_factor=2, pooling_type=self.data_pooling)
                    x[i + 1], c[i + 1] = self.encoder[i]((data_ds, conf_ds))

            # The decoder
            for i in range(nds):
                x_up = F.interpolate(x[i+nds], size=c[nds-i].shape[2:], mode='nearest')
                c_up = F.interpolate(c[i+nds], size=c[nds-i].shape[2:], mode='nearest')
                x[i+nds+1], c[i+nds+1] = self.decoder[i]((torch.cat((x_up, x[nds-i]), 1), torch.cat((c_up, c[nds-i]), 1)))

        # Map back to in_channels
        xout, cout = self.nconv_out((x[-1], c[-1]))

        return xout, cout


##### Normalized Convolution Layer
class NConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=None, dilation=(1, 1), groups=1,
                 bias=False, pos_fn='softplus', prop_conf=True, init_method='n'):
        if padding is None:
            padding = _pair(kernel_size[0] // 2)
        super(NConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, transposed=False,
                                      output_padding=_pair(0), groups=groups, bias=bias, padding_mode='zeros', )
        self.eps = 1e-20
        self.pos_fn = pos_fn
        self.init_method = init_method
        self.prop_conf = prop_conf

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize weights and bias
        self.init_parameters()

        if self.pos_fn is not None:
            EnforcePos.apply(self, 'weight', pos_fn)

    def forward(self, inpt):
        data = inpt[0]
        conf = inpt[1]

        # Normalized Convolution
        denom = F.conv2d(conf, self.weight, None, self.stride,
                         self.padding, self.dilation, self.groups)
        nomin = F.conv2d(data * conf, self.weight, None, self.stride,
                         self.padding, self.dilation, self.groups)
        nconv = nomin / (denom + self.eps)

        # Add bias
        if self.bias is not None:
            b = self.bias.view(1, len(self.bias), 1, 1)
            b = b.expand_as(nconv)
            nconv += b
        if self.prop_conf:
            # Propagate confidence
            cout = denom
            sz = cout.size()
            cout = cout.view(sz[0], sz[1], -1)

            k = self.weight
            k_sz = k.size()
            k = k.view(k_sz[0], -1)
            s = torch.sum(k, dim=-1, keepdim=True)
            #k = k.view(k_sz[0], k_sz[1], -1)
            #s = torch.sum(k, dim=-1, keepdim=True).squeeze(-1)

            cout = cout / s
            cout = cout.view(sz)

        else:
            cout = None

        return nconv, cout

    def init_parameters(self):
        # Init weights
        if self.init_method == 'x':  # Xavier
            torch.nn.init.xavier_uniform_(self.weight)
        elif self.init_method == 'k':  # Kaiming
            torch.nn.init.kaiming_uniform_(self.weight)
        elif self.init_method == 'n':  # Normal dist
            n = self.kernel_size[0] * self.kernel_size[1] * self.out_channels
            self.weight.data.normal_(2, math.sqrt(2. / n))

        # Init bias
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

#### Non-negativity enforcement function for the applicability function
class EnforcePos(object):
    def __init__(self, name, pos_fn):
        self.name = name
        self.pos_fn = pos_fn

    def compute_weight(self, module):
        return _pos(getattr(module, self.name + '_p'), self.pos_fn)

    @staticmethod
    def apply(module, name, pos_fn):
        fn = EnforcePos(name, pos_fn)

        weight = getattr(module, name)

        # remove w from parameter list
        del module._parameters[name]

        #
        module.register_parameter(name + '_p', Parameter(_pos(weight, pos_fn).data))
        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        return fn

    def remove(self, module):
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + '_p']
        module.register_parameter(self.name, Parameter(weight.data))

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module))


def _pos(p, pos_fn):
    pos_fn = pos_fn.lower()
    if pos_fn == 'softmax':
        p_sz = p.size()
        p = p.view(p_sz[0], p_sz[1], -1)
        p = F.softmax(p, -1)
        return p.view(p_sz)
    elif pos_fn == 'exp':
        return torch.exp(p)
    elif pos_fn == 'softplus':
        return F.softplus(p, beta=10)
    elif pos_fn == 'sigmoid':
        return F.sigmoid(p)
    else:
        print('Undefined positive function!')
        return


def remove_weight_pos(module, name='weight'):
    r"""Removes the weight normalization reparameterization from a module.

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter

    Example:
        >>> m = weight_norm(nn.Linear(20, 40))
        >>> remove_weight_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, EnforcePos) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("weight_norm of '{}' not found in {}"
                     .format(name, module))


if __name__ == '__main__':

    net = NConvUNet(1, channels_multiplier=2, num_downsampling=1, encoder_filter_sz=5, use_bias=True,
                    decoder_filter_sz=3, pos_fn='SoftPlus', groups=1, data_pooling='max_pooling',
                    shared_encoder=True, use_double_conv=False)

    x0 = torch.rand((1,1,100,100))
    c0 = torch.rand((1, 1, 100, 100))

    xout, cout = net((x0, c0))

    print(xout.shape)
    print(cout.shape)


