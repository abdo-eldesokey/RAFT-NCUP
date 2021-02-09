import torch
import torch.nn as nn
import torch.nn.functional as F

from interp_weights_est import Simple, UNet
from nconv_modules import NConvUNet, NConv2d
from pac_upsampler import PacJointUpsample, DJIF


def get_upsampler(in_ch, guidance_ch, args):

    upsampler_name = 'nconvupsampler' #args.final_upsampling.lower()

    if upsampler_name == 'nconvupsampler':
        # Define the normalized convolution interpolation network
        #TODO: Normalized convolution network only supports in_ch of 1, add support for more channels using groups.
        interpolation_net = NConvUNet(in_ch=1, channels_multiplier=args.interp_net_channels_multiplier,
                                      num_downsampling=args.interp_net_num_downsampling,
                                      encoder_filter_sz=args.interp_net_encoder_filter_sz,
                                      decoder_filter_sz=args.interp_net_decoder_filter_sz,
                                      out_filter_sz=args.interp_net_out_filter_sz,
                                      use_bias=args.interp_net_use_bias,
                                      data_pooling=args.interp_net_data_pooling,
                                      shared_encoder=args.interp_net_shared_encoder,
                                      use_double_conv=args.interp_net_use_double_conv,
                                      pos_fn='SoftPlus', groups=1)


        # Define the weights estimation network
        weights_est_net = None

        # Add the number of input channels at the beginning of num_ch
        #num_channels = [guidance_ch, 64, 32] #args.weights_est_net_num_ch.copy()
        num_channels = args.weights_est_net_num_ch.copy()

        if args.final_upsampling_use_data_for_guidance:
            num_channels.insert(0, guidance_ch + in_ch)
        else:
            num_channels.insert(0, guidance_ch)

        # Disable Batch Normalization for KITTI
        use_bn = args.dataset == 'sintel'
        if args.weights_est_net.lower() == 'simple':
            weights_est_net = Simple(num_ch=num_channels, out_ch=in_ch, use_bn=use_bn,
                                     filter_sz=args.weights_est_net_filter_sz, dilation=args.weights_est_net_dilation,
                                     final_act=torch.sigmoid)

        elif args.weights_est_net.lower() == 'unet':
            weights_est_net = UNet(num_ch=num_channels, out_ch=in_ch, final_act=torch.sigmoid)

        # Define the NConv upsampler
        upsampler = NConvUpsampler(scale=args.final_upsampling_scale, interpolation_net=interpolation_net,
                                   weights_est_net=weights_est_net,
                                   use_data_for_guidance=args.final_upsampling_use_data_for_guidance,
                                   channels_to_batch=args.final_upsampling_channels_to_batch,
                                   use_residuals=args.final_upsampling_use_residuals,
                                   est_on_high_res=args.final_upsampling_est_on_high_res)


    elif upsampler_name == 'bilinear':
        upsampler = Bilinear(args.final_upsampling_scale)

    elif upsampler_name == 'pacjointupsamplefull':
        upsampler = PacJointUpsampleFull(scale=args.final_upsampling_scale, in_ch=1, guidance_ch=guidance_ch)

    elif upsampler_name == 'djiforiginal':
        upsampler = DjifOriginal(scale=args.final_upsampling_scale, in_ch=1, guidance_ch=guidance_ch)

    else:
        raise NotImplementedError('Upsampler `{}` is not implemented!'.format(upsampler_name))

    return upsampler


class NConvUpsampler(torch.nn.Module):
    def __init__(self, scale=None, size=None, interpolation_net=None, weights_est_net=None, use_data_for_guidance=True,
                 channels_to_batch=True, use_residuals=False, est_on_high_res=False):
        """
        An upsampling layer using Normalized CNN with an input weights estimation network.
        Either `scale` or `size` needs to be specified.

        Args:
            scale: The uspampling factor.
            size: The desired size of the output.
            interpolation_net: Interpolation network. Needs to be an object of `NConvUNetFull`.
            weights_est_net: Weights estimation network.
            use_data_for_guidance: Either to use the low-resolution data as input to the weights estimation network with
                                   the guidance data.
            channels_to_batch: Either to reshape data tensor to B*Cx1xHxW before performing interpolation.
        """
        super().__init__()
        self.__name__ = 'NConvUpsampler'
        # Check the validity of arguments
        if scale is None and size is None:
            raise ValueError('Either scale or size needs to be set!')
        elif scale is not None and size is not None:
            raise ValueError('You can set either scale or size at a time!')
        elif scale is not None and size is None:
            if isinstance(scale, tuple):
                self.scaleW = float(scale[1])
                self.scaleH = float(scale[0])
            elif isinstance(scale, int):
                self.scaleW = self.scaleH = float(scale)
            else:
                raise ValueError('Scale value can be tuple or integer only!')
        elif scale is None and size is not None:
            if isinstance(size, tuple):
                self.osize = size
                self.scaleW = self.scaleH = None
            else:
                raise ValueError('Size has to be a tuple!')

        # Interpolation network must be provided and from `NConv` family
        if interpolation_net is None:
            raise ValueError('An interpolation network mush be provided!')
        else:
            assert 'NConv' in interpolation_net.__name__, 'Only `NConv` interpolaion networks are supported!'
            self.interpolation_net = interpolation_net
            # Get the number of data input channels
            self.data_ich = self.interpolation_net._modules['nconv_in'].in_channels

        if weights_est_net is None:  # No weights estimation network provided, use binary weights mask
            self.weights_est_net = self.get_binary_weights
            self.guidance_ich = self.data_ich
        else:
            self.weights_est_net = weights_est_net
            # Get the number of guidance input channels
            self.guidance_ich = self.weights_est_net.in_ch

        self.use_data_for_guidance = use_data_for_guidance
        self.channels_to_batch = channels_to_batch
        self.use_residuals = use_residuals
        self.est_on_high_res = est_on_high_res

        # If data is used for guidance, check that the guidance network has the right number of in_channels
        if self.use_data_for_guidance:
            assert(self.guidance_ich >= self.data_ich)

    @staticmethod
    def get_binary_weights(t):
        return (t > 0).float()

    def forward(self, x_lowres, x_guidance=None):
        x_highres = self.get_out_tensor(x_lowres)

        # Prepare guidance data
        if self.est_on_high_res:
            x_data_for_guidance = x_highres
        else:
            x_guidance = F.interpolate(x_guidance, x_lowres.size()[2:], mode='area')  # Downsample the guidance
            x_data_for_guidance = x_lowres

        # Feed guidance data to weights estimation network
        if self.use_data_for_guidance:
            w_lowres = self.weights_est_net(torch.cat((x_data_for_guidance, x_guidance), 1))
        else:
            w_lowres = self.weights_est_net(x_guidance)

        if self.est_on_high_res:
            w_highres = w_lowres
        else:
            w_highres = self.get_out_tensor(w_lowres)

        ib, ic, oh, ow = x_highres.shape

        # Perform interpolation using NConv
        if self.channels_to_batch:
            output, _ = self.interpolation_net((x_highres.view(ib * ic, 1, oh, ow), w_highres.view(ib * ic, 1, oh, ow)))
        else:
            output, _ = self.interpolation_net((x_highres, w_highres))

        output = output.view(ib, ic, oh, ow)

        if self.use_residuals:
            output[x_highres > 0] = x_highres[x_highres > 0]

        return output

    def get_out_tensor(self, inp):
        b, ic, ih, iw = inp.shape

        if self.scaleH is None and self.scaleW is None:  # Size was provided
            oh = self.osize[0]
            ow = self.osize[1]

            # Calculate the scaling factor
            sH = oh / ih
            sW = ow / iw
        else:
            sH = int(self.scaleH)
            sW = int(self.scaleW)
            oh = round(ih * sH)
            ow = round(iw * sW)

        out_t = torch.zeros((b, ic, oh, ow), dtype=inp.dtype).to(inp.device)
        """
        ix = torch.arange(iw).to(inp.device).float()
        iy = torch.arange(ih).to(inp.device).float()

        ox = torch.round(ix * sW).long()
        oy = torch.round(iy * sH).long()

        gy, gx = torch.meshgrid([oy, ox])
        
        out_t[:, :, gy+4, gx+4] = inp
        """

        out_t[:, :, sH//2::sH, sW//2::sW] = inp

        return out_t


class Bilinear(nn.Module):
    def __init__(self, scale=None):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)

    def forward(self, x, *argv):
        return self.up(x)


class PacJointUpsampleFull(nn.Module):
    def __init__(self, scale=None, in_ch=1, guidance_ch=3):
        super().__init__()

        self.up = PacJointUpsample(factor=scale, channels=in_ch, guide_channels=guidance_ch)
        #print(tools.get_num_params(self))

    def forward(self, x, guide):
        return self.up(x, guide)


class DjifOriginal(nn.Module):
    def __init__(self, scale=None, in_ch=1, guidance_ch=3):
        super().__init__()

        self.up = DJIF(factor=scale, channels=in_ch, guide_channels=guidance_ch)
        #print(tools.get_num_params(self))

    def forward(self, x, guide):
        return self.up(x, guide)


if __name__ == '__main__':
    upsampler = get_upsampler(3, 3, None).cuda()
    lr = torch.rand((10,3,50,50)).cuda()
    hr = upsampler(lr, lr)

