import sys
sys.path.append('core')

import argparse
import os
import numpy as np
import torch
import cv2

import datasets
from utils import frame_utils, flow_viz

from raft import RAFT
from utils.utils import InputPadder, forward_interpolate

import torch.nn as nn
from raft_nc import RAFT as RAFT_NC
from raft_nc_sep import RAFT as RAFT_NC_SEP
from raft_nc_dbl import RAFT as RAFT_NC_DBL


@torch.no_grad()
def create_sintel_submission(model, iters=32, warm_start=False, output_path='sintel_submission', write_png=False):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)

        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame + 1))
            if write_png:
                output_dir_png = os.path.join(output_path+'_png', dstype, sequence)
                output_file_png = os.path.join(output_dir_png, 'frame%04d.png' % (frame + 1))
                if not os.path.exists(output_dir_png):
                    os.makedirs(output_dir_png)
                cv2.imwrite(output_file_png, flow_viz.flow_to_image(flow))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission', write_png=False):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if write_png:
        out_path_png = output_path+'_png'
        if not os.path.exists(out_path_png):
            os.makedirs(out_path_png)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id,) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        if write_png:
            output_filename_png = os.path.join(out_path_png, frame_id+'.png')
            cv2.imwrite(output_filename_png, flow_viz.flow_to_image(flow))

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_chairs(model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt) ** 2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}


@torch.no_grad()
def validate_sintel(model, iters=32):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all < 1)
        px3 = np.mean(epe_all < 3)
        px5 = np.mean(epe_all < 5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_kitti(model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="model name")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--upsampler_bi', action='store_true', help='use bilinear upsampling')
    parser.add_argument('--align_corners', action='store_true', help='align_corners for bilinear upsampling')
    parser.add_argument('--load_pretrained', default=None, help='freeze the optical flow network and train only nc')
    parser.add_argument('--freeze_raft', action='store_true', help='freeze the optical flow network and train only nc')
    parser.add_argument('--compressed_ft', action='store_true', help='load the compressed version of FlyingThings3D')

    from utils.args import _add_arguments_for_module, str2bool, str2intlist
    import upsampler
    _add_arguments_for_module(
        parser,
        upsampler,
        name="final_upsampling",
        default_class=None,
        exclude_classes=["_*"],
        exclude_params=["self", "args", "interpolation_net", "weights_est_net", "size"],
        forced_default_types={"scale": int,
                              "use_data_for_guidance": str2bool,
                              "channels_to_batch": str2bool,
                              "use_residuals": str2bool,
                              "est_on_high_res": str2bool},
    )

    import nconv_modules
    _add_arguments_for_module(
        parser,
        nconv_modules,
        name="interp_net",
        default_class=None,
        exclude_classes=["_*"],
        exclude_params=["self", "args"],
        forced_default_types={"encoder_fiter_sz": int,
                               "decoder_fiter_sz": int,
                               "out_filter_size": int,
                               "use_double_conv": str2bool,
                               "use_bias": str2bool}
    )

    import interp_weights_est
    _add_arguments_for_module(
        parser,
        interp_weights_est,
        name="weights_est_net",
        default_class=None,
        exclude_classes=["_*"],
        exclude_params=["self", "args", "out_ch", "final_act"],
        unknown_default_types={"num_ch": str2intlist,
                               "filter_sz": str2intlist},
        forced_default_types={"dilation": str2intlist,
                              }
    )

    args = parser.parse_args()

    if args.model == 'raft':
        model = nn.DataParallel(RAFT(args))
    elif args.model == 'raft_nc':
        model = nn.DataParallel(RAFT_NC(args))
    elif args.model == 'raft_nc_sep':
        model = nn.DataParallel(RAFT_NC_SEP(args))
    elif args.model == 'raft_nc_dbl':
        model = nn.DataParallel(RAFT_NC_DBL(args))
    else:
        raise NotImplementedError('Model %s not found!' % args.model)

    #model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.restore_ckpt))

    model.cuda()
    model.eval()

    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model.module)

        elif args.dataset == 'sintel':
            #create_sintel_submission(model.module, warm_start=True, write_png=True) ; exit()
            validate_sintel(model.module)

        elif args.dataset == 'kitti':
            #create_kitti_submission(model.module, write_png=True)
            validate_kitti(model.module)



