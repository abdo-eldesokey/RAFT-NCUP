from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from raft import RAFT
from raft_nc import RAFT as RAFT_NC
from raft_nc_sep import RAFT as RAFT_NC_SEP
from raft_nc_dbl import RAFT as RAFT_NC_DBL
import evaluate
import datasets

from torch.utils.tensorboard import SummaryWriter

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100

def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics

def show_image(img):
    img = img.permute(1,2,0).cpu().numpy()
    plt.imshow(img/255.0)
    plt.show()
    # cv2.imshow('image', img/255.0)
    # cv2.waitKey()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    if args.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    elif args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    else:
        raise NotImplementedError('{} optimizer is not implemented!'.format(args.optimizer))

    if args.scheduler.lower() == 'cyclic':
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
                                                  pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    elif args.scheduler.lower() == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=0.5)
    else:
        raise NotImplementedError('{} scheduler is not implemented!'.format(args.scheduler))
    return optimizer, scheduler
    

class Logger:
    def __init__(self, scheduler, args):
        self.scheduler = scheduler
        self.args = vars(args)
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

        if not os.path.isdir('checkpoints/'+args.name):
            os.mkdir('checkpoints/'+args.name)

        self.txt_file = open('checkpoints/'+args.name+'/log.txt', 'a')
        self.print_args()

    def print_args(self):
        print('\n### Experiments Arguments ###')
        self.txt_file.writelines('\n### Experiments Arguments ### \n')
        for k,v in self.args.items():
            print('{}: {}'.format(k, v))
            self.txt_file.writelines('{}: {}\n'.format(k, v))
        self.txt_file.flush()

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)
        self.txt_file.writelines(training_str + metrics_str + '\n')
        self.txt_file.flush()

        if self.writer is None:
            self.writer = SummaryWriter(log_dir='checkpoints/%s' % args.name)

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)
            if key in args.validation + ['clean', 'final', 'kitti-epe', 'kitti-f1']:
                self.txt_file.writelines('Validation %s EPE: %f\n' % (key, results[key]))

    def close(self):
        self.writer.close()


def train(args):
    if args.model == 'raft':
        model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    elif args.model == 'raft_nc':
        model = nn.DataParallel(RAFT_NC(args), device_ids=args.gpus)
    elif args.model == 'raft_nc_sep':
        model = nn.DataParallel(RAFT_NC_SEP(args), device_ids=args.gpus)
    elif args.model == 'raft_nc_dbl':
        model = nn.DataParallel(RAFT_NC_DBL(args), device_ids=args.gpus)
    else:
        raise NotImplementedError('Model %s not found!' % args.model)

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model.cuda()
    model.train()

    if args.stage != 'chairs':
        model.module.freeze_bn()

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(scheduler, args)

    num_params = count_parameters(model)
    print("Parameter Count: %d" % num_params)
    logger.txt_file.writelines("Parameter Count: %d\n" % num_params)

    VAL_FREQ = 5000
    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda() for x in data_blob]

            # show_image(image1[0])
            # show_image(image2[0])

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).to(image1.device)).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).to(image2.device)).clamp(0.0, 255.0)

            flow_predictions = model(image1, image2, iters=args.iters)            

            loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma)
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = 'checkpoints/%s/%d_%s.pth' % (args.name, total_steps+1, args.name)
                torch.save(model.state_dict(), PATH)

                results = {}
                for val_dataset in args.validation:
                    if val_dataset == 'chairs':
                        if args.model == 'raft_nc_sep':
                            results.update(evaluate.validate_chairs(model.module, 12))
                        else:
                            results.update(evaluate.validate_chairs(model.module))
                    elif val_dataset == 'sintel':
                        results.update(evaluate.validate_sintel(model.module))
                    elif val_dataset == 'kitti':
                        results.update(evaluate.validate_kitti(model.module))

                logger.write_dict(results)
                
                model.train()
                if args.stage != 'chairs':
                    model.module.freeze_bn()
            
            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = 'checkpoints/%s/final_model.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--model', default='raft', help="model to train")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')


    # By Abdo
    parser.add_argument('--optimizer', default='adamw')
    parser.add_argument('--scheduler', default='cyclic')
    parser.add_argument('--scheduler_step', type=int, default=20000)
    parser.add_argument('--upsampler_bi', action='store_true', help='use bilinear upsampling')
    parser.add_argument('--align_corners', action='store_true', help='align_corners for bilinear upsampling')
    parser.add_argument('--freeze_raft', action='store_true', help='freeze the optical flow network and train only nc')
    parser.add_argument('--load_pretrained', default=None, help='freeze the optical flow network and train only nc')
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

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)