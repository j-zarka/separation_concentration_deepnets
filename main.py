import argparse
import json
import os
import random
import shutil
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from kymatio.torch import Scattering2D
from phase_scattering2d_torch import ScatteringTorch2D_wph
from models.Analysis import Analysis
from models.LinearProj import LinearProj
from models.Classifier import Classifier
from models.ScatNetAnalysis import ScatNetAnalysis


parser = argparse.ArgumentParser()
parser.add_argument('--data', help='path to dataset')
parser.add_argument('--dataset', default='imagenet', choices=['mnist', 'cifar10', 'cifar100', 'imagenet'],
                    help='dataset to train on (default: imagenet)')
parser.add_argument('-a', '--arch', default='scatnetblockanalysis',
                    choices=['analysis', 'scatnet', 'scatnetblock', 'scatnetblockanalysis'],
                    help='model architecture (default: scatnetblockanalysis)')
parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=150, type=int, help='number of total epochs to run (default: 150)')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help='initial learning rate (default: 0.01)', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int, help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
parser.add_argument('--learning-rate-adjust-frequency', default=30, type=int,
                    help='number of epoch after which learning rate is decayed by 10 (default: 30)')
parser.add_argument('--dir', default='default_dir', type=str, help='directory for training logs and checkpoints')
parser.add_argument('--pars-beta', default=0.0005, type=float, help='learning rate for pars reg (default: 0.0005)')
parser.add_argument('--n-blocks', default=4, type=int, help='number of blocks in the pipeline (default: 4)')

# Scattering parameters
parser.add_argument('--scat-angles', default=8, type=int, nargs='+', help='number of orientations for wavelet frame(s) (default: 8)')
parser.add_argument('--backend', default='torch', type=str, help='scattering backend')
parser.add_argument('--scattering-J', default=4, type=int,
                    help='maximum scale for the scattering transform - for scatnet arch only (default: 4)')

# Linear projection parameters
parser.add_argument('--P-proj-size', default=256, type=int, nargs='+',
                    help='output dimension of the linear projection(s) P(s) (default: 256)')
parser.add_argument('--P-kernel-size', default=1, type=int, nargs='+', help='kernel size of P(s) (default: 1)')

# Analysis parameters
parser.add_argument('--non-linearity', default='relu', type=str, choices=['absolute', 'softshrink', 'relu'],
                    help='non linearity for analysis (default: relu)')
parser.add_argument('--zero-bias', action='store_true', help='force zero bias for ReLU')
parser.add_argument('--frame-width', default=2048, type=int, nargs='+', help='size(s) of tight frame(s) (default: 2048)')
parser.add_argument('--frame-kernel-size', default=1, type=int, nargs='+', help='kernel size of frame(s) (default: 1)')
parser.add_argument('--frame-stride', default=1, type=int, nargs='+', help='stride of frame(s) (default: 1)')

# Classifier parameters
parser.add_argument('--classifier-type', default='fc', type=str, choices=['fc', 'mlp'], help='classifier type (default: fc)')
parser.add_argument('--avg-ker-size', default=1, type=int, help='size of averaging kernel (default: 1)')
parser.add_argument('--nb-hidden-units', default=2048, type=int, help='number of hidden units for mlp classifier '
                                                                      '(default: 2048)')
parser.add_argument('--dropout-p-mlp', default=0.3, type=float, help='dropout probability in mlp (default: 0.3)')
parser.add_argument('--nb-l-mlp', default=2, type=int, help='number of hidden layers in mlp (default: 2)')

# Train on a subsample of classes
parser.add_argument('--nb-classes', default=1000, type=int, help='ImageNet only - number of classes randomly chosen '
                    'used for training and validation (default: 1000 = whole train/val dataset)')
parser.add_argument('--class-indices', default=None,
                    help='ImageNet only - numpy array of indices used in case nb-classes < 1000')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # fill missing arguments
    num_args = ["scat_angles", "frame_kernel_size", "frame_width", "frame_stride", "P_proj_size", "P_kernel_size"]
    if args.arch in ['scatnet', 'analysis']: #override default in case not specified and ensure a single block is used
        setattr(args, "n_blocks", 1)

    n_blocks = args.n_blocks

    for item in num_args:
        value = getattr(args, item)
        if type(value) == int or type(value) == float:  # value repeated
            setattr(args, item, [value] * n_blocks)
        elif len(value) < n_blocks:  # default value added
            setattr(args, item, value + [parser.get_default(item)] * (n_blocks - len(value)))

    main_worker(args)


def load_model(args, logfile, summaryfile):
    n_blocks = args.n_blocks

    # Create building blocks
    scattering = [nn.Identity()] * n_blocks

    standardization = [nn.Identity()] * n_blocks
    proj = [nn.Identity()] * n_blocks
    linear_proj = [nn.Identity()] * n_blocks

    analysis = [nn.Identity()] * n_blocks

    model = [nn.Identity()] * n_blocks

    # Create model log
    ###########################################################################################

    if args.arch == 'scatnetblockanalysis':
        arch_log = "=> creating model ScatNetBlockAnalysis with {} blocks, scat angles {}, linear projection " \
                   "dimensions {}, frame widths {}, non linearity {}, frame kernel sizes {}, frame strides {} " \
                   "pars beta {}, classifier {} pipeline".\
                   format(n_blocks, args.scat_angles, args.P_proj_size, args.frame_width,  args.non_linearity,
                          args.frame_kernel_size, args.frame_stride, args.pars_beta, args.classifier_type)

    elif args.arch == 'scatnetblock':
        arch_log = "=> creating model ScatNetBlock with {} blocks, scat angles {}, linear projection dimensions {}, " \
                   "pars beta {}, classifier {} pipeline". \
                   format(n_blocks, args.scat_angles, args.P_proj_size, args.pars_beta, args.classifier_type)

    elif args.arch == 'scatnet':
        arch_log = "=> creating model ScatNet with scattering J {}, scat angles {}, " \
                   "linear projection dimension {}, pars beta {}, classifier {} pipeline". \
                   format(args.scattering_J, args.scat_angles[0], args.P_proj_size[0],
                          args.pars_beta, args.classifier_type)

    elif args.arch == 'analysis':
        arch_log = "=> creating model Analysis with linear projection dimension {}, " \
                    "frame width {}, non linearity {}, frame kernel size {}, frame stride {} pars beta {}, " \
                    "classifier {} pipeline". \
                    format(args.P_proj_size[0], args.frame_width[0], args.non_linearity,
                           args.frame_kernel_size[0], args.frame_stride[0], args.pars_beta, args.classifier_type)

    # Model creation
    ###########################################################################################

    if args.dataset == 'cifar10':
        nb_channels_in = 3
        n_space = 32
        nb_classes = 10

    elif args.dataset == 'cifar100':
        nb_channels_in = 3
        n_space = 32
        nb_classes = 100

    elif args.dataset == 'mnist':
        nb_channels_in = 1
        n_space = 28
        nb_classes = 10

    elif args.dataset == 'imagenet':
        nb_channels_in = 3
        n_space = 224
        nb_classes = 1000

    else:
        assert False

    nb_params, nb_proj_params, nb_frame_params, nb_classifier_params = compute_model_size(
        args, n_space, nb_channels_in, nb_classes)

    print_and_write('Total number of params:{:.2f}M, proj:{:.2f}M, frame:{:.2f}M, classifier:{:.2f}M'.
                    format(nb_params / 1e6, nb_proj_params / 1e6, nb_frame_params / 1e6,
                           nb_classifier_params / 1e6), logfile, summaryfile)

    for i in range(n_blocks):
        if args.arch in ['scatnet', 'scatnetblock', 'scatnetblockanalysis']:
            # create scattering
            if args.arch == 'scatnet':
                J = args.scattering_J
            else:
                J = 1  # one block per scale
            L_ang = args.scat_angles[i]

            if args.arch == 'scatnet':
                # use traditional scattering transform from kymatio
                scattering[i] = Scattering2D(J=J, shape=(n_space, n_space), L=L_ang, max_order=2,
                                             backend=args.backend)
            else:
                scattering[i] = ScatteringTorch2D_wph(J=J, shape=(n_space, n_space), L=L_ang, backend=args.backend)

            # Flatten scattering
            scattering[i] = nn.Sequential(scattering[i], nn.Flatten(1, 2))

            if args.arch == 'scatnet':
                factor_channels = 1 + L_ang * J + (L_ang ** 2) * J * (J - 1) // 2  # scattering of order 2
            else:
                factor_channels = 1 + 4 * L_ang  # 4 phases used
            nb_channels_in *= factor_channels

            n_space = n_space // (2 ** J)

        ###########################################################################################
        # create linear proj

        if args.arch in ['scatnet', 'scatnetblock', 'scatnetblockanalysis']:
            standardization[i] = nn.BatchNorm2d(nb_channels_in, affine=False)

            if args.P_proj_size[i] > nb_channels_in:
                raise ValueError('Proj dimension must be lower than the one of input space')

            proj[i] = nn.Conv2d(nb_channels_in, args.P_proj_size[i], kernel_size=args.P_kernel_size[i],
                                stride=1, padding=0, bias=False)
            nn.init.orthogonal_(proj[i].weight.data)

            nb_channels_in = args.P_proj_size[i]

            linear_proj[i] = LinearProj(standardization[i], proj[i], args.P_kernel_size[i],
                                        args.frame_kernel_size[i], args.frame_stride[i])
        else:
            # no need to standardize on two-layer network since input is already standardized, but normalize
            linear_proj[i] = LinearProj(nn.Identity(), nn.Identity(), args.P_kernel_size[i],
                                        args.frame_kernel_size[i], args.frame_stride[i])

        ###########################################################################################
        # Create tight frame analysis block

        if args.arch in ['scatnetblockanalysis', 'analysis']:
            if args.non_linearity == 'relu' and not args.zero_bias:
                lambda_ = 1.
            elif args.non_linearity == 'softshrink':
                lambda_ = 1.5
            else:
                lambda_ = 0  # ReLU with zero bias or absolute value (ignores lambda anyway)

            analysis[i] = Analysis(nb_channels_in, frame_width=args.frame_width[i],
                                   lambda_=lambda_/np.sqrt(args.frame_width[i]),
                                   frame_kernel_size=args.frame_kernel_size[i], frame_stride=args.frame_stride[i],
                                   non_linearity=args.non_linearity, n_space=n_space)

    ###########################################################################################
    # Create classifier
    ###########################################################################################

    classifier = Classifier(n_space, nb_channels_in, classifier_type=args.classifier_type,
                            nb_classes=nb_classes, nb_hidden_units=args.nb_hidden_units,
                            nb_l_mlp=args.nb_l_mlp, dropout_p_mlp=args.dropout_p_mlp,
                            avg_ker_size=args.avg_ker_size)

    # Create model
    ###########################################################################################
    for i in range(n_blocks):
        # scattering[i] and/or analysis[[i] can be nn.Identity()
        model[i] = ScatNetAnalysis(scattering[i], linear_proj[i], analysis[i],
                                   nn.Identity() if i < n_blocks - 1 else classifier)

    print_and_write(arch_log, logfile, summaryfile)
    print_and_write('Number of epochs {}, learning rate decay epochs {}, learning rate {}'.
                    format(args.epochs, args.learning_rate_adjust_frequency, args.lr), logfile, summaryfile)

    model = nn.Sequential(*model)
    return model


def main_worker(args):
    best_acc1 = 0
    best_acc5 = 0
    best_epoch_acc1 = 0
    best_epoch_acc5 = 0

    checkpoint_savedir = os.path.join('./checkpoints', args.dir)
    if not os.path.exists(checkpoint_savedir):
        os.makedirs(checkpoint_savedir)
    checkpoint_savefile = os.path.join(checkpoint_savedir, '{}_batchsize_{}_lrfreq_{}.pth.tar'.format(
        args.arch, args.batch_size, args.learning_rate_adjust_frequency))
    best_checkpoint_savefile = os.path.join(checkpoint_savedir,
                                            '{}_batchsize_{}_lrfreq_{}_best.pth.tar'.format(
                                                args.arch, args.batch_size, args.learning_rate_adjust_frequency))

    logs_dir = os.path.join('./training_logs', args.dir)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    logfile = open(os.path.join(logs_dir, 'training_{}_b_{}_lrfreq_{}.log'.format(
        args.arch, args.batch_size, args.learning_rate_adjust_frequency)), 'a')
    summaryfile = open(os.path.join(logs_dir, 'summary_file.txt'), 'a')
    writer = SummaryWriter(logs_dir)

    # Also save args.
    with open(os.path.join(checkpoint_savedir, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Data loading code
    ###########################################################################################
    # Input normalization (valid for every dataset except MNIST)

    print_and_write(f"Working on {args.dataset.upper()}", logfile, summaryfile)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = datasets.CIFAR10(root='./data', train=False, download=True,
                                       transform=transforms.Compose([transforms.ToTensor(), normalize]))

    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = datasets.CIFAR100(root='./data', train=False, download=True,
                                        transform=transforms.Compose([transforms.ToTensor(), normalize]))

    elif args.dataset == 'mnist':
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        print_and_write("Working on MNIST", logfile, summaryfile)
        train_dataset = datasets.MNIST(root='./data', train=True, download=True,
                                       transform=transforms.Compose([transforms.ToTensor(), normalize]))

        val_dataset = datasets.MNIST(root='./data', train=False, download=True,
                                     transform=transforms.Compose([transforms.ToTensor(), normalize]))

    elif args.dataset == 'imagenet':
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')

        train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

        val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

        # can use a subset of all classes for ImageNet (specified in a file or randomly chosen)
        if args.nb_classes < 1000:
            train_indices = list(np.load('utils_sampling/imagenet_train_class_indices.npy'))
            val_indices = list(np.load('utils_sampling/imagenet_val_class_indices.npy'))
            classes_names = torch.load('utils_sampling/labels_frame')
            if args.class_indices is not None:
                class_indices = torch.load(args.class_indices)
            else:
                perm = torch.randperm(1000)
                class_indices = perm[:args.nb_classes].tolist()
            train_indices_full = [x for i in range(len(class_indices))
                                  for x in range(train_indices[class_indices[i]], train_indices[class_indices[i] + 1])]
            val_indices_full = [x for i in range(len(class_indices))
                                for x in range(val_indices[class_indices[i]], val_indices[class_indices[i] + 1])]
            classes_indices_file = os.path.join(logs_dir, 'classes_indices_selected')
            selected_classes_names = [classes_names[i] for i in class_indices]
            torch.save(class_indices, classes_indices_file)
            print_and_write('Selected {} classes indices: {}'.format(args.nb_classes, class_indices), logfile,
                            summaryfile)
            print_and_write('Selected {} classes names: {}'.format(args.nb_classes, selected_classes_names), logfile,
                            summaryfile)
            if args.seed is not None:
                print_and_write('Random seed used {}'.format(args.seed), logfile, summaryfile)

            train_dataset = torch.utils.data.Subset(train_dataset, train_indices_full)
            val_dataset = torch.utils.data.Subset(val_dataset, val_indices_full)

    else:
        assert False

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)
    ###########################################################################################

    model = load_model(args,  logfile, summaryfile)
    model = torch.nn.Sequential(*[torch.nn.DataParallel(submodel) for submodel in model]).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # Do not put any weight decay for pars reg weights within the optimizer
    frame_params, all_other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".frame_weight") or name.endswith(".proj.weight"):
            frame_params.append(param)
        else:
            all_other_params.append(param)

    optimizer = torch.optim.SGD(
        [{'params': all_other_params}, {'params': frame_params, 'weight_decay': 0.}],
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_and_write("=> loading checkpoint '{}'".format(args.resume), logfile, summaryfile)
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print_and_write("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']), logfile,
                            summaryfile)
        else:
            print_and_write("=> no checkpoint found at '{}'".format(args.resume), logfile, summaryfile)

    cudnn.benchmark = True

    print_model_info(args, model, logfile, summaryfile)

    if args.evaluate:
        print_and_write("Evaluating model at epoch {}...".format(args.start_epoch), logfile)
        one_epoch(loader=val_loader, model=model, criterion=criterion, optimizer=optimizer, epoch=args.start_epoch,
                  args=args, logfile=logfile, summaryfile=summaryfile, writer=writer, is_training=False)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        one_epoch(loader=train_loader, model=model, criterion=criterion, optimizer=optimizer, epoch=epoch,
                  args=args, logfile=logfile, summaryfile=summaryfile, writer=writer, is_training=True)

        # evaluate on validation set
        acc1, acc5 = one_epoch(loader=val_loader, model=model, criterion=criterion, optimizer=optimizer, epoch=epoch,
                               args=args, logfile=logfile, summaryfile=summaryfile, writer=writer, is_training=False)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            best_epoch_acc1 = epoch
        if acc5 > best_acc5:
            best_acc5 = acc5
            best_epoch_acc5 = epoch
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint_filename=checkpoint_savefile, best_checkpoint_filename=best_checkpoint_savefile)

    print_model_info(args, model, logfile, summaryfile)
    print_and_write(
        "Best top 1 accuracy {:.2f} at epoch {}, best top 5 accuracy {:.2f} at epoch {}".
        format(best_acc1, best_epoch_acc1, best_acc5, best_epoch_acc5), logfile, summaryfile)


@torch.no_grad()
def print_model_info(args, model, logfile, summaryfile):
    n_blocks = len(model)

    for i in range(n_blocks):
        module = model[i].module

        if args.arch in ['analysis', 'scatnetblockanalysis']:
            count = 0
            for k in range(args.frame_width[i]):
                if module.analysis.frame_weight.data[k].norm(p=2) < module.analysis.frame_norm_mean - 0.01 \
                        or module.analysis.frame_weight.data[k].norm(p=2) > module.analysis.frame_norm_mean + 0.01:
                    count += 1

            if count == 0:
                print_and_write("frame {} atoms well normalized".format(i), logfile, summaryfile)
            else:
                print_and_write("{} frame {} atoms not well normalized".format(count, i), logfile,
                                summaryfile)

            frame_norms = module.analysis.frame_weight.data.norm(p=2, dim=(1, 2, 3))
            print_and_write("Max/min norms ratio {:.2f}, mean {:.2f} {} for frame {} atoms".
                            format(torch.max(frame_norms) / torch.min(frame_norms), torch.mean(frame_norms),
                                   '(norm ref {:.2f})'.format(module.analysis.frame_norm_mean),
                                   i), logfile, summaryfile)

            F_reshaped = module.analysis.frame_weight.data.reshape((module.analysis.frame_weight.shape[0], -1))
            N, C = F_reshaped.shape  # N: number of atoms, C input dimension
            if N > C:  # typical case, then better to compute F^T F (C x C)
                frame_singular_values = torch.symeig(F_reshaped.t() @ F_reshaped)[0]
            else:  # better to compute F F^T (N X N)
                frame_singular_values = torch.symeig(F_reshaped @ F_reshaped.t())[0]
            min_sing, max_sing = frame_singular_values[0], frame_singular_values[-1]
            print_and_write(
                "Parseval regularization: min {:.3f} and max {:.3f} singular values for frame {}".
                format(min_sing, max_sing, i), logfile)

            gram = torch.matmul(F_reshaped, F_reshaped.t())  # (N, N)
            idx = torch.triu_indices(*gram.shape, offset=1, device=gram.device)  # (2, N(N-1)/2)
            gram_triu = gram[idx[0], idx[1]].abs()
            print_and_write("Frame {} max coherence {:.3f}, median coherence {:.3f}".
                            format(i, gram_triu.max().item(), gram_triu.median().item()), logfile, summaryfile)

        if args.arch in ['scatnet', 'scatnetblock', 'scatnetblockanalysis']:  # no proj for analysis
            lin_p = module.linear_proj.proj.weight.data
            lin_p_reshaped = lin_p.reshape((lin_p.shape[0], -1))
            lin_singular_values = torch.symeig(lin_p_reshaped @ lin_p_reshaped.t())[0]
            min_sing, max_sing = lin_singular_values[0], lin_singular_values[-1]
            print_and_write(
                "Parseval regularization: min {:.3f} and max {:.3f} singular values for linear projector {}".
                format(min_sing, max_sing, i), logfile, summaryfile)


def one_epoch(loader, model, criterion, optimizer, epoch, args, logfile, summaryfile, writer, is_training):
    batch_time = AverageMeter('Time', ':.1f')
    data_time = AverageMeter('Data', ':.1f')
    losses = AverageMeter('Loss', ':.2f')
    top1 = AverageMeter('Acc@1', ':.1f')
    top5 = AverageMeter('Acc@5', ':.1f')
    name_epoch = "Train" if is_training else "Validation"
    progress = ProgressMeter(
        len(loader), [batch_time, data_time, losses, top1, top5],
        prefix="{} Epoch: [{}]".format(name_epoch, epoch))

    n_blocks = len(model)

    if is_training:
        model.train()
    else:
        model.eval()

    with torch.set_grad_enabled(is_training):
        end = time.time()
        for i, (input, target) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            for j in range(n_blocks):
                input = model[j](input, j=j)
            output = input

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            if is_training:
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Parseval step
                pars_update(model, args)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print_and_write('\n', logfile)
                progress.display(i, logfile)

    # Print statistics summary
    logfiles = [logfile, summaryfile]
    if not is_training and epoch == 0:
        epoch_text = ' (First epoch)'
    elif not is_training and epoch == args.epochs - 1:
        epoch_text = ' (Final epoch)'
    elif not is_training and (epoch % args.learning_rate_adjust_frequency) == (args.learning_rate_adjust_frequency - 1):
        epoch_text = ' (before learning rate adjustment n° {})'.format(1 + epoch // args.learning_rate_adjust_frequency)
    else:
        epoch_text = ''
        logfiles = [logfile, None]
    print_and_write('\n{} Epoch {}{}, * Acc@1 {:.2f} Acc@5 {:.2f}'.
                    format(name_epoch, epoch, epoch_text, top1.avg, top5.avg), *logfiles)
    if not is_training:
        print_model_info(args, model, logfile, summaryfile=None)

    if writer is not None:
        suffix = "train" if is_training else "val"
        writer.add_scalar(f'top5_{suffix}', top5.avg, global_step=epoch)
        writer.add_scalar(f'top1_{suffix}', top1.avg, global_step=epoch)

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, checkpoint_filename='checkpoint.pth.tar',
                    best_checkpoint_filename='model_best.pth.tar'):
    torch.save(state, checkpoint_filename)
    if is_best:
        shutil.copyfile(checkpoint_filename, best_checkpoint_filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, logfile):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_and_write('\t'.join(entries), logfile)

    def add(self, *meters):
        for meter in meters:
            self.meters.append(meter)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every args.learning_rate_adjust_frequency epochs"""
    lr = args.lr * (0.1 ** (epoch // args.learning_rate_adjust_frequency))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def compute_model_size(args, n_space, nb_channels=3, nb_classes=1000):
    nb_proj_params = 0
    nb_frame_params = 0
    nb_classifier_params = 0

    for i in range(args.n_blocks):
        # Compute proj parameters
        if args.arch in ['scatnet', 'scatnetblock', 'scatnetblockanalysis']:
            if args.arch != 'scatnet':
                J = 1
                A = 4
                max_order = 1
            else:
                J = args.scattering_J
                A = 1
                max_order = 2
            L_ang = args.scat_angles[i]

            factor_channels = 1 + A * L_ang * J
            if max_order == 2:
                factor_channels += (L_ang ** 2) * J * (J - 1) // 2
            nb_channels *= factor_channels
            n_space = n_space // 2 ** J

            nb_proj_params += args.P_kernel_size[i] ** 2 * nb_channels * args.P_proj_size[i]
            nb_channels = args.P_proj_size[i]

        # Compute frame parameters
        if args.arch in ['analysis', 'scatnetblockanalysis']:
            nb_frame_params += args.frame_kernel_size[i] ** 2 * nb_channels * args.frame_width[i]

    # Compute classifier parameters
    nb_classifier_params += 2 * nb_channels  # Batch norm
    if args.avg_ker_size > 1:
        n = n_space - args.avg_ker_size + 1
    else:
        n = n_space

    in_planes = nb_channels * (n ** 2)
    if args.classifier_type == 'fc':
        nb_classifier_params += in_planes * nb_classes
    else: # mlp
        nb_classifier_params += in_planes * args.hidden_units
        for i in range(args.nb_l_mlp - 1):
            nb_classifier_params += args.hidden_units**2
        nb_classifier_params += args.hidden_units * nb_classes


    nb_params = nb_classifier_params + nb_proj_params + nb_frame_params
    return nb_params, nb_proj_params, nb_frame_params, nb_classifier_params


@torch.no_grad()
def pars_update(model, args):
    beta = args.pars_beta

    def update(frame_param):
        frame = frame_param.data.reshape((frame_param.shape[0], -1))
        N, C = frame.shape  # N: number of atoms, C input dimension

        # F = (1 + beta) F - beta F F^T F ; F F^T is N x N, F^T F is C x C
        if N > C:  # typical case, better to compute F^T F (C x C)
            prod = torch.matmul(frame, torch.matmul(frame.t(), frame))
        else:  # better to compute F F^T (N X N)
            prod = torch.matmul(torch.matmul(frame, frame.t()), frame)

        frame_param.data = ((1 + beta) * frame - beta * prod).reshape(frame_param.shape)

    for i in range(len(model)):
        if args.arch in ['analysis', 'scatnetblockanalysis']:
            update(model[i].module.analysis.frame_weight)
        if args.arch in ['scatnet', 'scatnetblock', 'scatnetblockanalysis']:
            update(model[i].module.linear_proj.proj.weight)


def print_and_write(log, *logfiles):
    """ Prints log (a string) and writes it to logfiles, which should have a write method (open files) or None. """
    print(log)
    for logfile in logfiles:
        if logfile is not None:
            logfile.write(log + '\n')


if __name__ == '__main__':
    main()
