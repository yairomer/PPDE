#!python
# pylint: disable=too-many-lines,unused-argument,unnecessary-lambda,unnecessary-lambda-assignment
import os
import copy
import importlib.util
from functools import partial
import inspect
import threading
import time
import random
from collections import OrderedDict
import argparse
import datetime
import shutil
import socket
from io import StringIO,BytesIO
import base64
from contextlib import contextmanager
import uuid
from http.server import HTTPServer, SimpleHTTPRequestHandler

from line_profiler import LineProfiler
from PIL import Image
import numpy as np
import tqdm.auto as tqdm
import plotly
import plotly.graph_objects as go
import plotly.express as px
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as distrib
import torch.multiprocessing as mp
import torchvision
import torchinfo
import wandb

## Configurations
## ==============
# profile_flag = True
profile_flag = False

## Default folders locations
## -------------------------

## Example for creating a RAM disk:
## > mkdir /tmp/ramdisk
## > sudo mount -t tmpfs -o size=5g tmpfs /tmp/ramdisk/
## > rsync -avhP ~/datasets/celeba  /tmp/ramdisk/datasets/

DEFAULT_FOLDERS = dict(
    data_folders = [
        '/host/tmp/ramdisk/datasets/',
        '/host/tmp/datasets/',
        os.path.join(os.environ['HOME'], 'datasets/'),
    ],
    exp_folder = os.path.join(os.environ['HOME'], 'workspace/tmp/exp/{project_name}/{model_name}'),
    wandb_exp_folder = os.path.join(os.environ['HOME'], 'workspace/tmp/exp/{project_name}/{model_name}'),
    # checkpoints_folder = '/tmp/checkpoints/',
    checkpoints_folder = None,
    wandb_storage_folder = '/tmp/',
)

## Parameters override
## -------------------
## A place to override the default parameters
## (for quickly playing around with temporary settings).
PARAMS_OVERRIDE = dict(
    # overfit = True,
    use_wandb = False,
    # use_exp_folder = False,
    # use_current_file = True,
    # load_folder = ...
    # load_wandb_id = ...

    # ## MNIST - In-painting
    # ## -------------------
    # dataset = 'mnist',
    # img_size = None,
    # distortion_type = 'in_painting',
    # mask = (0, 19, 0, 27),
    # net_type = 'unet',
    # lr = 1e-4,
    # n_epochs = 50,
    # log_every = None,
    # benchmark_every = None,

    # ## MNIST - Denoising
    # ## -----------------
    # dataset = 'mnist',
    # img_size = None,
    # distortion_type = 'denoising1',
    # net_type = 'unet',
    # lr = 1e-3,
    # n_epochs = 600,
    # log_every = None,
    # benchmark_every = None,

    # ## CelebA 256x256 - In-painting eyes
    # ## ---------------------------------
    # dataset = 'celeba_hq_256',
    # img_size = 256,
    # distortion_type = 'in_painting',
    # mask = (80, 149, 40, 214),
    # net_type = 'res_unet_256',
    # batch_size = 16,
    # max_batch_chunk_size = 8,
    # lr = 3e-5,
    # n_epochs = 15,
    # log_every = 20,
    # benchmark_every = None,

    # ## CelebA 256x256 - In-painting mouth
    # ## ----------------------------------
    # dataset = 'celeba_hq_256',
    # img_size = 256,
    # distortion_type = 'in_painting',
    # mask = (165, 249, 55, 199),
    # net_type = 'res_unet_256',
    # batch_size = 16,
    # max_batch_chunk_size = 8,
    # lr = 3e-5,
    # n_epochs = 15,
    # log_every = 20,
    # benchmark_every = None,

    # ## CelebA 256x256 - Colorization
    # ## -----------------------------
    # dataset = 'celeba_hq_256',
    # img_size = 256,
    # distortion_type = 'colorization',
    # net_type = 'res_unet_256',
    # batch_size = 16,
    # max_batch_chunk_size = 8,
    # lr = 3e-5,
    # n_epochs = 30,
    # log_every = 20,
    # benchmark_every = None,

    # ## FFHQ - Colorization
    # ## -------------------
    # dataset = 'ffhq',
    # img_size = 256,
    # distortion_type = 'colorization',
    # net_type = 'res_unet_256',
    # batch_size = 16,
    # max_batch_chunk_size = 8,
    # lr = 3e-5,
    # n_epochs = 20,
    # log_every = 20,
    # benchmark_every = 200,

    # ## CelebA 256x256 - Super resolution
    # ## ---------------------------------
    # dataset = 'celeba_srflow',
    # img_size = 256,
    # distortion_type = 'super_resolution2',
    # net_type = 'res_cnn',
    # batch_size = 64,
    # lr = 1e-4,
    # n_epochs = 150,
    # log_every = 200,
    # benchmark_every = None,

    ## DIV2K - Super resolution
    ## ------------------------
    dataset = 'div2k_sr',
    img_size = 256,
    distortion_type = 'super_resolution1',
    net_type = 'res_cnn',
    # net_type = 'unet2',
    batch_size = 32,
    lr = 1e-4,
    # loss_type = 'mae',
    n_epochs = 4000,
    log_every = 200,
    benchmark_every = 1000,

    # ## DIV2K - Colorization
    # ## --------------------
    # dataset = 'div2k',
    # img_size = 128,
    # distortion_type = 'colorization',
    # net_type = 'res_unet_128',
    # batch_size = 32,
    # max_batch_chunk_size = 32,
    # lr = 3e-5,
    # n_epochs = 2500,
    # log_every = 20,
    # benchmark_every = 200,

    # ## ImageNet - Colorization
    # ## -----------------------
    # dataset = 'imagenet',
    # img_size = 128,
    # distortion_type = 'colorization',
    # net_type = 'res_unet_128',
    # batch_size = 16,
    # max_batch_chunk_size = 16,
    # lr = 3e-5,
    # n_epochs = 4,
    # log_every = 20,
    # benchmark_every = 200,
)

DEFAULT_PARAMS = dict(
    ## General parameters
    ## ------------------
    general = dict(
        load_wandb_id = None,
        load_folder = None,

        exp_folder = DEFAULT_FOLDERS['exp_folder'],
        wandb_exp_folder = DEFAULT_FOLDERS['wandb_exp_folder'],
        checkpoints_folder = DEFAULT_FOLDERS['checkpoints_folder'],

        ## Devices parameters
        device = 'cuda:0',
        ddp_port = '29500',
    ),

    ## Model specific parameters
    ## -------------------------
    model = dict(
        project_name = 'nppc_ebm/restore',
        name = 'exp_{now}_restore_{dataset}_{distortion_type}',
        data_folders = DEFAULT_FOLDERS['data_folders'],
        dataset = 'mnist',
        img_size = None,
        store_dataset = False,

        distortion_type = 'denoising',
        mask = None,

        net_type = 'unet',

        loss_type = 'mse',
        lr = 1e-3,
        weight_decay = 0.,

        ema_step = 1,
        ema_alpha = None,

        model_random_seed = 42,
    ),

    ## Training parameters
    ## -------------------
    train = dict(
        batch_size = 256,
        max_batch_chunk_size = None,
        n_epochs = 200,
        n_steps = None,
        gradient_clip_val = None,
        overfit = False,

        ## Benchmark parameters
        max_benchmark_samples = 256,

        ## Logging parameters
        log_every = None,
        benchmark_every = None,
        restart_logging = False,
        use_exp_folder = True,
        use_wandb = True,

        html_min_interval = 60.,
        save_min_interval = 30. * 60.,
        n_points_to_save = 1,

        initial_port = 8000,

        ## Weights and biases parameters
        wandb_storage_folder = DEFAULT_FOLDERS['wandb_storage_folder'],
        wandb_entity = 'sipl',

        ## General parameters
        num_workers = 0,
        train_random_seed = 43,
    ),
)

## Lambda function parameters
## --------------------------
LR_LAMBDA = lambda step: 1
# LR_LAMBDA = lambda step: 0.5 ** (step // 5000)  # Step decay


## Profiler
## ========
class Profiler:
    def __init__(self, enable=True, output_filename='/tmp/profile.txt'):
        self._enable = enable
        self._funcs_to_profile = []
        self._output_filenmae = output_filename

    @property
    def enable(self):
        return self._enable

    def add_function(self, func):
        if self.enable:
            self._funcs_to_profile.append(func)
        return func

    @contextmanager
    def run_and_profile(self):
        if len(self._funcs_to_profile) > 0:
            profiler = LineProfiler()
            for func in self._funcs_to_profile:
                profiler.add_function(func)
            profiler.enable_by_count()
            try:
                yield
            finally:
                with StringIO() as str_stream:
                    profiler.print_stats(str_stream)
                    string = str_stream.getvalue()
                print(f'Writing profile data to "{self._output_filenmae}"')
                with open(self._output_filenmae, 'w', encoding='utf-8') as fid:
                    fid.write(string)
        else:
            yield

    def top_func(self, func):
        def wrapper_func(*args, **kwargs):
            with self.run_and_profile():
                func(*args, **kwargs)
        return wrapper_func

global_profiler = Profiler(enable=profile_flag)
## Use a "@global_profiler.top_func" decorator on the top function in your code.
## The top function should be called only once in code.
## Use a "@global_profiler.add_function" decorator to profile a function.
## The profiling results will be saved to "/tmp/profile.txt" at the end of the run.


## Model
## =====
class Model(object):  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    def __init__(self, device='cpu', **params):

        self.device = device
        self.ddp = DDPManager()

        store_dataset = params.pop('store_dataset')
        self.params = params
        self.extra_data = {}

        ## Initializing random state
        ## -------------------------
        set_random_seed(self.params['model_random_seed'])

        ## Run name
        ## --------
        name = self.params['name']
        if name is None:
            self.name = 'exp_{now}'
        now = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
        self.name = name.format(now=now, **params)

        ## Prepare datasets
        ## ----------------
        if self.params['dataset'] == 'mnist':
            self.data_module = MNISTDataModule(data_folders=self.params['data_folders'], remove_labels=True, n_valid=0, device=self.device)
        elif self.params['dataset'] == 'imagenet':
            self.data_module = ImageNetDataModule(img_size=int(self.params['img_size']), data_folders=self.params['data_folders'], store_dataset=store_dataset)
        elif self.params['dataset'] == 'celeba_hq_256':
            self.data_module = CelebAHQ256DataModule(img_size=int(self.params['img_size']), data_folders=self.params['data_folders'], store_dataset=store_dataset)
        elif self.params['dataset'] == 'celeba_srflow':
            self.data_module = CelebASRFlowDataModule(data_folders=self.params['data_folders'], scale=8, store_dataset=store_dataset)
        elif self.params['dataset'] == 'ffhq':
            self.data_module = FFHQDataModule(img_size=int(self.params['img_size']), data_folders=self.params['data_folders'], store_dataset=store_dataset)
        elif self.params['dataset'] == 'div2k':
            self.data_module = DIV2KDataModule(img_size=int(self.params['img_size']), data_folders=self.params['data_folders'], store_dataset=store_dataset)
        elif self.params['dataset'] == 'div2k_sr':
            self.data_module = DIV2KSRDataModule(img_size=int(self.params['img_size']), data_folders=self.params['data_folders'], scale=4, store_dataset=store_dataset)
        else:
            raise Exception(f'Unsupported dataset: "{self.params["dataset"]}"')

        self.x_shape = self.data_module.shape

        ## Prepare distortion model
        ## ------------------------
        distortion_type = self.params['distortion_type'].lower()
        self.mask = None
        self.upscale_factor = None
        self.x_distorted_shape = self.x_shape

        if distortion_type == 'denoising1':
            self.distortion_model = Denoising(noise_std=1., clip_noise=True)
        elif distortion_type == 'in_painting':
            top, bottom, left, right = self.params['mask']
            self.mask = torch.zeros(self.x_shape, device=self.device)
            self.mask[:, top:(bottom + 1), left:(right + 1)] = 1.
            self.distortion_model = InPainting(mask=self.mask, fill=self.data_module.mean).to(self.device)
        elif distortion_type == 'colorization':
            self.distortion_model = Colorization().to(self.device)
            self.x_distorted_shape = (1,) + self.x_shape[1:]
        elif distortion_type == 'super_resolution1':
            self.distortion_model = SuperResolution(factor=4)
            self.upscale_factor = 4
            self.x_distorted_shape = (self.x_shape[0], self.x_shape[1] // 4, self.x_shape[2] // 4)
        elif distortion_type == 'super_resolution2':
            self.distortion_model = SuperResolution(factor=8)
            self.upscale_factor = 8
            self.x_distorted_shape = (self.x_shape[0], self.x_shape[1] // 8, self.x_shape[2] // 8)
        elif distortion_type == 'super_resolution3':
            self.distortion_model = SuperResolution(factor=8, noise_std=0.05)
            self.upscale_factor = 8
            self.x_distorted_shape = (self.x_shape[0], self.x_shape[1] // 8, self.x_shape[2] // 8)

        # elif distortion_type == 'super_resolution3':
        #     self.distortion_model = SuperResolution(factor=8)
        # elif distortion_type == 'super_resolution4':
        #     self.distortion_model = SuperResolution(factor=8, noise_std=0.05)

        else:
            raise Exception(f'Unsupported distortion_type: "{distortion_type}"')

        ## Set parametric model
        ## --------------------
        pad_base_size = None

        net_type = self.params['net_type'].lower()
        if net_type == 'unet':
            base_net = UNet(
                channels_in=self.x_distorted_shape[0],
                channels_out=self.x_shape[0],
                channels_list=(32, 64, 128, 256),
                n_blocks_list=(1, 1, 1, 2),
                upscale_factor=self.upscale_factor,
            )
            pad_base_size = 2 ** 3

        elif net_type == 'res_unet_128':
            ## DDPM
            base_net = ResUNet(
                in_channels=self.x_distorted_shape[0],
                out_channels=self.x_shape[0],
                channels_list=(64, 128, 128, 256, 256),
                bottleneck_channels=512,
                downsample_list=(False, True, True, True, True),
                attn_list=(False, False, False, True, False),
                n_blocks=2,
                # min_channels_decoder=64,
                n_groups=8,
                attn_heads=1,
                upscale_factor=self.upscale_factor,
            )
            pad_base_size = 2 ** 4

        elif net_type == 'res_unet_256':
            ## DDPM
            base_net = ResUNet(
                in_channels=self.x_distorted_shape[0],
                out_channels=self.x_shape[0],
                # channels_list=(128, 128, 256, 256, 512, 512),
                channels_list=(64, 64, 128, 128, 256, 256),
                bottleneck_channels=512,
                downsample_list=(False, True, True, True, True, True),
                attn_list=(False, False, False, False, True, False),
                n_blocks=2,
                # min_channels_decoder=64,
                n_groups=8,
                attn_heads=1,
                upscale_factor=self.upscale_factor,
            )
            pad_base_size = 2 ** 5

        elif net_type == 'res_cnn':
            base_net = ResCNN(
                channels_in=self.x_distorted_shape[0],
                channels_out=self.x_shape[0],
                channels_hidden=64,
                n_blocks=16,
                upscale_factor=self.upscale_factor,
            )

        else:
            raise Exception(f'Unsupported net_type: "{net_type}"')

        net = RestorationWrapper(
            net=base_net,
            offset=self.data_module.mean,
            scale=self.data_module.std,
            mask=self.mask,
            pad_base_size=pad_base_size,
            naive_restore_func=self.distortion_model.naive_restore,
        )

        ## Set network wrapper (optimizer, scheduler, ema & ddp)
        self.networks = {}
        self.networks['net'] = NetWrapper(
            net,
            optimizer_type='adam',
            optimizer_params=dict(lr=self.params['lr'], betas=(0.9, 0.999), weight_decay=self.params['weight_decay']),
            lr_lambda=LR_LAMBDA,
            ema_alpha=self.params['ema_alpha'],
            ema_update_every=self.params['ema_step'],
            device=self.device,
            ddp_active=self.ddp.is_active,
        )

    def __getitem__(self, net_name):
        return self.networks[net_name]

    def split_batch(self, batch, n):
        if isinstance(batch, (tuple, list)):
            batches = tuple(zip(*[torch.chunk(batch_, n, dim=0) for batch_ in batch]))
        else:
            batches = torch.chunk(batch, n, dim=0)
        return batches

    def process_batch(self, batch):
        if isinstance(batch, (tuple, list)):
            x_org = batch[0].to(self.device)
            x_distorted = batch[1].to(self.device)
        else:
            x_org = batch.to(self.device)
            x_distorted = self.distort(x_org)
        return x_org, x_distorted

    def distort(self, x):
        x_distorted = self.distortion_model.distort(x)
        return x_distorted

    def restore(self, x_distorted, use_best=True, **kwargs):
        x_restored = self['net'](x_distorted, use_best=use_best, **kwargs)
        return x_restored

    ## Save and load
    ## -------------
    def state_dict(self):
        state_dict = dict(
            name = self.name,
            params = self.params,
            extra_data = self.extra_data,
            networks = {key: network.state_dict() for key, network in self.networks.items()},
            )
        return state_dict

    def load_state_dict(self, state_dict):
        self.name = state_dict['name']
        self.extra_data = state_dict['extra_data']
        for key, val in state_dict['networks'].items():
            self.networks[key].load_state_dict(val)

    @classmethod
    def load(cls, checkpoint_filename, device='cpu', **kwargs):
        state_dict = torch.load(checkpoint_filename, map_location=device)
        state_dict['params'].update(kwargs)

        model = cls(device=device, **state_dict['params'])
        model.load_state_dict(state_dict)
        step_str = ', '.join([f'{key}: {net.step}' for key, net in model.networks.items()])
        print(f'Resuming step: {step_str}')
        return model


## Distortion models
## =================
class Denoising(nn.Module):
    def __init__(
            self,
            noise_std,
            clip_noise=False,
        ):

        super().__init__()
        self.noise_std = noise_std
        self.clip_noise= clip_noise

    def distort(self, x):
        x_distorted = x + torch.randn_like(x) * self.noise_std
        if self.clip_noise:
            x_distorted = x_distorted.clamp(0, 1)
        return x_distorted

    def naive_restore(self, x):
        return x

    def forward(self, x):
        return self.distort(x)


class InPainting(nn.Module):
    def __init__(
            self,
            mask,
            fill=0.
        ):

        super().__init__()
        self.fill = fill
        self.register_buffer('mask', mask)

    def distort(self, x):
        x = x * (1 - self.mask) + self.fill * self.mask
        return x

    def naive_restore(self, x):
        return x

    def forward(self, x):
        return self.distort(x)


class Colorization(nn.Module):
    def distort(self, x):
        # x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = x.mean(dim=1, keepdim=True)
        return x

    def naive_restore(self, x):
        x = x.repeat_interleave(3, dim=1)
        return x

    def forward(self, x):
        return self.distort(x)


class SuperResolution(nn.Module):
    def __init__(
            self,
            factor,
            noise_std=0.,
        ):

        super().__init__()
        self.factor = factor
        self.noise_std = noise_std

    def distort(self, x):
        x = F.avg_pool2d(x, self.factor)
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        return x

    def naive_restore(self, x):
        x = F.interpolate(x, scale_factor=self.factor, mode='nearest')
        return x

    def forward(self, x):
        return self.distort(x)


## Trainer
## =======
class Trainer(object):
    def __init__(self, model, **params):
        self.model = model
        self.params = params
        self.train_log_data = None

        self.ddp = self.model.ddp

        ## Set batch size
        ## --------------
        self.batch_size = self.params['batch_size'] // self.ddp.size

        ## Use batches accumulation if necessary
        if (self.params['max_batch_chunk_size'] is not None) and (self.batch_size > params['max_batch_chunk_size']):
            self.n_batch_accumulation = self.batch_size // params['max_batch_chunk_size']
            self.batch_size = self.batch_size // self.n_batch_accumulation * self.n_batch_accumulation
        else:
            self.n_batch_accumulation = 1

        ## Store a fixed batch from the validation set (for visualizations and benchmarking)
        ## ---------------------------------------------------------------------------------
        dataloader = torch.utils.data.DataLoader(
            model.data_module.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(0),
        )
        self.fixed_batch = next(iter(dataloader))

        if not self.params['overfit']:
            dataloader = torch.utils.data.DataLoader(
                model.data_module.valid_set,
                batch_size=self.batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(0),
            )
            self.valid_batch = next(iter(dataloader))
        else:
            self.valid_batch = self.fixed_batch

        self.header = []
        self.folder_manager = None
        self.status_msgs = OrderedDict([(key, f'--({key})--') for key in ('state', 'step', 'next', 'lr',
                                                                          'train', 'fixed', 'valid', 'fullv')])
        self._status_msgs_h = None
        padding_size = 2
        self.preview_width = min((780 - padding_size) // (model.x_shape[-1] + padding_size), self.batch_size)
        self.wandb = None

    def train(self):
        model = self.model

        ## Test step
        ## ---------
        self.base_step(model.split_batch(self.fixed_batch, self.n_batch_accumulation)[0])

        ## Initializing random state
        ## -------------------------
        set_random_seed(self.params['train_random_seed'])

        ## PyTorch settings
        ## ---------------
        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True  # !!Warning!!: This makes things run A LOT slower
        # torch.autograd.set_detect_anomaly(True)  # !!Warning!!: This also makes things run slower

        ## Initializing data loaders
        ## -------------------------
        sampler = torch.utils.data.distributed.DistributedSampler(
            model.data_module.train_set,
            num_replicas=self.ddp.size,
            rank=self.ddp.rank,
            shuffle=True,
            drop_last=True,
        )
        dataloader = torch.utils.data.DataLoader(
            model.data_module.train_set,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.params['num_workers'],
            persistent_workers=self.params['num_workers'] > 0,
            drop_last=True,
            # pin_memory=True,
        )

        ## Initialize train logging
        ## ------------------------
        benchmark_every = self.params['benchmark_every']
        if benchmark_every is None:
            benchmark_every = len(dataloader)

        if self.ddp.is_main:
            self.train_log_data = self._init_train_log_data()

            ## Set up experiment folder
            ## ------------------------
            if self.params['use_exp_folder']:
                self.init_exp_folder()
                self.folder_manager.serve_folder(self.params['initial_port'])

                ## Set up weights and biases logger
                ## --------------------------------
                if self.params['use_wandb']:
                    self.init_wandb()

            ## Log model
            ## ---------
            self.header.append(f'Project name: {model.params["project_name"]}')
            self.header.append(f'Model name: {model.name}')
            self.header.append(f'Batch size: {self.batch_size * self.ddp.size} = {self.batch_size // self.n_batch_accumulation} x {self.n_batch_accumulation} x {self.ddp.size} (batch part x accumulation x GPUs)')
            self.header.append(f'Number of steps in epoch: {len(dataloader)}; Logging every {self.params["log_every"]}. Benchmarking every {benchmark_every}')
            self.log_model()

            ## Set up console logging
            ## ----------------------
            print('\n'.join(self.header))
            self.init_msgs()

        ## Initialize timers
        ## -----------------
        timers = dict(
            html = Timer(self.params['html_min_interval'], reset=False),
            save = Timer(self.params['save_min_interval'], reset=True),
        )

        ## Training loop
        ## -------------
        if global_profiler.enable:
            n_steps = 10
        elif self.params['n_epochs'] is not None:
            n_steps = self.params['n_epochs'] * len(dataloader)
        else:
            n_steps = self.params['n_steps']

        model['net'].train()
        train_data_iter = iter(loop_loader(dataloader))
        for i_step in tqdm.trange(n_steps, ncols=0, disable=not self.ddp.is_main):
            last = i_step == n_steps - 1
            if self.ddp.is_main:
                self.set_msg('state', 'Training...')
                step_str = ', '.join([f'{key}: {net.step}' for key, net in model.networks.items()])
                lr_str = ', '.join([f'{key}: {net.lr}' for key, net in model.networks.items()])
                if self.wandb is not None:
                    self.set_msg('step', f'Step: {step_str} (W&B step: {self.wandb.step})')
                else:
                    self.set_msg('step', f'Step: {step_str}')
                self.set_msg('next', 'Next: ' + ', '.join([f'{key}: {val}' for key, val in timers.items()]))
                self.set_msg('lr', f'Learning rate: {lr_str}')

            ## Train step
            ## ----------
            net = model['net']

            batch = next(train_data_iter)
            if self.params['overfit']:
                batch = self.fixed_batch

            logs = []
            net.optimizer.zero_grad()
            for i_accum, batch_part in enumerate(model.split_batch(batch, self.n_batch_accumulation)):
                last_part = i_accum == self.n_batch_accumulation - 1
                with net.set_ddp_sync(last_part):
                    objective, log = self.base_step(batch_part)
                    objective = objective / self.n_batch_accumulation
                    objective.backward()
                logs.append(log)
            train_log = self.cat_logs(logs)
            net.clip_grad_norm(self.params['gradient_clip_val'])
            net.optimizer.step()
            net.increment()

            ## Logging, benchmarking & save
            ## ----------------------------
            if self.ddp.is_main:
                benchmark_flag = ((i_step + 1) % benchmark_every == 0) or last
                if self.params['log_every'] is not None:
                    log_flag = (net.step == 1) or (net.step % self.params['log_every'] == 0) or benchmark_flag
                else:
                    log_flag = benchmark_flag

                if log_flag:
                    net.eval()

                    # self.log_step(train_log, 'train')
                    self.set_msg('state', 'Running fixed batch...')
                    with EncapsulatedRandomState(42):
                        logs = []
                        for batch_part in model.split_batch(self.fixed_batch, self.n_batch_accumulation):
                            with torch.no_grad():
                                _, log = self.base_step(batch_part)
                            logs.append(log)
                        fixed_log = self.cat_logs(logs)
                        self.log_step(fixed_log, 'fixed')

                        logs = []
                        for batch_part in model.split_batch(self.valid_batch, self.n_batch_accumulation):
                            with torch.no_grad():
                                _, log = self.base_step(batch_part)
                            logs.append(log)
                        valid_log = self.cat_logs(logs)
                        self.log_step(valid_log, 'valid')

                        self.post_log_valid(train_log, fixed_log, valid_log)

                        if benchmark_flag:
                            self.set_msg('state', 'Running benchmark...')
                            score = self.benchmark()  # pylint: disable=assignment-from-none
                            net.update_best(score)
                            if timers['save']:
                                self.set_msg('state', 'Saving...')
                                self.save_checkpoint()
                                timers['save'].reset()

                    self.log_wandb()
                    if timers['html']:
                        self.set_msg('state', 'Writing HTML ...')
                        self.log_html()
                        timers['html'].reset()
                    net.train()

        self.set_msg('state', 'Saving...')
        self.save_checkpoint()

    def base_step(self, batch):
        model = self.model

        x_org, x_distorted = model.process_batch(batch)
        x_restored = model.restore(x_distorted, use_best=False, use_ddp=True)
        err = x_restored - x_org

        ## Mean vector loss
        ## ----------------
        err_mse = err.pow(2).flatten(1).mean(dim=-1)
        err_mae = err.abs().flatten(1).mean(dim=-1)

        loss_type = model.params['loss_type'].lower()
        if loss_type == 'mse':
            objective = err_mse.mean()
        elif loss_type == 'mae':
            objective = err_mae.mean()
        else:
            raise Exception(f'Unsupported loss type "{loss_type}"')

        ## Store log
        ## ---------
        log = dict(
            x_org=x_org,
            x_distorted=x_distorted,
            x_restored=x_restored.detach(),

            err_mse=err_mse.detach(),
            err_mae=err_mae.detach(),

            objective=objective.detach(),
        )
        return objective, log

    ## Logging
    ## -------
    def _init_train_log_data(self):
        model = self.model
        with EncapsulatedRandomState(42):
            x_org_fixed, x_distorted_fixed = model.process_batch(model.split_batch(self.fixed_batch, self.n_batch_accumulation)[0])
        with EncapsulatedRandomState(42):
            x_org_valid, x_distorted_valid = model.process_batch(model.split_batch(self.valid_batch, self.n_batch_accumulation)[0])

        train_log_data = {}

        train_log_data['general'] = {}

        ## Prepare running logs fields
        ## ---------------------------
        model.extra_data['train_params'] = self.params
        if 'train_log_data' not in model.extra_data:
            model.extra_data['train_log_data'] = {}
        if 'logs' in model.extra_data['train_log_data']:
            logs = model.extra_data['train_log_data']['logs']
        else:
            logs = {f'{key}_{field}': [] for field in ('train', 'fixed', 'valid', 'fullv')
                                         for key in ('step', 'lr', 'objective', 'err_rmse', 'err_mae')}
            model.extra_data['train_log_data']['logs'] = logs
        train_log_data['logs'] = logs

        ## Prepare summary
        ## ---------------
        summary = torchinfo.summary(model['net'].net, input_data=x_distorted_fixed[:1], depth=10, verbose=0, device=model.device)
        # summary.formatting.verbose = 2
        summary = str(summary)
        train_log_data['summary'] = summary

        ## Prepare images
        ## --------------
        imgs = {
            'x_org_fixed': imgs_to_grid(x_org_fixed),
            'x_distorted_fixed': imgs_to_grid(model.distortion_model.naive_restore(x_distorted_fixed)),
            'x_org_valid': imgs_to_grid(x_org_valid),
            'x_distorted_valid': imgs_to_grid(model.distortion_model.naive_restore(x_distorted_valid)),

            'batch_train': imgs_to_grid(torch.zeros((4, self.preview_width) + model.x_shape)),
            'batch_fixed': imgs_to_grid(torch.zeros((4, self.preview_width) + model.x_shape)),
            'batch_valid': imgs_to_grid(torch.zeros((4, self.preview_width) + model.x_shape)),
            'batch_fullv': imgs_to_grid(torch.zeros((4, self.preview_width) + model.x_shape)),
        }
        train_log_data['imgs'] = imgs

        ## Prepare figures
        ## ---------------
        figs = {}

        figs['lr'] = go.Figure(
            data=[go.Scatter(mode='lines', x=[], y=[])],
            layout=go.Layout(yaxis_title='lr', xaxis_title='step', yaxis_type='log', height=400, width=550, margin=dict(t=0, b=20, l=20, r=0)),
        )

        figs['objective'] = go.Figure(
            data=[go.Scatter(mode='lines', x=[], y=[], name=title) for title in ('Train', 'Fixed', 'Validation', 'Full validation')],
            layout=go.Layout(yaxis_title='Objective', xaxis_title='step', height=400, width=550, margin=dict(t=0, b=20, l=20, r=0)),
        )

        figs['err_rmse'] = go.Figure(
            data=[go.Scatter(mode='lines', x=[], y=[], name=title) for title in ('Train', 'Fixed', 'Validation', 'Full validation')],
            layout=go.Layout(yaxis_title='Error RMSE', xaxis_title='step', height=400, width=550, margin=dict(t=0, b=20, l=20, r=0)),
        )

        figs['err_mae'] = go.Figure(
            data=[go.Scatter(mode='lines', x=[], y=[], name=title) for title in ('Train', 'Fixed', 'Validation', 'Full validation')],
            layout=go.Layout(yaxis_title='Error MAE', xaxis_title='step', height=400, width=550, margin=dict(t=0, b=20, l=20, r=0)),
        )

        train_log_data['figs'] = figs

        return train_log_data

    def cat_logs(self, logs):
        log = {}
        for key in logs[0].keys():
            if key in ('x_org', 'x_distorted', 'x_restored', 'err_mse', 'err_mae'):
                log[key] = torch.cat([log_[key] for log_ in logs], dim=0)
            elif key in ('objective',):
                total = 0.
                total_n = 0
                for log_ in logs:
                    n = log_['x_org'].shape[0]
                    total += log_[key] * n
                    total_n += n
                log[key] = total / total_n
            else:
                raise Exception(f'Unknown key {key}')

        return log

    def log_step(self, log, field):
        model = self.model

        ## Prepare data
        ## ------------
        step = model.networks['net'].step
        lr = model.networks['net'].lr

        x_org = log['x_org']
        x_distorted = log['x_distorted']
        x_restored = log['x_restored']

        err_mse_full = log['err_mse']
        err_mae_full = log['err_mae']

        objective = log['objective']


        err_mse = err_mse_full.mean().item()
        err_mae = err_mae_full.mean().item()
        err_rmse = err_mse ** 0.5
        objective = objective.item()

        x_org = self.sample_preview_width(x_org)
        x_distorted = self.sample_preview_width(x_distorted)
        x_restored = self.sample_preview_width(x_restored)

        err = x_restored - x_org

        if x_org.shape[1] == 1:
            x_org = x_org.repeat_interleave(3, 1)
            x_distorted = x_distorted.repeat_interleave(3, 1)
            x_restored = x_restored.repeat_interleave(3, 1)
            err_scaled = err.repeat_interleave(3, 1)
            for i in range(err_scaled.shape[0]):
                err_scaled[i] = color_img(err[i], max_val=torch.abs(err[i]).max() / 1.5)
        else:
            # err_scaled = torch.abs(err)
            # err_scaled = err_scaled / err_scaled.flatten(-3).max(-1)[0][..., None, None, None]
            err_scaled = err / torch.abs(err).flatten(-3).max(-1)[0][..., None, None, None] / 1.5 + 0.5

        x_distorted = model.distortion_model.naive_restore(x_distorted)

        batch_img = imgs_to_grid(torch.stack((x_org, x_distorted, x_restored, err_scaled)))

        ## Update console message
        ## ----------------------
        self.set_msg(field, f'{field}: step:{step:7d};   err_rmse: {err_rmse:9.4g};   err_mae: {err_mae:9.4g};   objective: {objective:9.4g}')

        ## Store log data
        ## --------------
        logs = self.train_log_data['logs']
        logs[f'step_{field}'].append(step)
        logs[f'lr_{field}'].append(lr)
        logs[f'objective_{field}'].append(objective)
        logs[f'err_rmse_{field}'].append(err_rmse)
        logs[f'err_mae_{field}'].append(err_mae)

        figs = self.train_log_data['figs']
        if field in ('fixed',):
            figs['lr'].data[0].update(x=logs['step_fixed'], y=logs['lr_fixed'])
        for key in ('objective', 'err_rmse', 'err_mae'):
            figs[key].data[('train', 'fixed', 'valid', 'fullv').index(field)].update(x=logs[f'step_{field}'], y=logs[f'{key}_{field}'])

        imgs = self.train_log_data['imgs']
        imgs[f'batch_{field}'] = batch_img

        ## Log to weights and biases
        ## -------------------------
        if self.wandb is not None:
            if field in ('fixed',):
                self.wandb.log({
                    'train_params/lr': lr,
                }, commit=False)

            self.wandb.log({
                f'objective1/objective_{field}': objective,
            }, commit=False)

            if field not in ('train',):
                self.wandb.log({
                    f'imgs/batch_imgs_{field}': wandb.Image(batch_img),
                }, commit=False)

    def post_log_valid(self, train_log, fixed_log, valid_log):
        pass

    def benchmark(self):
        model = self.model

        dataset = model.data_module.valid_set
        indices = np.random.RandomState(42).permutation(len(dataset))  # pylint: disable=no-member
        indices = indices[:self.params['max_benchmark_samples']]
        dataset = torch.utils.data.Subset(dataset, indices)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.params['num_workers'],
            persistent_workers=self.params['num_workers'] > 0,
            # pin_memory=True,
        )

        logs = []
        for batch in tqdm.tqdm(dataloader, dynamic_ncols=True, leave=False):
            with torch.no_grad():
                _, log = self.base_step(batch)
            logs.append(log)
        log = self.cat_logs(logs)
        self.log_step(log, 'fullv')

        score = log['err_mse'].mean().item()

        return score

    def log_html(self):
        if self.folder_manager is not None:
            model = self.model

            fields = model.params.copy()
            fields.update(self.train_log_data['general'])
            fields['header'] = '<br>\n'.join(self.header + list(self.status_msgs.values()))
            fields['now'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fields['summary'] = '<div style="white-space:pre;">' + self.train_log_data['summary'] + '</div>'

            for key, val in self.train_log_data['imgs'].items():
                fields[f'{key}_img'] = img_to_html(val)

            for key, val in self.train_log_data['figs'].items():
                fields[f'{key}_fig'] = fig_to_html(val)

            html = inspect.cleandoc("""
            <!doctype html>
            <html lang="en">
            <head>
                <meta charset="utf-8" />
                <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no"/>
                <!--<meta http-equiv="refresh" content="30" />-->
                <script src="https://cdn.plot.ly/plotly-2.18.2.min.js"></script>
                <style>
                html {{font-size: 16px; line-height: 1.25em;}}
                .twoColumns {{display: grid; grid-template-columns: repeat(auto-fit, minmax(750px, 1fr));}}
                </style>
            </head>
            <body style="background-color: #4c5e79">
                <div style="max-width: 1600px; padding-right:15px; padding-left: 15px; margin: auto">
                <h1>Results</h1>

                <div>
                    <h2>Experiment detailes</h2>
                    {header}<br>
                    <br>
                    Updated at: {now} (GMT)<br>
                </div>

                <div>
                    <h2>Training images</h2>
                    <div class="twoColumns">
                    {batch_fixed_img}
                    {batch_valid_img}
                    </div>
                </div>

                <div>
                    <h2>Training metrics</h2>

                    {objective_fig}
                    <br>
                    {err_rmse_fig}
                    {err_mae_fig}
                    <br>
                    {lr_fig}
                </div>

                <div>
                    <h2>Model</h2>

                    <h3>Train data</h3>
                    {x_org_fixed_img}
                    {x_distorted_fixed_img}
                    <br>
                    {x_org_valid_img}
                    {x_distorted_valid_img}
                </div>
                
                <div>
                    <h2>Networks</h2>
                    {summary}<br>
                </div>
                </div>
            </body>
            </html>
            """).format(**fields)

            self.folder_manager.write_file('index.html', html)

    def log_wandb(self):
        model = self.model

        if self.wandb is not None:
            self.wandb.log({
                'train_params/model_step': model['net'].step,
            }, commit=True)

    def log_model(self):
        model = self.model

        ## Log to weights and biases
        ## -------------------------
        if (model.networks['net'].step == 0) and (self.wandb is not None):
            self.wandb.log({
                'model/summary': wandb.Html(self.train_log_data['summary']),
                'model/x_org_fixed': wandb.Image(self.train_log_data['imgs']['x_org_fixed']),
                'model/x_distorted_fixed': wandb.Image(self.train_log_data['imgs']['x_distorted_fixed']),
                'model/x_org_vaild': wandb.Image(self.train_log_data['imgs']['x_org_vaild']),
                'model/x_distorted_vaild': wandb.Image(self.train_log_data['imgs']['x_distorted_vaild']),
                }, commit=True)

    def sample_preview_width(self, x):
        indices = np.linspace(0, x.shape[0] - 1, self.preview_width).astype(int)
        x = x[indices]
        return x

    ## Console logging
    ## ---------------
    def init_msgs(self):
        self._status_msgs_h = {key: tqdm.tqdm([], desc=msg, bar_format='{desc}') for key, msg in self.status_msgs.items()}

    def set_msg(self, key, msg):
        self.status_msgs[key] = msg
        if self._status_msgs_h is not None:
            self._status_msgs_h[key].set_description_str(msg)

    def init_exp_folder(self):
        model = self.model
        self.folder_manager = ExpFolderManager(
            self.params['exp_folder'].format(model_name=model.name, **model.params),
            checkpoints_folder = self.params['checkpoints_folder'],
        )
        if self.params['restart_logging']:
            self.folder_manager.create(reset=True)
            self.folder_manager.save_exp_file()

    ## Weights and biases
    ## ------------------
    def init_wandb(self):
        model = self.model
        if self.params['restart_logging']:
            self.params['wandb_run_id'] = None
        elif self.params['wandb_run_id'] is None:
            self.params['wandb_run_id'] = model.extra_data.get('wandb_run_id', None)

        self.wandb = init_wandb_logger(
            run_name=model.name,
            entity=self.params['wandb_entity'],
            project=model.params['project_name'],
            storage_folder=self.params['wandb_storage_folder'],
            run_id=self.params['wandb_run_id'],
            params=model.params,
        )
        wandb_log_exp_file(self.wandb, self.folder_manager.exp_filename)
        model.extra_data['wandb_run_id'] = self.wandb.id
        model.extra_data['wandb_entity'] = self.wandb.entity

    ## Checkpoints
    ## -----------
    def save_checkpoint(self):
        if self.folder_manager is not None:
            self.folder_manager.save_checkpoint(self.model.state_dict())
            if self.wandb is not None:
                self.wandb.save(self.folder_manager.checkpoint_filename, base_path=os.path.dirname(self.folder_manager.checkpoint_filename))
                if (self.folder_manager.intermediate_checkpoint_filename is not None) and os.path.isfile(self.folder_manager.intermediate_checkpoint_filename):
                    self.wandb.save(self.folder_manager.intermediate_checkpoint_filename, base_path=self.folder_manager.intermediate_checkpoint_folder)


## Networks auxiliaries
## ====================
def zero_weights(module, factor=1e-6):
    module.weight.data = module.weight.data * factor
    if hasattr(module, 'bias') and (module.bias is not None):
        nn.init.constant_(module.bias, 0)
    return module


class ShortcutBlock(nn.Module):
    def __init__(self, base, shortcut=None, factor=None):
        super().__init__()

        self.base = base
        self.shortcut = shortcut
        self.factor = factor

    def forward(self, x):
        shortcut = x
        x = self.base(x)
        if self.shortcut is not None:
            shortcut = self.shortcut(shortcut)
        if self.factor is not None:
            x = x * self.factor
        x = x + shortcut
        return x


class ConcatShortcutBlock(nn.Module):
    def __init__(self, base, shortcut=None, factor=None):
        super().__init__()

        self.base = base
        self.shortcut = shortcut
        self.factor = factor

    def forward(self, x):
        shortcut = x
        x = self.base(x)
        if self.shortcut is not None:
            shortcut = self.shortcut(shortcut)
        if self.factor is not None:
            x = x * self.factor
        x = torch.cat((x, shortcut), dim=1)
        return x


class RestorationWrapper(nn.Module):
    def __init__(self, net, offset=None, scale=None, mask=None, pad_base_size=None, naive_restore_func=None):
        super().__init__()
        self.net = net
        self.offset = offset
        self.scale = scale
        self.mask = mask
        self.pad_base_size = pad_base_size
        self.naive_restore_func = naive_restore_func

    def _get_padding(self, x):
        s = self.pad_base_size
        _, _, height, width = x.shape
        if (s is not None) and ((height % s != 0) or (width % s != 0)):
            pad_h = height % s
            pad_w = width % s
            padding = torch.tensor((pad_h // 2, pad_h // 2, pad_w // 2, pad_w // 2))
        else:
            padding = None
        return padding

    def forward(self, x_distorted):
        x_in = x_distorted
        x_naive = self.naive_restore_func(x_distorted)
        if self.offset is not None:
            x_distorted = x_distorted - self.offset
            x_naive = x_naive - self.offset
        if self.scale is not None:
            x_distorted = x_distorted / self.scale
            x_naive = x_naive / self.scale

        padding = self._get_padding(x_distorted)
        if padding is not None:
            x_distorted = F.pad(x_distorted, tuple(padding))

        x_restored = self.net(x_distorted)

        if padding is not None:
            x_restored = F.pad(x_restored, tuple(-padding))  # pylint: disable=invalid-unary-operand-type
        
        x_restored = x_naive + x_restored

        if self.scale is not None:
            x_restored = x_restored * self.scale
        if self.offset is not None:
            x_restored = x_restored + self.offset

        if self.mask is not None:
            x_restored = x_in * (1 - self.mask[None]) + x_restored * self.mask[None]

        return x_restored


class ResBlock(nn.Module):
    def __init__(self, dim, dim_out, n_groups=8):
        super().__init__()

        self.block = ShortcutBlock(
            nn.Sequential(
                nn.Conv2d(dim, dim_out, 3, padding=1),
                nn.GroupNorm(n_groups, dim_out),
                nn.SiLU(),
                nn.Conv2d(dim_out, dim_out, 3, padding=1),
                nn.GroupNorm(n_groups, dim_out),
                nn.SiLU(),
            ),
            shortcut= nn.Conv2d(dim, dim_out, 1) if dim != dim_out else None,
        )

    def forward(self, x):
        return self.block(x)


class Attention(nn.Module):
    def __init__(
        self,
        in_channels,
        embedding_channels=None,
        heads=4,
    ):
        super().__init__()
        self.heads = heads

        if embedding_channels is None:
            embedding_channels = in_channels

        self.conv_in = nn.Conv1d(in_channels, 3 * embedding_channels, 1, bias=False)
        self.conv_out = zero_weights(nn.Conv1d(embedding_channels, in_channels, 1))

    def forward(self, x):
        x_in = x
        shape = x.shape
        x = x.flatten(2)

        x = self.conv_in(x)
        x = x.unflatten(1, (3, self.heads, -1))
        q, k, v = x[:, 0], x[:, 1], x[:, 2]

        attn = torch.einsum(f"bhki,bhka->bhia", q, k)
        attn = attn * attn.shape[1] ** -0.5
        attn = attn.softmax(dim=-1)
        x = torch.einsum(f"bhia,bhda->bhdi", attn, v)

        x = x.flatten(1, 2)
        x = self.conv_out(x)

        x = x.unflatten(2, shape[2:])
        x = x + x_in
        return x


## Networks
## ========
class UNet(nn.Module):
    def __init__(
            self,
            channels_in,
            channels_out=None,
            channels_list=(32, 64, 128, 256, 512),
            n_blocks_list=(2, 2, 2, 2, 2),
            min_channels_decoder=64,
            upscale_factor=None,
            n_groups=8,
        ):

        super().__init__()

        if channels_out is None:
            channels_out = channels_in

        ## Input block
        ## ===========
        in_block = nn.Conv2d(channels_in, channels_list[0], 3, padding=1)

        ## UNet levels
        ## ===========
        ## Bottleneck
        ## ----------
        i_level = len(channels_list) - 1
        ch = channels_list[i_level - 1]
        n_blocks = n_blocks_list[i_level]

        layers = []
        for _ in range(n_blocks):
            ch_in = ch
            ch = channels_list[i_level]
            layers += [
                nn.Conv2d(ch_in, ch, 3, padding=1),
                nn.GroupNorm(n_groups, ch),
                # nn.InstanceNorm2d(ch, affine=True),
                nn.LeakyReLU(0.1),
            ]
        main_block = nn.Sequential(*layers)
        last_ch = ch

        ## Higher levels (constructed botton-up)
        ## -------------------------------------
        for i_level in range(len(channels_list) - 2, -1, -1):
            layers = []

            ch = channels_list[max(i_level - 1, 0)]
            n_blocks = n_blocks_list[i_level]

            ## Encoder
            ## -------
            for _ in range(n_blocks):
                ch_in = ch
                ch = channels_list[i_level]
                layers += [
                    nn.Conv2d(ch_in, ch, 3, padding=1),
                    nn.GroupNorm(n_groups, ch),
                    # nn.InstanceNorm2d(ch, affine=True),
                    nn.LeakyReLU(0.1),
                ]

            layers += [
                ConcatShortcutBlock(nn.Sequential(
                    nn.MaxPool2d(2),
                    main_block,
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    # nn.ConvTranspose2d(ch, ch, 2, stride=2),
                )),
            ]
            ch += last_ch

            for _ in range(n_blocks):
                ch_in = ch
                ch = max(channels_list[i_level], min_channels_decoder)
                layers += [
                    nn.Conv2d(ch_in, ch, 3, padding=1),
                    nn.GroupNorm(n_groups, ch),
                    # nn.InstanceNorm2d(ch, affine=True),
                    nn.LeakyReLU(0.1),
                ]

            main_block = nn.Sequential(*layers)
            last_ch = ch

        ch = last_ch

        ## Output block
        ## ============
        layers = []
        if upscale_factor is not None:
            factors = (2,) * int(np.log2(upscale_factor))
            assert (np.prod(factors) == upscale_factor), 'Upscale factor must be a power of 2'
            for f in factors:
                layers += [
                    nn.Conv2d(ch, ch * f ** 2, 3, padding=1),
                    nn.PixelShuffle(f),
                ]
        layers += [
            nn.Conv2d(ch, channels_out, 1),
        ]
        out_block = nn.Sequential(*layers)

        self.net = ShortcutBlock(
            nn.Sequential(
                in_block,
                main_block,
                out_block,
            ),
            shortcut=nn.Upsample(scale_factor=upscale_factor, mode='nearest') if (upscale_factor is not None) else nn.Identity(),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class ResUNet(nn.Module):
    def __init__(
            self,
            in_channels=3,
            out_channels=None,
            channels_list=(128, 128, 256, 256, 512, 512),
            bottleneck_channels=512,
            downsample_list=(False, True, True, True, True, True),
            attn_list=(False, False, False, False, True, False),
            n_blocks=2,
            # min_channels_decoder=64,
            upscale_factor=None,
            n_groups=8,
            attn_heads=1,
        ):

        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        ch = in_channels

        ## Encoder
        ## =======
        self.encoder_blocks = nn.ModuleList([])
        ch_hidden_list = []

        layers = []
        ch_ = channels_list[0]
        layers.append(nn.Conv2d(ch, ch_, 3, padding=1))
        ch = ch_
        self.encoder_blocks.append(nn.Sequential(*layers))
        ch_hidden_list.append(ch)

        for i_level in range(len(channels_list)):
            ch_ = channels_list[i_level]
            downsample = downsample_list[i_level]
            attn = attn_list[i_level]

            if downsample:
                layers = []
                layers.append(nn.Conv2d(ch, ch, 3, padding=1, stride=2))
                self.encoder_blocks.append(nn.Sequential(*layers))
                ch_hidden_list.append(ch)

            for _ in range(n_blocks):
                layers = []
                layers.append(ResBlock(ch, ch_, n_groups=n_groups))
                ch = ch_
                if attn:
                    layers.append(Attention(ch, heads=attn_heads))
                self.encoder_blocks.append(nn.Sequential(*layers))
                ch_hidden_list.append(ch)

        ## Bottleneck
        ## ==========
        ch_ = bottleneck_channels
        layers = []
        layers.append(ResBlock(ch, ch_, n_groups=n_groups))
        ch = ch_
        layers.append(Attention(ch, heads=attn_heads))
        layers.append(ResBlock(ch, ch, n_groups=n_groups))
        self.bottleneck = nn.Sequential(*layers)

        ## Decoder
        ## =======
        self.decoder_blocks = nn.ModuleList([])
        for i_level in reversed(range(len(channels_list))):
            ch_ = channels_list[i_level]
            downsample = downsample_list[i_level]
            attn = attn_list[i_level]

            for _ in range(n_blocks):
                layers = []
                layers.append(ResBlock(ch + ch_hidden_list.pop(), ch_, n_groups=n_groups))
                ch = ch_
                if attn:
                    layers.append(Attention(ch, heads=attn_heads))
                self.decoder_blocks.append(nn.Sequential(*layers))

            if downsample:
                layers = []
                layers.append(ResBlock(ch + ch_hidden_list.pop(), ch, n_groups=n_groups))
                if attn:
                    layers.append(Attention(ch, heads=attn_heads))
                layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
                layers.append(nn.Conv2d(ch, ch, 3, padding=1))
                self.decoder_blocks.append(nn.Sequential(*layers))

        layers = []
        layers.append(ResBlock(ch + ch_hidden_list.pop(), ch, n_groups=n_groups))
        layers.append(nn.GroupNorm(n_groups, ch))
        layers.append(nn.SiLU())
        if upscale_factor is not None:
            factors = (2,) *  int(np.log2(self.upscale_factor))
            assert (np.prod(factors) == self.upscale_factor), 'Upscale factor must be a power of 2'
            for f in factors:
                layers.append(nn.Conv2d(ch, ch * f ** 2, kernel_size=3, padding=1))
                layers.append(nn.PixelShuffle(f))
        layers.append(zero_weights(nn.Conv2d(ch, out_channels, 1)))
        self.decoder_blocks.append(nn.Sequential(*layers))

    def forward(self, x):
        h = []
        for block in self.encoder_blocks:
            x = block(x)
            h.append(x)
        
        x = self.bottleneck(x)
        for block in self.decoder_blocks:
            x = torch.cat((x, h.pop()), dim=1)
            x = block(x)

        return x


class ResCNN(nn.Module):
    def __init__(
            self,
            channels_in,
            channels_out=None,
            channels_hidden=64,
            n_blocks=16,
            upscale_factor=None,
        ):

        super().__init__()

        self.upscale_factor = upscale_factor

        if channels_out is None:
            channels_out = channels_in

        ch = channels_in

        net = []

        ## Input block
        ## ===========
        net += [nn.Conv2d(ch, channels_hidden, 3, padding=1)]
        ch = channels_hidden

        ## Main block
        ## ==========
        block = []
        for _ in range(n_blocks):
            block += [
                ShortcutBlock(
                    nn.Sequential(
                        nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                    ),
                ),
            ]
        block += [
            nn.Conv2d(ch, ch, 3, padding=1),
        ]
        net += [ShortcutBlock(nn.Sequential(*block))]

        ## Output block
        ## ============
        if self.upscale_factor is not None:
            factors = (2,) *  int(np.log2(self.upscale_factor))
            assert (np.prod(factors) == self.upscale_factor), 'Upscale factor must be a power of 2'
            for f in factors:
                net += [
                    nn.Conv2d(ch, ch * f ** 2, kernel_size=3, padding=1),
                    nn.PixelShuffle(f),
                ]
        net += [
            nn.Conv2d(ch, channels_out, kernel_size=3, padding=1),
        ]
        self.net = nn.Sequential(*net)

    def forward(self, x):
        # x_in = x

        x = self.net(x)

        # if self.upscale_factor is not None:
        #     x_in = F.interpolate(x_in, scale_factor=self.upscale_factor, mode='nearest')
        # x = x_in + x

        return x


## Dataset Auxiliary
## =================
def find_data_folder(data_folders, folder_name):
    data_folder = None
    for data_folder_tmp in data_folders:
        data_folder_tmp = os.path.join(data_folder_tmp, folder_name)
        if os.path.isdir(data_folder_tmp):
            data_folder = data_folder_tmp
            break
    if data_folder is None:
        raise Exception('Could not fond the data in any of the provided folders')
    return data_folder

def split_dataset(dataset, split_size, rand=True):
    n_samples = len(dataset)
    if rand:
        indices = np.random.RandomState(42).permutation(n_samples)  # pylint: disable=no-member
    else:
        indices = np.arange(n_samples)
    indices1 = indices[:-split_size]
    indices2 = indices[-split_size:]
    dataset1 = torch.utils.data.Subset(dataset, indices1)
    dataset2 = torch.utils.data.Subset(dataset, indices2)
    return dataset1, dataset2


class ImageFilesDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, transform=None):
        super().__init__()

        if isinstance(filenames, str):
            filenames = [os.path.join(filenames, filename) for filename in np.sort(os.listdir(filenames))]
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        img = Image.open(filename).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img


class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, store_dataset=False, fixed_size_tensor=True, device=None, transform=None):
        super().__init__()

        self.dataset = dataset
        self.transform = transform

        if not store_dataset:
            self.stored = None
            self.is_stored = None
        elif fixed_size_tensor:
            x = self.dataset[0]
            self.stored = torch.zeros((len(self),) + x.shape, dtype=x.dtype, device=device)
            self.is_stored = torch.zeros((len(self),), dtype=torch.long)
        else:
            self.stored = [None] * len(self)
            self.is_stored = torch.zeros((len(self),), dtype=torch.long)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.stored is None:
            x = self.dataset[index]
        elif self.is_stored[index] == 0:
            x = self.dataset[index]
            self.stored[index] = x
            self.is_stored[index] = 1
        else:
            x = self.stored[index]

        if self.transform is not None:
            x = self.transform(x)
        return x


class PairsDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets, transform=None):
        super().__init__()

        self.datasets = datasets
        self.transform = transform

    def __len__(self):
        return min([len(dataset) for dataset in self.datasets])

    def __getitem__(self, index):
        x = tuple([dataset[index] for dataset in self.datasets])
        return x


class GetIndex(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index
    
    def forward(self, imgs):
        return imgs[self.index]

    def __repr__(self):
        return self.__class__.__name__ + f'(index={self.index})'


class TupleTransform(nn.Module):
    def __init__(self, *transform):
        super().__init__()
        self.transform = transform
    
    def forward(self, imgs):
        if len(self.transform) == 1:
            trans = self.transform[0]
            imgs = tuple([trans(img) for img in imgs])
        else:
            imgs = tuple([trans(img) for img, trans in zip(imgs, self.transform)])
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(' + ','.join([repr(trans) for trans in self.transform]) + ')'


def crop_scaled_pair(hr_img, lr_img, patch_size, method='rand'):
    hr_width, _ = hr_img.size
    lr_width, lr_height = lr_img.size

    scale = hr_width // lr_width
    lr_patch_size = patch_size // scale
    if method.lower() == 'rand':
        left = random.randrange(0, lr_width - lr_patch_size + 1)
        top = random.randrange(0, lr_height - lr_patch_size + 1)
    elif method.lower() == 'center':
        left = (lr_width - lr_patch_size) // 2
        top = (lr_height - lr_patch_size) // 2
    else:
        raise Exception(f'Unsuported method type: "{method}"')
    right = left + lr_patch_size
    bottom = top + lr_patch_size

    lr_patch = lr_img.crop((left, top, right, bottom))

    left *= scale
    top *= scale
    right *= scale
    bottom *= scale
    hr_patch = hr_img.crop((left, top, right, bottom))

    return hr_patch, lr_patch


class CropScaledPair(nn.Module):
    def __init__(self, patch_size=None, method='rand'):
        super().__init__()
        self.patch_size = patch_size
        self.method = method
    
    def forward(self, imgs):
        imgs = crop_scaled_pair(imgs[0], imgs[1], patch_size=self.patch_size, method=self.method)
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + f'(patch={self.patch_size}, method={self.method.lower()})'


## Datasets
## ========
class MNISTDataModule(object):
    shape = (1, 28, 28)
    # mean = None
    # std = None
    mean = 0.5
    std = 0.2

    def __init__(self, data_folders, n_valid=256, rand_valid=True, remove_labels=False, store_dataset=False, device='cpu'):  # pylint: disable=abstract-method
        super().__init__()


        data_folder = find_data_folder(data_folders, 'MNIST')
        print(f'Loading data from: {data_folder}')

        ## Base dataset
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            ])
        train_set = torchvision.datasets.MNIST(root=os.path.dirname(data_folder), train=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root=os.path.dirname(data_folder), train=False, transform=transform)

        ## Remove labels
        if remove_labels:
            train_set = DatasetWrapper(train_set, transform=GetIndex(0))
            test_set = DatasetWrapper(test_set, transform=GetIndex(0))
            ## Store dataset
            if store_dataset:
                train_set = DatasetWrapper(train_set, store_dataset=True, device=device)
                test_set = DatasetWrapper(test_set, store_dataset=True, device=device)

        ## Split
        if n_valid != 0:
            train_set, valid_set = split_dataset(train_set, n_valid, rand=rand_valid)
        else:
            valid_set = test_set

        ## set datasets
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set


class CelebAHQ256DataModule(object):
    mean = 0.5
    # std = 0.2
    std = 0.5

    def __init__(self, img_size, data_folders, store_dataset=False):
        super().__init__()

        self.img_size = img_size
        self.shape = (3, self.img_size, self.img_size)

        data_folder = find_data_folder(data_folders, 'CelebAMask-HQ-256')
        print(f'Loading data from: {data_folder}')

        ## Base dataset
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.img_size, interpolation=torchvision.transforms.InterpolationMode.BOX),
            torchvision.transforms.ToTensor(),
            ])

        ## Base dataset
        train_set = ImageFilesDataset(os.path.join(data_folder, 'train'), transform=transform)
        valid_set = ImageFilesDataset(os.path.join(data_folder, 'valid'), transform=transform)
        test_set = ImageFilesDataset(os.path.join(data_folder, 'test'), transform=transform)

        ## Store dataset
        if store_dataset:
            train_set = DatasetWrapper(train_set, store_dataset=True)
            valid_set = DatasetWrapper(valid_set, store_dataset=True)
            test_set = DatasetWrapper(test_set, store_dataset=True)

        ## set datasets
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set


class CelebASRFlowDataModule(object):
    mean = 0.5
    # std = 0.2
    std = 0.5

    def __init__(self, data_folders, scale=8, n_valid=256, rand_valid=True, store_dataset=False):
        super().__init__()

        self.img_size = 160
        self.shape = (3, self.img_size, self.img_size)

        data_folder = find_data_folder(data_folders, 'CelebA_SRFlow')
        print(f'Loading data from: {data_folder}')

        ## Base dataset
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            ])
        filenames = np.sort(os.listdir(os.path.join(data_folder, 'GT')))
        hr_filenames = [os.path.join(data_folder, 'GT', filename) for filename in filenames]
        lr_filenames = [os.path.join(data_folder, f'x{scale}', filename) for filename in filenames]
        train_set = PairsDataset(
            ImageFilesDataset(hr_filenames, transform=transform),
            ImageFilesDataset(lr_filenames, transform=transform)
        )

        ## Store dataset
        if store_dataset:
            train_set = DatasetWrapper(train_set, store_dataset=True, fixed_size_tensor=False)

        ## Split
        if n_valid != 0:
            train_set, valid_set = split_dataset(train_set, n_valid, rand=rand_valid)
        else:
            valid_set = train_set

        ## set datasets
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = valid_set


class FFHQDataModule(object):
    mean = 0.5
    std = 0.2

    def __init__(self, img_size, data_folders, n_valid=10000, rand_valid=False, store_dataset=False):
        super().__init__()

        self.img_size = img_size
        self.shape = (3, self.img_size, self.img_size)

        data_folder = find_data_folder(data_folders, 'ffhq')
        print(f'Loading data from: {data_folder}')

        ## Base dataset
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.img_size, interpolation=torchvision.transforms.InterpolationMode.BOX),
            torchvision.transforms.ToTensor(),
            ])
        data_folder = os.path.join(data_folder, 'images1024x1024')
        filenames = []
        for dir_name in np.sort(os.listdir(data_folder)):
            data_folder_ = os.path.join(data_folder, dir_name)
            if os.path.isdir(data_folder_):
                filenames += [os.path.join(data_folder_, filename) for filename in np.sort(os.listdir(data_folder_))]
        train_set = ImageFilesDataset(filenames, transform=transform)

        ## Store dataset
        if store_dataset:
            train_set = DatasetWrapper(train_set, store_dataset=True)

        ## Split
        if n_valid != 0:
            train_set, valid_set = split_dataset(train_set, n_valid, rand=rand_valid)
        else:
            valid_set = train_set

        ## set datasets
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = valid_set


class DIV2KDataModule(object):
    mean = 0.5
    # std = 0.2
    std = 0.5

    def __init__(self, img_size, data_folders, n_valid=100, rand_valid=False, store_dataset=False):
        super().__init__()

        self.img_size = img_size
        self.shape = (3, self.img_size, self.img_size)

        data_folder = find_data_folder(data_folders, 'DIV2K')
        print(f'Loading data from: {data_folder}')

        ## Base dataset
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            ])
        train_set = ImageFilesDataset(os.path.join(data_folder, 'DIV2K_train_HR'), transform=transform)

        ## Store dataset
        if store_dataset:
            train_set = DatasetWrapper(train_set, store_dataset=True, fixed_size_tensor=False)

        ## Split
        if n_valid != 0:
            train_set, valid_set = split_dataset(train_set, n_valid, rand=rand_valid)
        else:
            valid_set = train_set

        ## Crop
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop((self.img_size, self.img_size)),
            ])
        train_set = DatasetWrapper(train_set, transform=train_transform)

        valid_transform = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop((self.img_size, self.img_size)),
            ])
        valid_set = DatasetWrapper(valid_set, transform=valid_transform)

        ## Set datasets
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = valid_set


class DIV2KSRDataModule(object):
    mean = 0.5
    # std = 0.2
    std = 0.5

    def __init__(self, img_size, data_folders, scale=4, n_valid=100, rand_valid=False, store_dataset=False):
        super().__init__()

        self.img_size = img_size
        self.shape = (3, self.img_size, self.img_size)

        data_folder = find_data_folder(data_folders, 'DIV2K')
        print(f'Loading data from: {data_folder}')

        ## Base dataset
        filenames = np.sort(os.listdir(os.path.join(data_folder, 'DIV2K_train_HR')))
        hr_filenames = [os.path.join(data_folder, 'DIV2K_train_HR', filename) for filename in filenames]
        lr_filenames = [os.path.join(data_folder, 'DIV2K_train_LR_bicubic' ,f'X{scale}', ('{0}' + f'x{scale}' + '{1}').format(*os.path.splitext(filename))) for filename in filenames]
        train_set = PairsDataset(
            ImageFilesDataset(hr_filenames),
            ImageFilesDataset(lr_filenames),
        )

        ## Store dataset
        if store_dataset:
            train_set = DatasetWrapper(train_set, store_dataset=True, fixed_size_tensor=False)

        ## Split
        if n_valid != 0:
            train_set, valid_set = split_dataset(train_set, n_valid, rand=rand_valid)
        else:
            valid_set = train_set

        ## Crop
        train_transform = torchvision.transforms.Compose([
            CropScaledPair(self.img_size, method='rand'),
            TupleTransform(torchvision.transforms.ToTensor()),
            ])
        train_set = DatasetWrapper(train_set, transform=train_transform)

        valid_transform = torchvision.transforms.Compose([
            CropScaledPair(self.img_size, method='center'),
            TupleTransform(torchvision.transforms.ToTensor()),
            ])
        valid_set = DatasetWrapper(valid_set, transform=valid_transform)

        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = valid_set


class ImageNetDataModule(object):
    mean = 0.5  # [0.485, 0.456, 0.406]
    std = 0.2  # [0.229, 0.224, 0.225]
    n_classes = 1000

    def __init__(self, data_folders, img_size=224, n_valid=0, rand_valid=False, store_dataset=False):
        super().__init__()

        self.img_size = img_size
        self.shape = (3, self.img_size, self.img_size)

        data_folder = find_data_folder(data_folders, 'imagenet')
        print(f'Loading data from: {data_folder}')

        # with open(os.path.join(data_folder, 'imagenet1000_clsidx_to_labels.txt'), 'r') as fid:
        #     data = yaml.load(fid, Loader=yaml.SafeLoader)
        # self.class_names = [data[i].split(',')[0] for i in range(len(data))]

        ## Base dataset
        train_set = torchvision.datasets.ImageFolder(os.path.join(data_folder, 'train'))
        test_set = torchvision.datasets.ImageFolder(os.path.join(data_folder, 'val'))

        # ## Store dataset
        # if store_dataset:
        #     train_set = DatasetWrapper(train_set, store_dataset=True, fixed_size_tensor=False)
        #     test_set = DatasetWrapper(test_set, store_dataset=True, fixed_size_tensor=False)

        ## Split
        if n_valid != 0:
            train_set, valid_set = split_dataset(train_set, n_valid, rand=rand_valid)
        else:
            valid_set = test_set

        ## Transforms
        train_transform = torchvision.transforms.Compose([
            GetIndex(0),
            torchvision.transforms.RandomResizedCrop(self.img_size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            ])
        train_set = DatasetWrapper(train_set, transform=train_transform)

        valid_transform = torchvision.transforms.Compose([
            GetIndex(0),
            torchvision.transforms.Resize(int(self.img_size * 256 / 224)),
            torchvision.transforms.CenterCrop(self.img_size),
            torchvision.transforms.ToTensor(),
            ])
        valid_set = DatasetWrapper(valid_set, transform=valid_transform)
        test_set = DatasetWrapper(test_set, transform=valid_transform)

        ## Store dataset
        if store_dataset:
            train_set = DatasetWrapper(train_set, store_dataset=True)
            valid_set = DatasetWrapper(test_set, store_dataset=True)

        ## Set datasets
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set


## Net wrapper
## ===========
class NetWrapper(object):  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    def __init__(
            self,
            net,
            optimizer_type='adam',
            optimizer_params=None,
            lr_lambda=None,
            ema_alpha=None,
            ema_update_every=1,
            use_ema_for_best=None,
            device='cpu',
            ddp_active=False,
        ):

        self._device = device
        self._step = 0

        self._requires_grad = True
        self.net = net
        self.net.to(self.device)
        self.net.eval()

        ## Set field to store best network
        ## -------------------------------
        if use_ema_for_best is None:
            use_ema_for_best = ema_alpha is not None
        self._use_ema_for_best = use_ema_for_best
        self._net_best = None
        self._score_best = None
        self._step_best = None

        ## Setup Exponantial moving average (EMA)
        ## --------------------------------------
        self._ema_alpha = ema_alpha
        self._ema_update_every = ema_update_every
        if self._ema_alpha is not None:
            self._net_ema = self._make_copy(self.net)
        else:
            self._net_ema = None

        ## Initialize DDP
        ## --------------
        if ddp_active:
            self._net_ddp = DDP(self.net, device_ids=[self.device])
        else:
            self._net_ddp = self.net

        ## Set optimizer
        ## -------------
        if optimizer_type.lower() == 'adam':
            if ('weight_decay' in optimizer_params) and (optimizer_params['weight_decay'] > 0.):
                self.optimizer = torch.optim.AdamW(self._net_ddp.parameters(), **optimizer_params)
            else:
                self.optimizer = torch.optim.Adam(self._net_ddp.parameters(), **optimizer_params)
        else:
            raise Exception(f'Unsuported optimizer type: "{optimizer_type}"')

        ## Set schedualer
        ## --------------
        if lr_lambda is None:
            lr_lambda = lambda step: 1
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    @property
    def device(self):
        return self._device

    @property
    def step(self):
        return self._step

    def get_net(self, use_ddp=False, use_ema=False, use_best=False):
        if use_ddp:
            net = self._net_ddp
        elif use_ema:
            if self._net_ema is not None:
                net = self._net_ema
            else:
                net = self.net
        elif use_best:
            if self._net_best is not None:
                net = self._net_best
            else:
                if self._use_ema_for_best:
                    net = self._net_ema
                else:
                    net = self.net
        else:
            net = self.net
        return net

    def __call__(self, *args, use_ddp=False, use_ema=False, use_best=False, **kwargs):
        return self.get_net(use_ddp=use_ddp, use_ema=use_ema, use_best=use_best)(*args, **kwargs)

    @property
    def lr(self):
        return self.optimizer.param_groups[0]['lr']

    def train(self):
        self._net_ddp.train()

    def eval(self):
        self._net_ddp.eval()

    @property
    def requires_grad_(self):
        return self._requires_grad

    @requires_grad_.setter
    def requires_grad_(self, requires_grad):
        self.net.requires_grad_(requires_grad)
        self._requires_grad = requires_grad

    def increment(self):
        self._step += 1
        self.scheduler.step()

        ## EMA step
        if (self._net_ema is not None) and (self.step % self._ema_update_every == 0):
            alpha = max(self._ema_alpha, 1 / (self.step // self._ema_update_every))
            for p, p_ema in zip(self.net.parameters(), self._net_ema.parameters()):
                p_ema.data.mul_(1 - alpha).add_(p.data, alpha=alpha)

    def clip_grad_norm(self, max_norm):
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._net_ddp.parameters(), max_norm=max_norm)

    @contextmanager
    def set_ddp_sync(self, active=True):
        if isinstance(self._net_ddp, torch.nn.parallel.distributed.DistributedDataParallel):
            old_val = self._net_ddp.require_backward_grad_sync
            self.net.require_backward_grad_sync = active  # pylint: disable=attribute-defined-outside-init
            try:
                yield
            finally:
                self.net.require_backward_grad_sync = old_val  # pylint: disable=attribute-defined-outside-init
        else:
            try:
                yield
            finally:
                pass

    def update_best(self, score):
        if score is not None:
            if (self._score_best is None) or (score <= self._score_best):
                if self._use_ema_for_best and (self._net_ema is not None):
                    self._net_best = self._make_copy(self._net_ema)
                else:
                    self._net_best = self._make_copy(self.net)
                self._score_best = score
                self._step_best = self._step

    @staticmethod
    def _make_copy(network):
        network = copy.deepcopy(network)
        network.requires_grad_(False)
        network.eval()
        return network

    ## Save and load
    ## -------------
    def state_dict(self):
        state_dict = dict(
            step = self._step,
            net = self.net.state_dict(),
            optimizer = self.optimizer.state_dict(),
            scheduler = self.scheduler.state_dict(),
            net_ema = self._net_ema.state_dict() if (self._net_ema is not None) else None,
            net_best = self._net_best.state_dict() if (self._net_best is not None) else None,
            score_best = self._score_best,
            step_best = self._step_best,
        )
        return state_dict

    def load_state_dict(self, state_dict):
        self._step = state_dict['step']
        self.net.load_state_dict(state_dict['net'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
        if (self._net_ema is not None) and (state_dict['net_ema'] is not None):
            self._net_ema.load_state_dict(state_dict['net_ema'])
        if state_dict['net_best'] is None:
            self._net_best = None
        else:
            self._net_best = self._make_copy(self.net)
            self._net_best.load_state_dict(state_dict['net_best'])
        self._score_best = state_dict['score_best']
        self._step_best = state_dict['step_best']


## Auxiliary functions
## ===================
def override_params(params, params_override):
    for key in params_override:
        found = False
        for field_params in params.values():
            if key in field_params:
                field_params[key] = params_override[key]
                found = True
                break
        if not found:
            raise Exception(f'Could not override key. "{key}" was not found in params.')

def update_params_from_cli(params):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for field_key, field_params in params.items():
        parser_group = parser.add_argument_group(field_key)
        for key, val in field_params.items():
            if isinstance(val, bool):
                if val:
                    parser_group.add_argument(f'--disable_{key}', dest=key, action='store_false')
                else:
                    parser_group.add_argument(f'--{key}', action='store_true')
            else:
                parser_group.add_argument(f'--{key}', default=val, type=type(val) if (val is not None) else str, help=f'(default: {val})')

    args = vars(parser.parse_args())
    for field_params in params.values():
        for key in field_params:
            field_params[key] = args[key]

def set_random_seed(random_seed=0):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)


class EncapsulatedRandomState:
    def __init__(self, random_seed=0):
        self._random_seed = random_seed
        self._random_state = None
        self._np_random_state = None
        self._torch_random_state = None
        self._torch_cuda_random_state = None

    def __enter__(self):
        self._random_state = random.getstate()
        self._np_random_state = np.random.get_state()
        self._torch_random_state = torch.random.get_rng_state()
        self._torch_cuda_random_state = {i: torch.cuda.get_rng_state(i)  for i in range(torch.cuda.device_count())}

        random.seed(self._random_seed)
        np.random.seed(self._random_seed)
        torch.random.manual_seed(self._random_seed)
        torch.cuda.manual_seed_all(self._random_seed)

    def __exit__(self, type_, value, traceback):
        random.setstate(self._random_state)
        np.random.set_state(self._np_random_state)
        torch.random.set_rng_state(self._torch_random_state)
        for i, random_state in self._torch_cuda_random_state.items():
            torch.cuda.set_rng_state(random_state, i)


class Timer(object):
    def __init__(self, interval, reset=True):
        self._interval = interval
        if self._interval is None:
            self._end_time = None
        else:
            self._end_time = time.time()

        if reset:
            self.reset()

    def __bool__(self):
        if self._end_time is None:
            return False
        else:
            return time.time() > self._end_time

    def __str__(self):
        if self._end_time is None:
            return '----'
        else:
            timedelta = int(self._end_time - time.time())
            if timedelta >= 0:
                return str(datetime.timedelta(seconds=timedelta))
            else:
                return '-' + str(datetime.timedelta(seconds=-timedelta))

    def reset(self, interval=None):
        if interval is not None:
            self._interval = interval

        if self._interval is None:
            self._end_time = None
        else:
            self._end_time = time.time() + self._interval

def loop_loader(data_loader, max_steps=None, max_epochs=None):
    step = 0
    epoch = 0
    while True:
        if isinstance(data_loader.sampler, torch.utils.data.distributed.DistributedSampler):
            data_loader.sampler.set_epoch(epoch)

        for x in data_loader:
            if (max_steps is not None) and (step >= max_steps):
                return None
            yield x
            step += 1

        if (max_epochs is not None) and (epoch >= max_epochs):
            return None
        epoch += 1

## Visualizations
## ==============
def imgs_to_grid(imgs, **make_grid_args):
    imgs = imgs.detach().cpu()
    if imgs.ndim == 5:
        nrow = imgs.shape[1]
        imgs = imgs.view(imgs.shape[0] * imgs.shape[1], imgs.shape[2], imgs.shape[3], imgs.shape[4])
    else:
        nrow = int(np.ceil(imgs.shape[0] ** 0.5))

    make_grid_args2 = dict(scale_range=(0, 1), pad_value=1.)
    make_grid_args2.update(make_grid_args)
    img = torchvision.utils.make_grid(imgs, nrow=nrow, **make_grid_args2).clamp(0, 1)
    return img

def color_img(x, max_val=1., min_val=None, colorscale='RdBu'):
    if min_val is None:
        min_val = -max_val
    x = (x - min_val) / (max_val - min_val)
    x.clamp_(0., 1.)

    x_shape_out = x.shape[:-3] + x.shape[-2:]
    x = x.flatten()

    x = torch.tensor(plotly.colors.sample_colorscale(colorscale, x.cpu().numpy(), colortype='tuple'))
    x = x.unflatten(0, x_shape_out).movedim(-1, -3).contiguous()
    return x


class Imshow:
    def __init__(self, img, scale=1, **kwargs):
        img = self._preproc(img)
        self.fig = px.imshow(img, **kwargs).update_layout(  # , color_continuous_scale='gray')
            height=img.shape[0] * scale,
            width=img.shape[1] * scale,
            margin=dict(t=0, b=0, l=0, r=0),
            xaxis_showticklabels=False,
            yaxis_showticklabels=False,
            # coloraxis_showscale=False
        )

    def _preproc(self, img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy().transpose((1, 2, 0))
        img = img.clip(0, 1)
        return img

    def update(self, img):
        img = self._preproc(img)
        self.fig.data[0].update(source=None, z=img * 255)

    def show(self):
        self.fig.show()

    def get_widget(self):
        self.fig = go.FigureWidget(self.fig)
        return self.fig

def img_to_html(img):
    if img is None:
        return ''
    buffered = BytesIO()
    img_np = img.detach().cpu().numpy().transpose((1, 2, 0))
    img_np = (img_np * 255).astype('uint8')
    Image.fromarray(img_np, 'RGB').save(buffered, format='PNG')
    buffered.seek(0)
    data = base64.b64encode(buffered.getvalue()).decode('ascii')
    html = f'<img src="data:image/png;base64, {data}" alt="Image" />'
    return html

def fig_to_html(fig, id_=None, style='display:inline-block;'):
    if isinstance(fig, (list, tuple)):
        return '\n'.join([fig_to_html(x) for x in fig])
    else:
        if fig is None:
            return ''
        if id_ is None:
            id_ = str(uuid.uuid4())
        html =  f'<div id="fig_{id_}" style="{style}"></div>\n'
        html +=  '<script>\n'
        html +=  '  {\n'
        html += f'    let figure = {plotly.io.to_json(fig)};\n'
        html += f'    Plotly.newPlot(document.getElementById("fig_{id_}"), figure.data, figure.layout, {{responsive: true}});\n'
        html +=  '  }\n'
        html +=  '</script>'
        return html


## Experiment folder manager
## =========================
class ExpFolderManager:
    def __init__(
            self,
            folder,
            n_points_to_save=0,
            checkpoints_folder=None,
        ):

        self.folder = folder
        self.n_points_to_save = n_points_to_save

        self.exp_filename = os.path.join(self.folder, 'exp.py')
        if checkpoints_folder is None:
            checkpoints_folder = self.folder
        self.checkpoint_filename = os.path.join(checkpoints_folder, 'checkpoint.pt')
        self.intermediate_checkpoint_folder = os.path.join(self.folder, 'intermediate')

        self.intermediate_checkpoint_filename = None
        self._intermediate_point_num = 0

        print(f'Experiment folder: {self.folder}')

    def reset(self):
        if os.path.isdir(self.folder):
            shutil.rmtree(self.folder)

    def create(self, reset=False):
        if reset:
            self.reset()
        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)

    def save_exp_file(self, org_exp_filename=None):
        if org_exp_filename is None:
            org_exp_filename = __file__
        if org_exp_filename != self.exp_filename:
            shutil.copyfile(org_exp_filename, self.exp_filename)

    def get_exp(self):
        spec = importlib.util.spec_from_file_location('exp', self.exp_filename)
        exp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exp)
        return exp

    @classmethod
    def from_wandb(cls, run_id, folder, *args, point_num=None, **kwargs):
        wandb.login()
        wandb_run = wandb.Api().run(run_id)
        folder = folder.format(model_name=wandb_run.name, project_name=wandb_run.project)

        folder_manager = cls(folder, *args, **kwargs)
        folder_manager.create()

        ## Download exp.py file
        if not os.path.exists(folder_manager.exp_filename):
            wandb_run.file('exp.py').download(folder_manager.folder, replace=True)

        ## Download checkpoint
        if point_num is None:
            wandb_file = wandb_run.file('checkpoint.pt')
            if wandb_file.size > 0:
                wandb_file.download(os.path.dirname(folder_manager.checkpoint_filename), replace=True)
        else:
            remote_checkpoint_filename = folder_manager.get_intermediat_filename(point_num)
            wandb_file = wandb_run.file(remote_checkpoint_filename)
            if wandb_file.size > 0:
                wandb_file.download('/tmp/', replace=True)
                os.rename(os.path.join('/tmp', remote_checkpoint_filename), folder_manager.checkpoint_filename)

        return folder_manager, wandb_run

    def get_intermediat_filename(self, point_num=None):
        if point_num is None:
            point_num = self._intermediate_point_num
        return f'inter_checkpoint_{point_num:03d}.pt'

    def save_checkpoint(self, state_dict):
        ## Copy existing file as an intermediate checkpoint
        ## ------------------------------------------------
        if os.path.isfile(self.checkpoint_filename) and self.n_points_to_save > 0:
            if not os.path.isdir(self.intermediate_checkpoint_folder):
                os.makedirs(self.intermediate_checkpoint_folder)

            self._intermediate_point_num = (self._intermediate_point_num + 1) % self.n_points_to_save
            self.intermediate_checkpoint_filename = os.path.join(self.intermediate_checkpoint_folder, self.get_intermediat_filename())
            os.replace(self.checkpoint_filename, self.intermediate_checkpoint_filename)

        ## Save checkpoint
        ## ---------------
        if not os.path.isdir(os.path.dirname(self.checkpoint_filename)):
            os.makedirs(os.path.dirname(self.checkpoint_filename))
        torch.save(state_dict, self.checkpoint_filename)

    def write_file(self, filename, text):
        filename = os.path.join(self.folder, filename)
        with open(filename, 'w', encoding="utf-8") as fid:
            fid.write(text)

    def serve_folder(self, initial_port=8000, n_attempts=100):
        ## Find a free port
        port = initial_port
        handler = partial(QuietHTTPRequestHandler, directory=os.path.abspath(self.folder))
        while port < initial_port + n_attempts:
            try:
                httpd = HTTPServer(('', port), handler , False)
                httpd.server_bind()
                break
            except OSError:
                port += 1
        ## Sratr server
        print(f'Serving folder on http://{httpd.server_name}:{httpd.server_port}')
        httpd.server_activate()
        def serve_forever(httpd):
            with httpd:
                httpd.serve_forever()

        thread = threading.Thread(target=serve_forever, args=(httpd,))
        thread.setDaemon(True)
        thread.start()

class QuietHTTPRequestHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):  # pylint: disable=redefined-builtin
        pass

## Weights and Biases auxiliary
## ============================
def init_wandb_logger(
        run_name=None,
        entity=None,
        project=None,
        storage_folder='/tmp/',
        run_id=None,
        params=None,
    ):

    wandb.login()
    if run_id is not None:
        wandb_logger = wandb.init(
            entity=entity,
            project=project,
            id=run_id,
            resume='must',
            force=True,
            dir=storage_folder,
        )
    else:
        wandb_logger = wandb.init(
            entity=entity,
            project=project,
            name=run_name,
            resume='never',
            save_code=True,
            force=True,
            dir=storage_folder,
            config=params,
        )
        # define default x-axis (for latest wandb versions)
        if getattr(wandb_logger, "define_metric", None):
            wandb_logger.define_metric("train_params/model_step")
            wandb_logger.define_metric("*", step_metric="train_params/model_step", step_sync=True)

    print(f'W&B run name and id: {wandb_logger.name} / {wandb_logger.id}')
    return wandb_logger

def wandb_log_exp_file(wandb_logger, exp_filename=None):
    if exp_filename is None:
        exp_filename = __file__
    wandb_logger.save(exp_filename, base_path=os.path.dirname(exp_filename))
    print(f'{exp_filename} was saved to Weights & Biases')


## DDP Manager
## ===========
class DDPManager:
    def __init__(self, store_port=56895):
        self._store_port = store_port

        self.is_active = distrib.is_available() and distrib.is_initialized()
        if self.is_active:
            self.rank = distrib.get_rank()
            self.is_main = self.rank == 0
            self.size = distrib.get_world_size()
        else:
            self.rank = 0
            self.is_main = True
            self.size = 1
        self.store = None

    def broadcast(self, x):
        if self.is_active:
            if self.store is None:
                self.store = distrib.TCPStore("127.0.0.1", self._store_port, self.size, self.is_main)

            if self.is_main:
                self.store.set('broadcasted_var', x)
                distrib.barrier()
            else:
                self.store.wait(['broadcasted_var'], datetime.timedelta(minutes=5))
                x = self.store.get('broadcasted_var')
                distrib.barrier()
        return x

    def gather(self, x):
        if self.is_active:
            res = [x.clone() for _ in range(self.size)]
            distrib.gather(x, res if self.is_main else None)
        else:
            res = [x]
        return res

    def convert_model(self, model, device_ids):
        if self.is_active:
            model = DDP(model, device_ids=device_ids)
        return model


## CLI
## ===
def cli():
    params = copy.deepcopy(DEFAULT_PARAMS)
    override_params(params, PARAMS_OVERRIDE)
    update_params_from_cli(params)

    ## Start a new experiment or contintue an existing one
    ## ---------------------------------------------------
    params['train']['wandb_run_id'] = None
    if params['general']['load_wandb_id'] is not None:
        folder_manager, wandb_run = ExpFolderManager.from_wandb(
            params['general']['load_wandb_id'],
            folder=params['general']['wandb_exp_folder'],
            checkpoints_folder='/tmp/checkpoints/',
            # checkpoints_folder=params['general']['checkpoints_folder'],
            )
        params['train']['wandb_run_id'] = wandb_run.id
    elif params['general']['load_folder'] is not None:
        folder_manager = ExpFolderManager(
            params['general']['load_folder'],
            checkpoints_folder=params['general']['checkpoints_folder'],
            )
    else:
        folder_manager = None

    print('Running ...')
    devices_list = params['general']['device'].split(',')
    print(f'Hostname: {socket.gethostname()}-{",".join(devices_list)}')
    print(f'Process ID: {os.getgid()}')

    if len(devices_list) == 1:
        device = devices_list[0]
        train(params, folder_manager, device=device)
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = params['general']['ddp_port']
        os.environ['NCCL_IB_DISABLE'] = '1'
        world_size = len(devices_list)

        mp.spawn(
            train_ddp,
            args=(world_size, devices_list, params, folder_manager),
            nprocs=world_size,
            join=True,
            )

def train_ddp(rank, world_size, devices_list, *args):
    distrib.init_process_group('nccl', rank=rank, world_size=world_size)
    device = devices_list[rank]
    train(*args, device=device)
    distrib.destroy_process_group()

def train(params, folder_manager, device):
    if folder_manager is not None:
        exp = folder_manager.get_exp()
        exp_folder = folder_manager.folder
        Model_ = exp.Model
        Trainer_ = exp.Trainer
    else:
        exp_folder = params['general']['exp_folder']
        Model_ = Model
        Trainer_ = Trainer
        params['train']['restart_logging'] = True
    params['train']['exp_folder'] = exp_folder
    params['train']['checkpoints_folder'] = params['general']['checkpoints_folder']

    if (folder_manager is not None) and os.path.isfile(folder_manager.checkpoint_filename):
        model = Model_.load(folder_manager.checkpoint_filename, device=device)
    else:
        model = Model_(device=device, **params['model'])

    ## Train
    ## -----
    trainer = Trainer_(model=model, **params['train'])
    trainer.train()

## Call CLI
## ========
if __name__ == '__main__':
    cli()
