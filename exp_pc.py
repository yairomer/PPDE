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
    # restoration_model_folder = os.path.join(os.environ['HOME'], 'workspace/tmp/exp/nppc_ebm/restore/best/mnist_in_painting'),
    # net_type = 'unet',
    # batch_size = 128,
    # lr = 1e-4,
    # n_epochs = 100,
    # log_every = 200,
    # benchmark_every = 1000,

    # ## MNIST - Denoising
    # ## -----------------
    # restoration_model_folder = os.path.join(os.environ['HOME'], 'workspace/tmp/exp/nppc_ebm/restore/best/mnist_denoising1'),
    # net_type = 'unet',
    # batch_size = 128,
    # lr = 1e-4,
    # n_epochs = 1000,
    # log_every = 200,
    # benchmark_every = 1000,

    # ## CelebA 256x256 - In-painting eyes
    # ## ---------------------------------
    # restoration_model_folder = os.path.join(os.environ['HOME'], 'workspace/tmp/exp/nppc_ebm/restore/best/celeba_hq_256_in_painting_eyes'),
    # net_type = 'res_unet_256',
    # batch_size = 16,
    # max_batch_chunk_size = 8,
    # lr = 3e-5,
    # n_epochs = 20,
    # log_every = 20,
    # benchmark_every = 500,

    # ## CelebA 256x256 - In-painting mouth
    # ## ----------------------------------
    # restoration_model_folder = os.path.join(os.environ['HOME'], 'workspace/tmp/exp/nppc_ebm/restore/best/celeba_hq_256_in_painting_mouth'),
    # net_type = 'res_unet_256',
    # batch_size = 16,
    # max_batch_chunk_size = 8,
    # lr = 3e-5,
    # n_epochs = 20,
    # log_every = 20,
    # benchmark_every = 500,

    # ## CelebA 256x256 - Colorization
    # ## -----------------------------
    # restoration_model_folder = os.path.join(os.environ['HOME'], 'workspace/tmp/exp/nppc_ebm/restore/best/celeba_hq_256_colorization'),
    # net_type = 'res_unet_256',
    # batch_size = 16,
    # max_batch_chunk_size = 8,
    # lr = 3e-5,
    # n_epochs = 30,
    # log_every = 20,
    # benchmark_every = 500,

    # ## FFHQ - Colorization
    # ## -------------------
    # restoration_model_folder = os.path.join(os.environ['HOME'], 'workspace/tmp/exp/nppc_ebm/restore/best/ffhq_colorization'),
    # net_type = 'res_unet_256',
    # batch_size = 16,
    # max_batch_chunk_size = 8,
    # lr = 3e-5,
    # n_epochs = 30,
    # log_every = 20,
    # benchmark_every = 500,

    ## CelebA 256x256 - Super resolution
    ## ---------------------------------
    restoration_model_folder = os.path.join(os.environ['HOME'], 'workspace/tmp/exp/nppc_ebm/restore/best/celeba_super_resolution'),
    pre_net_type = 'pre_res_cnn',
    # net_type = 'unet2',
    # batch_size = 16,
    # lr = 1e-3,
    net_type = 'res_unet_256',
    batch_size = 16,
    max_batch_chunk_size = 16,
    lr = 1e-5,
    n_epochs = 30,
    log_every = 20,
    benchmark_every = 500,

    # ## DIV2K - Super resolution
    # ## ------------------------
    # restoration_model_folder = os.path.join(os.environ['HOME'], 'workspace/tmp/exp/nppc_ebm/restore/best/div2k_super_resolution'),
    # pre_net_type = 'pre_res_cnn',
    # net_type = 'unet2',
    # batch_size = 16,
    # lr = 1e-5,
    # second_moment_loss_lambda = 1e-1,
    # n_epochs = 8000,
    # log_every = 20,
    # benchmark_every = 200,

    # ## DIV2K - Colorization
    # ## --------------------
    # restoration_model_folder = os.path.join(os.environ['HOME'], 'workspace/tmp/exp/nppc_ebm/restore/best/div2k_colorization'),
    # net_type = 'res_unet_128',
    # batch_size = 32,
    # max_batch_chunk_size = 32,
    # lr = 3e-5,
    # n_epochs = 2500,
    # log_every = 20,
    # benchmark_every = 200,

    # ## ImageNet - Colorization
    # ## -----------------------
    # restoration_model_folder = os.path.join(os.environ['HOME'], 'workspace/tmp/exp/nppc_ebm/restore/best/imagenet_colorization'),
    # net_type = 'res_unet_128',
    # batch_size = 32,
    # max_batch_chunk_size = 32,
    # lr = 1e-5,
    # second_moment_loss_lambda = 1e-1,
    # n_epochs = 10,
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
        project_name = 'nppc_ebm/pc',
        name = 'exp_{now}_pc_{dataset}_{distortion_type}',
        restoration_model_folder = None,
        data_folders = DEFAULT_FOLDERS['data_folders'],
        img_size = None,
        store_dataset = False,

        n_dirs = 5,
        pre_net_type = 'none',
        net_type = 'unet',

        lr = 1e-3,
        second_moment_loss_lambda = 1e0,
        second_moment_loss_grace = 1000,
        weight_decay = 0.,

        ema_step = 1,
        ema_alpha = None,

        model_random_seed = 42,
    ),

    ## Training parameters
    ## -------------------
    train = dict(
        batch_size = 128,
        max_batch_chunk_size = None,
        n_epochs = 100,
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

        self.n_dirs = self.params['n_dirs']

        ## Initializing random state
        ## -------------------------
        set_random_seed(self.params['model_random_seed'])

        ## Load restoration model
        ## ----------------------
        restoration_model_folder = self.params['restoration_model_folder']
        folder_manager = ExpFolderManager(restoration_model_folder)
        restoration_model_exp = folder_manager.get_exp()
        self.restoration_model = restoration_model_exp.Model.load(folder_manager.checkpoint_filename, device=self.device, store_dataset=store_dataset)
        for net in self.restoration_model.networks.values():
            net.requires_grad_ = False

        self.data_module = self.restoration_model.data_module
        self.x_shape = self.restoration_model.x_shape

        ## Run name
        ## --------
        name = self.params['name']
        if name is None:
            self.name = 'exp_{now}'
        now = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
        params_ = self.restoration_model.params.copy()
        params_.update(self.params)
        self.name = name.format(now=now, **params_)

        ## Set parametric model
        ## --------------------
        pre_pad_base_size = None
        if self.params['pre_net_type'] == 'none':
            if self.restoration_model.upscale_factor is None:
                pre_net = None
            else:
                pre_net = nn.Upsample(scale_factor=self.restoration_model.upscale_factor, mode='nearest')
            n_pre_feat = self.restoration_model.x_distorted_shape[0]

        # elif self.params['pre_net_type'] == 'pre_unet1':
        #     n_pre_feat = 64
        #     pre_net = UNet(
        #         channels_in=self.restoration_model.x_distorted_shape[0],
        #         channels_out=n_pre_feat,
        #         channels_list=(32, 64, 128, 256, 512),
        #         n_blocks_list=(1, 1, 1, 1, 1),
        #         min_channels_decoder=64,
        #         upscale_factor=self.restoration_model.upscale_factor,
        #     )
        #     pre_pad_base_size = 2 ** 4

        elif self.params['pre_net_type'] == 'pre_res_cnn':
            n_pre_feat = 64
            pre_net = ResCNN(
                channels_in=self.restoration_model.x_distorted_shape[0],
                channels_out=n_pre_feat,
                channels_hidden=64,
                n_blocks=16,
                upscale_factor=self.restoration_model.upscale_factor,
            )

        else:
            raise Exception(f'Unsupported net_type: "{self.params["pre_net_type"]}"')

        pad_base_size = None
        if self.params['net_type'] == 'unet':
            base_net = UNet(
                channels_in=n_pre_feat + self.x_shape[0],
                channels_out=self.x_shape[0] * self.params['n_dirs'],
                channels_list=(32, 64, 128, 256),
                n_blocks_list=(1, 1, 1, 2),
                min_channels_decoder=64,
            )
            pad_base_size = 2 ** 3

        elif self.params['net_type'] == 'unet2':
            base_net = UNet(
                channels_in=n_pre_feat + self.x_shape[0],
                channels_out=self.x_shape[0] * self.params['n_dirs'],
                channels_list=(32, 64, 128, 256, 512),
                n_blocks_list=(2, 2, 2, 2, 2),
                min_channels_decoder=64,
            )
            pad_base_size = 2 ** 4

        elif self.params['net_type'] == 'res_unet_128':
            ## DDPM
            base_net = ResUNet(
                in_channels=n_pre_feat + self.x_shape[0],
                out_channels=self.x_shape[0] * self.params['n_dirs'],
                channels_list=(64, 128, 128, 256, 256),
                bottleneck_channels=512,
                downsample_list=(False, True, True, True, True),
                attn_list=(False, False, False, True, False),
                n_blocks=2,
                # min_channels_decoder=64,
                n_groups=8,
                attn_heads=1,
            )
            pad_base_size = 2 ** 5

        elif self.params['net_type'] == 'res_unet_256':
            ## DDPM
            base_net = ResUNet(
                in_channels=n_pre_feat + self.x_shape[0],
                out_channels=self.x_shape[0] * self.params['n_dirs'],
                # channels_list=(128, 128, 256, 256, 512, 512),
                channels_list=(64, 64, 128, 128, 256, 256),
                bottleneck_channels=512,
                downsample_list=(False, True, True, True, True, True),
                attn_list=(False, False, False, False, True, False),
                n_blocks=2,
                # min_channels_decoder=64,
                n_groups=8,
                attn_heads=1,
            )
            pad_base_size = 2 ** 5

        else:
            raise Exception(f'Unsupported net_type: "{self.params["net_type"]}"')

        net = PCWrapper(
            net=base_net,
            pre_net=pre_net,
            n_dirs=self.params['n_dirs'],
            offset=self.data_module.mean,
            scale=self.data_module.std,
            mask=self.restoration_model.mask,
            pad_base_size=pad_base_size,
            pre_pad_base_size=pre_pad_base_size,
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

    def get_dirs(self, x_distorted, x_restored, use_best=True, **kwargs):
        w_mat = self['net'](x_distorted, x_restored, use_best=use_best, **kwargs)
        return w_mat

    def split_batch(self, batch, n):
        batches = self.restoration_model.split_batch(batch, n)
        return batches

    def process_batch(self, batch):
        with torch.no_grad():
            x_org, x_distorted = self.restoration_model.process_batch(batch)
            x_restored = self.restoration_model.restore(x_distorted)
        return x_org, x_distorted, x_restored

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

        dim = np.prod(model.x_shape)

        x_org, x_distorted, x_restored = model.process_batch(batch)

        w_mat = model.get_dirs(x_distorted, x_restored, use_best=False, use_ddp=True)

        w_mat_ = w_mat.flatten(2)
        w_norms = w_mat_.norm(dim=2)
        w_hat_mat = w_mat_ / w_norms[:, :, None]

        err = (x_org - x_restored).flatten(1)

        ## Normalizing by the error's norm
        ## -------------------------------
        err_norm = err.norm(dim=1)
        err = err / err_norm[:, None]
        w_norms = w_norms / err_norm[:, None]

        ## W hat loss
        ## ----------
        err_proj = torch.einsum('bki,bi->bk', w_hat_mat, err)
        reconst_err = 1 - err_proj.pow(2).sum(dim=1)

        ## W norms loss
        ## ------------
        second_moment_mse = (w_norms.pow(2) - err_proj.detach().pow(2)).pow(2)

        second_moment_loss_lambda = -1 + 2 * model['net'].step / model.params['second_moment_loss_grace']
        second_moment_loss_lambda = max(min(second_moment_loss_lambda, 1) ,1e-6)
        second_moment_loss_lambda *= model.params['second_moment_loss_lambda']
        objective = reconst_err.mean() + second_moment_loss_lambda * second_moment_mse.mean()

        ## Store log
        ## ---------
        log = dict(
            x_org=x_org,
            x_distorted=x_distorted,
            x_restored=x_restored.detach(),
            w_mat=w_mat.detach(),

            err_mse=err_norm.detach().pow(2) / dim,
            err_proj=err_proj.detach(),
            w_norms=w_norms.detach(),
            reconst_err=reconst_err.detach(),
            second_moment_mse=second_moment_mse.detach(),

            objective=objective.detach(),
        )
        return objective, log

    ## Logging
    ## -------
    def _init_train_log_data(self):
        model = self.model
        with EncapsulatedRandomState(42):
            x_org_fixed, x_distorted_fixed, x_restored_fixed = model.process_batch(model.split_batch(self.fixed_batch, self.n_batch_accumulation)[0])
        with EncapsulatedRandomState(42):
            x_org_valid, x_distorted_valid, x_restored_valid = model.process_batch(model.split_batch(self.valid_batch, self.n_batch_accumulation)[0])

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
                                         for key in ('step', 'lr', 'objective', 'err_mse',
                                                     'reconst_err', 'second_moment_mse',
                                                     ) \
                                                     + tuple(f'err_proj_square_{k}' for k in range(model.params['n_dirs']))
                                                     + tuple(f'w_norms_square_{k}' for k in range(model.params['n_dirs']))
            }
            model.extra_data['train_log_data']['logs'] = logs
        train_log_data['logs'] = logs

        ## Prepare summary
        ## ---------------
        summary = torchinfo.summary(model.networks['net'].net, input_data=x_distorted_fixed[:1], depth=10, verbose=0, device=model.device, x_restored=x_restored_fixed[:1] if (x_restored_fixed is not None) else None)
        # summary.formatting.verbose = 2
        summary = str(summary)
        train_log_data['summary'] = summary

        ## Prepare images
        ## --------------
        imgs = {
            'x_org_fixed': imgs_to_grid(x_org_fixed),
            'x_distorted_fixed': imgs_to_grid(model.restoration_model.distortion_model.naive_restore(x_distorted_fixed)),
            'x_restored_fixed': imgs_to_grid(x_restored_fixed) if x_restored_fixed is not None else None,
            'x_org_valid': imgs_to_grid(x_org_valid),
            'x_distorted_valid': imgs_to_grid(model.restoration_model.distortion_model.naive_restore(x_distorted_valid)),
            'x_restored_valid': imgs_to_grid(x_restored_valid) if x_restored_valid is not None else None,

            'batch_train': imgs_to_grid(torch.zeros((2, self.preview_width) + model.x_shape)),
            'batch_fixed': imgs_to_grid(torch.zeros((2, self.preview_width) + model.x_shape)),
            'batch_valid': imgs_to_grid(torch.zeros((2, self.preview_width) + model.x_shape)),
            'batch_fullv': imgs_to_grid(torch.zeros((2, self.preview_width) + model.x_shape)),
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

        figs['err_mse'] = go.Figure(
            data=[go.Scatter(mode='lines', x=[], y=[], name=title) for title in ('Train', 'Fixed', 'Validation', 'Full validation')],
            layout=go.Layout(yaxis_title='Error MSE', xaxis_title='step', height=400, width=550, margin=dict(t=0, b=20, l=20, r=0)),
        )

        figs['reconst_err'] = go.Figure(
            data=[go.Scatter(mode='lines', x=[], y=[], name=title) for title in ('Train', 'Fixed', 'Validation', 'Full validation')],
            layout=go.Layout(yaxis_title='Reconstruction Error', xaxis_title='step', height=400, width=550, margin=dict(t=0, b=20, l=20, r=0)),
        )

        figs['second_moment_mse'] = go.Figure(
            data=[go.Scatter(mode='lines', x=[], y=[], name=title) for title in ('Train', 'Fixed', 'Validation', 'Full validation')],
            layout=go.Layout(yaxis_title='Second Moment', xaxis_title='step', height=400, width=550, margin=dict(t=0, b=20, l=20, r=0))
        )

        for field in ('fixed', 'valid'):
            figs[f'err_proj_{field}'] = go.Figure(
                data=[go.Scatter(mode='lines', x=[], y=[], name=str(i)) for i in range(model.params['n_dirs'])],
                layout=go.Layout(yaxis_title='Error proj.', xaxis_title='step', height=400, width=550, margin=dict(t=0, b=20, l=20, r=0)),
            )
            figs[f'w_norms_{field}'] = go.Figure(
                data=[go.Scatter(mode='lines', x=[], y=[], name=str(i)) for i in range(model.params['n_dirs'])],
                layout=go.Layout(yaxis_title='W norms', xaxis_title='step', height=400, width=550, margin=dict(t=0, b=20, l=20, r=0)),
            )

        train_log_data['figs'] = figs

        return train_log_data

    def cat_logs(self, logs):
        log = {}
        for key in logs[0].keys():
            if key in ('x_org', 'x_distorted', 'x_restored', 'w_mat', 'err_mse', 'err_proj', 'w_norms', 'reconst_err', 'second_moment_mse'):
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
        w_mat = log['w_mat']

        err_mse_full = log['err_mse']
        err_proj_full = log['err_proj']
        w_norms_full = log['w_norms']
        reconst_err_full = log['reconst_err']
        second_moment_mse_full = log['second_moment_mse']

        objective = log['objective']

        err_mse = err_mse_full.mean().item()
        reconst_err = reconst_err_full.mean().item()
        second_moment_mse = second_moment_mse_full.mean().item()
        objective = objective.item()

        x_org = self.sample_preview_width(x_org)
        x_distorted = self.sample_preview_width(x_distorted)
        x_restored = self.sample_preview_width(x_restored)
        w_mat = self.sample_preview_width(w_mat)

        w_norms = w_mat.flatten(2).norm(dim=-1)
        w_hat_mat = w_mat / w_norms[:, :, None, None, None]

        err = x_restored - x_org

        if x_org.shape[1] == 1:
            x_org = x_org.repeat_interleave(3, 1)
            x_distorted = x_distorted.repeat_interleave(3, 1)
            x_restored = x_restored.repeat_interleave(3, 1)
            err_scaled = err.repeat_interleave(3, 1)
            w_mat_scaled = w_mat.repeat_interleave(3, 2)
            for i in range(err_scaled.shape[0]):
                err_scaled[i] = color_img(err[i], max_val=torch.abs(err[i]).max() / 1.5)
            for i in range(w_mat_scaled.shape[0]):
                for j in range(w_mat_scaled.shape[1]):
                    w_mat_scaled[i, j] = color_img(w_hat_mat[i, j], max_val=torch.abs(w_hat_mat[i, j]).max() / 1.5)
        else:
            # err_scaled = torch.abs(err)
            # err_scaled = err_scaled / err_scaled.flatten(-3).max(-1)[0][..., None, None, None]
            # w_mat_scaled = torch.abs(w_mat)
            # w_mat_scaled = w_mat_scaled / w_mat_scaled.flatten(-3).max(-1)[0][..., None, None, None]
            err_scaled = err / torch.abs(err).flatten(-3).max(-1)[0][..., None, None, None] / 1.5 + 0.5
            w_mat_scaled = w_mat / torch.abs(w_mat).flatten(-3).max(-1)[0][..., None, None, None] / 1.5 + 0.5

        x_distorted = model.restoration_model.distortion_model.naive_restore(x_distorted)

        batch_img = imgs_to_grid(
            torch.cat((
                torch.stack((x_org, x_distorted, x_restored, err_scaled), dim=1),
                w_mat_scaled,
            ), dim=1).transpose(0, 1).contiguous())

        ## Update console message
        ## ----------------------
        self.set_msg(field, f'{field}: step:{step:7d};   reconst err: {reconst_err:9.4g};   second moment mse: {second_moment_mse:9.4g};   objective: {objective:9.4g}')

        ## Store log data
        ## --------------
        logs = self.train_log_data['logs']
        logs[f'step_{field}'].append(step)
        logs[f'lr_{field}'].append(lr)
        logs[f'objective_{field}'].append(objective)
        logs[f'err_mse_{field}'].append(err_mse)
        logs[f'reconst_err_{field}'].append(reconst_err)
        logs[f'second_moment_mse_{field}'].append(second_moment_mse)
        for k in range(model.params['n_dirs']):
            logs[f'err_proj_square_{k}_{field}'].append(err_proj_full[:, k].pow(2).mean().item())
            logs[f'w_norms_square_{k}_{field}'].append(w_norms_full[:, k].pow(2).mean().item())

        figs = self.train_log_data['figs']
        if field in ('fixed',):
            figs['lr'].data[0].update(x=logs['step_fixed'], y=logs['lr_fixed'])
        for key in ('objective', 'err_mse', 'reconst_err', 'second_moment_mse'):
            figs[key].data[('train', 'fixed', 'valid', 'fullv').index(field)].update(x=logs[f'step_{field}'], y=logs[f'{key}_{field}'])
        if field in ('fixed', 'valid'):
            for k in range(model.params['n_dirs']):
                figs[f'err_proj_{field}'].data[k].update(x=logs[f'step_{field}'], y=logs[f'err_proj_square_{k}_{field}'])
                figs[f'w_norms_{field}'].data[k].update(x=logs[f'step_{field}'], y=logs[f'w_norms_square_{k}_{field}'])

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
                f'objective2/reconst_err_{field}': reconst_err,
                f'objective3/second_moment_mse_{field}': second_moment_mse,
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
                # log = {key: log[key] for key in ('x_org', 'err_mse', 'reconst_err', 'second_moment_mse', 'objective')}
            logs.append(log)
        log = self.cat_logs(logs)
        self.log_step(log, 'fullv')

        score = log['reconst_err'].mean().item()

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
                    <br/>
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
                    <br/>
                    {err_mse_fig}
                    {reconst_err_fig}
                    {second_moment_mse_fig}
                    <br>
                    {err_proj_fixed_fig}
                    {err_proj_valid_fig}
                    <br/>
                    {w_norms_fixed_fig}
                    {w_norms_valid_fig}
                    <br/>
                    {lr_fig}
                </div>

                <div>
                    <h2>Model</h2>

                    <h3>Train data</h3>
                    {x_org_fixed_img}
                    {x_distorted_fixed_img}
                    {x_restored_fixed_img}
                    <br/>
                    {x_org_valid_img}
                    {x_distorted_valid_img}
                    {x_restored_valid_img}
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
            log_dict = {
                'model/summary': wandb.Html(self.train_log_data['summary']),
                'model/x_org_fixed': wandb.Image(self.train_log_data['imgs']['x_org_fixed']),
                'model/x_distorted_fixed': wandb.Image(self.train_log_data['imgs']['x_distorted_fixed']),
                'model/x_org_vaild': wandb.Image(self.train_log_data['imgs']['x_org_vaild']),
                'model/x_distorted_vaild': wandb.Image(self.train_log_data['imgs']['x_distorted_vaild']),
            }
            if self.train_log_data['imgs']['x_restored_fixed'] is not None:
                log_dict.update({
                    'model/x_restored_fixed': wandb.Image(self.train_log_data['imgs']['x_restored_fixed']),
                    'model/x_restored_valid': wandb.Image(self.train_log_data['imgs']['x_restored_valid']),
                })
            self.wandb.log(log_dict, commit=True)

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


def gram_schmidt(x):
    x_shape = x.shape
    x = x.flatten(2)

    x_orth = []
    proj_vec_list = []
    for i in range(x.shape[1]):
        w = x[:, i, :]
        for w2 in proj_vec_list:
            w = w - w2 * torch.sum(w * w2, dim=-1, keepdim=True)
        w_hat = w.detach() / w.detach().norm(dim=-1, keepdim=True)

        x_orth.append(w)
        proj_vec_list.append(w_hat)

    x_orth = torch.stack(x_orth, dim=1).view(*x_shape)
    return x_orth

# class GramSchmidt(nn.Module):
#     def forward(self, x):
#         return gram_schmidt(x)


class PCWrapper(nn.Module):
    def __init__(self, net, n_dirs, pre_net=None, offset=None, scale=None, mask=None, pad_base_size=None, pre_pad_base_size=None):
        super().__init__()

        self.net = net
        self.pre_net = pre_net
        self.n_dirs = n_dirs
        self.offset = offset
        self.scale = scale
        self.mask = mask
        self.pad_base_size = pad_base_size
        self.pre_pad_base_size = pre_pad_base_size

    @staticmethod
    def _get_padding(x, base_size):
        s = base_size
        _, _, height, width = x.shape
        if (s is not None) and ((height % s != 0) or (width % s != 0)):
            pad_h = height % s
            pad_w = width % s
            padding = torch.tensor((pad_h // 2, pad_h // 2, pad_w // 2, pad_w // 2))
        else:
            padding = None
        return padding

    def forward(self, x_distorted, x_restored):
        if self.offset is not None:
            x_distorted = x_distorted - self.offset
            x_restored = x_restored - self.offset
        if self.scale is not None:
            x_distorted = x_distorted / self.scale
            x_restored = x_restored / self.scale

        ## Pre-process distorted image
        ## ---------------------------
        if self.pre_net is None:
            x = x_distorted
        else:
            padding = self._get_padding(x_distorted, self.pre_pad_base_size)
            if padding is not None:
                x_distorted = F.pad(x_distorted, tuple(padding))
            x = self.pre_net(x_distorted)

            if padding is not None:
                x = F.pad(x, tuple(-padding))  # pylint: disable=invalid-unary-operand-type

        ## Process both images
        ## -------------------
        x = torch.cat((x, x_restored), dim=1)

        padding = self._get_padding(x, self.pad_base_size)
        if padding is not None:
            x = F.pad(x, tuple(padding))

        w_mat = self.net(x)
        if self.scale is not None:
            w_mat = w_mat / self.scale
        if padding is not None:
            w_mat = F.pad(w_mat, tuple(-padding))  # pylint: disable=invalid-unary-operand-type
        w_mat = w_mat.unflatten(1, (self.n_dirs, w_mat.shape[1] // self.n_dirs))
        if self.mask is not None:
            w_mat = w_mat * self.mask[None, None]
        w_mat = gram_schmidt(w_mat)
        # w_mat = w_mat / w_mat.flatten(3).shape[-1] ** 0.5

        return w_mat


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
            zero_weights(nn.Conv2d(ch, channels_out, 1)),
        ]
        out_block = nn.Sequential(*layers)

        self.net = nn.Sequential(
            in_block,
            main_block,
            out_block,
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


## Dataset
## =======
class ImageFilesDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, transform=None, store_dataset=False):
        super().__init__()

        self.filenames = filenames
        self.transform = transform
        self._store_dataset = store_dataset

        self.stored = {}

    def __call__(self, filename):
        if self._store_dataset:
            if filename in self.stored:
                img = self.stored[filename]
            else:
                img = Image.open(filename).convert('RGB')
                self.stored[filename] = img
        else:
            img = Image.open(filename).convert('RGB')
        return img

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        img = self(filename)
        if self.transform is not None:
            img = self.transform(img)
        return img


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
