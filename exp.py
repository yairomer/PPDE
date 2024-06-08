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
import matplotlib as mpl
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
    # store_dataset = True,

    # learning_method = 'multilevel-nce',
    # name = 'exp_{now}_ebm_{dataset}_{distortion_type}_nce',

    # ## MNIST - In-painting
    # ## -------------------
    # pc_model_folder = os.path.join(os.environ['HOME'], 'workspace/tmp/exp/nppc_ebm/pc/best/mnist_in_painting'),
    # batch_size = 256,
    # pre_net_type = 'none',
    # feat_net_type = 'half_resnet18',
    # dist_net_type = 'mlp1',
    # n_feat = 256,
    # lr = 1e-4,
    # n_epochs = 500,
    # z_scale = 1.5,

    # rand_rot = True,
    # rand_rot_benchmark = False,
    # name = 'exp_{now}_ebm_{dataset}_{distortion_type}_rand_dir',
    # n_epochs = 2000,

    # ## MNIST - Denoising
    # ## -----------------
    # pc_model_folder = os.path.join(os.environ['HOME'], 'workspace/tmp/exp/nppc_ebm/pc/best/mnist_denoising'),
    # batch_size = 256,
    # pre_net_type = 'none',
    # feat_net_type = 'half_resnet18',
    # dist_net_type = 'mlp1',
    # n_feat = 256,
    # lr = 1e-4,
    # n_epochs = 5000,
    # z_scale = 1.5,

    # ## CelebA 256x256 - In-painting eyes
    # ## ---------------------------------
    # pc_model_folder = os.path.join(os.environ['HOME'], 'workspace/tmp/exp/nppc_ebm/pc/best/celeba_hq_256_in_painting_eyes'),
    # batch_size = 32,
    # max_batch_chunk_size = 8,
    # pre_net_type = 'none',
    # feat_net_type = 'half_ddpm',
    # dist_net_type = 'mlp1',
    # n_feat = 512,
    # lr = 1e-4,
    # n_epochs = 300,
    # z_scale = 5.,
    # log_every = 50,
    # benchmark_every = 250,

    # ## CelebA 256x256 - In-painting mouth
    # ## ----------------------------------
    # pc_model_folder = os.path.join(os.environ['HOME'], 'workspace/tmp/exp/nppc_ebm/pc/best/celeba_hq_256_in_painting_mouth'),
    # batch_size = 32,
    # max_batch_chunk_size = 8,
    # pre_net_type = 'none',
    # feat_net_type = 'half_ddpm',
    # dist_net_type = 'mlp1',
    # n_feat = 512,
    # lr = 1e-4,
    # n_epochs = 300,
    # z_scale = 2.,
    # log_every = 50,
    # benchmark_every = 250,

    # ## CelebA 256x256 - Colorization
    # ## -----------------------------
    # pc_model_folder = os.path.join(os.environ['HOME'], 'workspace/tmp/exp/nppc_ebm/pc/best/celeba_hq_256_colorization'),
    # batch_size = 32,
    # max_batch_chunk_size = 8,
    # pre_net_type = 'none',
    # feat_net_type = 'half_ddpm',
    # dist_net_type = 'mlp1',
    # n_feat = 512,
    # lr = 1e-4,
    # n_epochs = 300,
    # z_scale = 9.,
    # log_every = 50,
    # benchmark_every = 250,

    # ## FFHQ - Colorization
    # ## -------------------
    # pc_model_folder = os.path.join(os.environ['HOME'], 'workspace/tmp/exp/nppc_ebm/pc/best/ffhq_colorization'),
    # batch_size = 32,
    # max_batch_chunk_size = 8,
    # pre_net_type = 'none',
    # feat_net_type = 'half_ddpm',
    # dist_net_type = 'mlp1',
    # n_feat = 512,
    # lr = 1e-4,
    # n_epochs = 300,
    # z_scale = 9.,
    # log_every = 50,
    # benchmark_every = 250,

    ## CelebA 256x256 - Super resolution
    ## ---------------------------------
    pc_model_folder = os.path.join(os.environ['HOME'], 'workspace/tmp/exp/nppc_ebm/pc/best/celeba_super_resolution'),
    batch_size = 8,
    pre_net_type = 'res_cnn',
    feat_net_type = 'resnet2',
    dist_net_type = 'mlp1',
    n_feat = 512,
    lr = 1e-4,
    n_epochs = 500,
    z_scale = 2,
    log_every = 50,
    benchmark_every = 250,

    # ## DIV2K - Super resolution
    # ## ------------------------
    # pc_model_folder = os.path.join(os.environ['HOME'], 'workspace/tmp/exp/nppc_ebm/pc/best/div2k_super_resolution'),
    # batch_size = 8,
    # feat_net_type = 'resnet3',
    # dist_net_type = 'mlp1',
    # n_feat = 512,
    # lr = 3e-5,
    # n_epochs = 10000,
    # z_scale = 3,

    # ## ImageNet - Colorization
    # ## -----------------------
    # pc_model_folder = os.path.join(os.environ['HOME'], 'workspace/tmp/exp/nppc_ebm/pc/best/imagenet_colorization'),
    # batch_size = 32,
    # max_batch_chunk_size = 32,
    # pre_net_type = 'none',
    # feat_net_type = 'half_ddpm',
    # dist_net_type = 'mlp1',
    # n_feat = 512,
    # lr = 1e-4,
    # n_epochs = 30,
    # z_scale = 4.,
    # log_every = 50,
    # benchmark_every = 250,
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
        project_name = 'nppc_ebm/ppd',
        name = 'exp_{now}_ebm_{dataset}_{distortion_type}',
        pc_model_folder = None,
        data_folders = DEFAULT_FOLDERS['data_folders'],
        img_size = None,
        store_dataset = False,

        n_levels = 128,
        z_scale = 1.,
        pk_decay = 1.,
        n_dirs = 2,
        # n_dirs = None,
        pre_net_type = 'none',
        feat_net_type = 'half_resnet18',
        dist_net_type = 'mlp1',
        n_feat = 256,

        lr = 1e-4,
        weight_decay = 0.,
        # weight_decay = 1e-4,

        ema_step = 1,
        ema_alpha = None,

        init_step_size_data = 0.001,
        init_step_size_ref = 0.1,

        model_random_seed = 42,
    ),

    ## Training parameters
    ## -------------------
    train = dict(
        batch_size = 256,
        max_batch_chunk_size = None,
        n_epochs = 500,
        n_steps = None,
        gradient_clip_val = None,
        overfit = False,

        learning_method = 'multilevel-cond-cd-nce',

        rand_rot = False,
        rand_rot_benchmark = False,
        n_leaps = 1,  # The number of leaps in the HMC sampler (1=MALA)
        mh_adjust = True,  # Use Metropolis-Heisting rejection
        target_prob = 0.6,
        # mh_adjust = False,  # Use adjusted contrastive divergence
        # target_prob = 0.25,

        max_benchmark_samples = 256,

        ## Logging parameters
        log_every = 300,
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

GET_N_MCMC_STEPS = lambda step: 10

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
    def __init__(self, device='cpu', store_dataset=False, **params):

        self.device = device
        self.ddp = DDPManager()

        self.params = params
        self.extra_data = {}

        self.n_levels = self.params['n_levels']

        ## Initializing random state
        ## -------------------------
        set_random_seed(self.params['model_random_seed'])

        ## Load pc model
        ## -------------
        pc_model_folder = self.params['pc_model_folder']
        folder_manager = ExpFolderManager(pc_model_folder)
        pc_model_exp = folder_manager.get_exp()
        self.pc_model = pc_model_exp.Model.load(folder_manager.checkpoint_filename, device=self.device, store_dataset=store_dataset)
        for net in self.pc_model.networks.values():
            net.requires_grad_ = False
        self.pc_params = self.pc_model.restoration_model.params
        self.pc_params.update(self.pc_model.params)

        self.n_dirs_org = self.pc_model.n_dirs
        self.n_dirs = self.params['n_dirs'] if (self.params['n_dirs'] is not None) else self.n_dirs_org

        ## Run name
        ## --------
        name = self.params['name']
        if name is None:
            self.name = 'exp_{now}'
        now = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
        params_ = self.pc_params.copy()
        params_.update(self.params)
        self.name = name.format(now=now, **params_)

        ## Prepare datasets
        ## ----------------
        self.data_module = self.pc_model.data_module
        self.x_shape = self.data_module.shape

        self.data_offset = self.data_module.mean
        self.data_scale = self.data_module.std

        ## Setup interpolation generator
        ## -----------------------------
        self.gen_interp = GenInterp(
            self.n_levels,
            scale=self.params['z_scale'] * 5,
            offset=0,
            gamma=1.3,
            pk_decay=self.params['pk_decay'],
        ).to(self.device)

        ## Step size
        ## ---------
        ## Initialize the step size for each k
        step_size_edges = torch.as_tensor((self.params['init_step_size_data'], self.params['init_step_size_ref']), device=self.device)
        self.step_size = ((self.gen_interp.mix_factors ** 2) @ (step_size_edges ** 2)) ** 0.5

        ## Set parametric model
        ## --------------------
        pre_pad_base_size = None
        if self.params['pre_net_type'] == 'none':
            if self.pc_model.restoration_model.upscale_factor is None:
                pre_net = None
            else:
                pre_net = nn.Upsample(scale_factor=self.pc_model.restoration_model.upscale_factor, mode='nearest')
            n_pre_feat = self.pc_model.restoration_model.x_distorted_shape[0]

        elif self.params['pre_net_type'] == 'res_cnn':
            n_pre_feat = 64
            pre_net = ResCNN(
                channels_in=self.pc_model.restoration_model.x_distorted_shape[0],
                channels_out=n_pre_feat,
                channels_hidden=64,
                n_blocks=16,
                upscale_factor=self.pc_model.restoration_model.upscale_factor,
            )

        else:
            raise Exception(f'Unsupported pre_net_type: "{self.params["pre_net_type"]}"')

        if self.params['feat_net_type'] == 'half_resnet18':
            ## HalfResNet18
            feat_net = ResNetA(
                in_channels=n_pre_feat + self.x_shape[0] + self.x_shape[0] * self.n_dirs,
                out_channels=self.params['n_feat'],
                # n_blocks_list=(1, 1, 1, 1),
                n_blocks_list=(2, 2, 2, 2),
                channels_list=(32, 64, 128, 256),
            )

        elif self.params['feat_net_type'] == 'resnet18':
            ## ResNet18
            feat_net = ResNetA(
                in_channels=n_pre_feat + self.x_shape[0] + self.x_shape[0] * self.n_dirs,
                out_channels=self.params['n_feat'],
                n_blocks_list=(2, 2, 2, 2),
                channels_list=(64, 128, 256, 512),
            )

        elif self.params['feat_net_type'] == 'resnet2':
            ## ResNet18
            feat_net = ResNetA(
                in_channels=n_pre_feat + self.x_shape[0] + self.x_shape[0] * self.n_dirs,
                out_channels=self.params['n_feat'],
                n_blocks_list=(2, 2, 2, 2, 2),
                channels_list=(32, 64, 128, 256, 512),
                in_stride=2,
            )

        elif self.params['feat_net_type'] == 'resnet3':
            ## ResNet50
            feat_net = ResNetB(
                in_channels=n_pre_feat + self.x_shape[0] + self.x_shape[0] * self.n_dirs,
                out_channels=self.params['n_feat'],
                n_blocks_list=(3, 4, 6, 3),
                channels_list=(64, 128, 256, 512),
            )

        elif self.params['feat_net_type'] == 'half_ddpm':
            ## Half DDPM
            feat_net = ResNet4(
                in_channels=n_pre_feat + self.x_shape[0] + self.x_shape[0] * self.n_dirs,
                out_channels=self.params['n_feat'],
                # channels_list=(128, 128, 256, 256, 512, 512, 512),
                channels_list=(64, 64, 128, 128, 256, 512, 512),
                downsample_list=(False, True, True, True, True, True, True),
                attn_list=(False, False, False, False, True, False, True),
                n_blocks=2,
                n_groups=8,
                attn_heads=1,
            )

        else:
            raise Exception(f'Unsupported feat_net_type: "{self.params["feat_net_type"]}"')

        if self.params['dist_net_type'] == 'mlp1':
            cond_ebm_net = CondMLP(
                in_channels=self.n_dirs,
                out_channels=128,
                feat_channels=self.params['n_feat'],
                hidden_channels=128,
                n_blocks=7,
            )

        else:
            raise Exception(f'Unsupported dist_net_type: "{self.params["dist_net_type"]}"')

        net = CondEBMWrapper(
            pre_net=pre_net,
            feat_net=feat_net,
            cond_ebm_net=cond_ebm_net,
            n_levels=self.n_levels,
            pad_base_size=None,
            pre_pad_base_size=pre_pad_base_size,
            input_offset=self.data_offset,
            input_scale=self.data_scale,
            z_scale=self.params['z_scale'],
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

    @staticmethod
    def rand_rotation(x):
        return x[:, torch.randperm(x.shape[1])]

    # @staticmethod
    # def rand_rotation(x):
    #     dim_shape = x.shape[2:]
    #     x = x.flatten(2)
    #     x = x / x.norm(dim=2, keepdim=True)

    #     q_mat, r_mat = torch.linalg.qr(torch.randn((x.shape[0], x.shape[1], x.shape[1]), device=x.device))
    #     sign_mat = (r_mat * torch.eye(r_mat.shape[1], device=x.device)[None]).sign()
    #     q_mat = torch.einsum('nik,nkj->nij', q_mat, sign_mat)
    #     x = torch.einsum('nki,nkd->nid', q_mat, x)

    #     x = x.unflatten(-1, dim_shape)
    #     return x

    def split_batch(self, batch, n):
        batches = self.pc_model.split_batch(batch, n)
        return batches

    def process_batch(self, batch):
        with torch.no_grad():
            x_org, x_distorted, x_restored = self.pc_model.process_batch(batch)
            dirs = self.pc_model.get_dirs(x_distorted, x_restored)
        return x_org, x_distorted, x_restored, dirs

    @staticmethod
    def project_x(x, x0, dirs):
        x = (x - x0).flatten(1)
        dirs = dirs.flatten(2)
        dirs = dirs / dirs.norm(dim=2, keepdim=True)
        z = torch.einsum('nic,nc->ni', dirs, x)
        return z

    def gen_feat(self, x_distorted, x_restored, dirs, use_best=True, **kwargs):
        feat = self['net'](x_distorted, x_restored, dirs, calc_feat=True, use_best=use_best)
        return feat

    def log_pzk(self, z_aug, feat, use_best=True, **kwargs):
        log_p_ref = self.gen_interp.log_p_ref(z_aug)  # pylint: disable=not-callable
        log_p_ratios = self['net'](z_aug, feat, use_best=use_best, **kwargs)
        log_pzk = log_p_ratios + log_p_ref[:, None] + self.gen_interp.log_pk[-1]
        return log_pzk

    def log_pz(self, z, feat, **kwargs):
        return self.log_pzk(z, feat, **kwargs)[:, 0] - self.gen_interp.log_pk[0]

    ## Save and load
    ## -------------
    def state_dict(self):
        state_dict = dict(
            name = self.name,
            params = self.params,
            extra_data = self.extra_data,
            networks = {key: network.state_dict() for key, network in self.networks.items()},
            step_size = self.step_size,
            )
        return state_dict

    def load_state_dict(self, state_dict):
        self.name = state_dict['name']
        self.extra_data = state_dict['extra_data']
        for key, val in state_dict['networks'].items():
            self.networks[key].load_state_dict(val)
        self.step_size.copy_(state_dict['step_size'])

    @classmethod
    def load(cls, checkpoint_filename, device='cpu', **kwargs):
        state_dict = torch.load(checkpoint_filename, map_location=device)
        state_dict['params'].update(kwargs)

        model = cls(device=device, **state_dict['params'])
        model.load_state_dict(state_dict)
        step_str = ', '.join([f'{key}: {net.step}' for key, net in model.networks.items()])
        print(f'Resuming step: {step_str}')
        return model


## Interpolation generator
## =======================
class GenInterp:
    def __init__(self, n_levels, scale=1., offset=0., gamma=1., pk_decay=1.):

        super().__init__()

        self.n_levels = n_levels
        self.offset = offset
        self.log_pn = GaussianModel(mean=self.offset, std=scale)

        beta = torch.linspace(0.01, 1, n_levels) ** gamma
        alpha = (1 - beta ** 2) ** 0.5
        self.mix_factors = torch.stack((alpha, beta), dim=1)

        pk = pk_decay ** torch.linspace(0, 1, n_levels)
        pk /= pk.sum()
        self.pk = pk
        self.log_pk = torch.log(self.pk)

    def to(self, device):
        self.mix_factors = self.mix_factors.to(device)
        self.log_pn = self.log_pn.to(self.device)
        self.pk = self.pk.to(self.device)
        self.log_pk = self.log_pk.to(self.device)
        return self

    @property
    def device(self):
        return self.mix_factors.device

    def log_p_ref(self, x):
        return self.log_pn(x)

    def draw_k(self, n_samples, sorted_k=False):
        # k = torch.randint(0, self.n_levels, (n_samples,), device=self.device)
        k = torch.multinomial(self.pk, num_samples=n_samples, replacement=True)
        if sorted_k:
            k = torch.sort(k)[0]
        return k

    def draw_ref(self, shape):
        x_ref = self.log_pn.draw_samples(shape)
        return x_ref

    def draw(self, x, k=None, x_ref=None, sorted_k=False):
        if x_ref is None:
            x_ref = self.log_pn.draw_samples(x.shape)
        if k is None:
            k = self.draw_k(x.shape[0], sorted_k=sorted_k)
        elif np.isscalar(k):
            k = torch.ones(x.shape[0], device=self.device).long() * k

        alpha = self.mix_factors[k, 0]
        beta = self.mix_factors[k, 1]
        ## Extend alpha & beta to the dimension of x
        alpha = alpha[(slice(None),) + (None,) * (x.dim() - 1)]
        beta = beta[(slice(None),) + (None,) * (x.dim() - 1)]

        x = x - self.offset
        x_ref = x_ref - self.offset
        x_aug = alpha * x + beta * x_ref
        x_aug = x_aug + self.offset

        return x_aug, k


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

        max_z = model.params['z_scale'] * 4
        z_range = torch.linspace(-max_z, max_z, 201)
        self.dist_calc_z = DistCalc(log_p_func=None, ranges=(z_range, z_range), device=model.device)

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
                    objective, log = self.base_step(batch_part, adjust_step_size=last_part)
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
                            score = self.benchmark()
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

    def base_step(self, batch, adjust_step_size=False):
        model = self.model
        n_leaps = self.params['n_leaps']
        mh_adjust = self.params['mh_adjust']

        learning_method = self.params['learning_method']

        x_org, x_distorted, x_restored, dirs = model.process_batch(batch)
        if self.params['rand_rot']:
            dirs = model.rand_rotation(dirs)
        dirs = dirs[:, :model.n_dirs]

        z = model.project_x(x_org, x_restored, dirs)
        z_aug, k = model.gen_interp.draw(z, sorted_k=True) ## Generate z_aug and k

        z_aug_gen = torch.zeros(z.shape, device=model.device)
        # k_gen = torch.zeros(z.shape[:1], device=model.device)
        ce_nll = torch.zeros(z.shape[:1], device=model.device)
        cd_nll = torch.zeros(z.shape[:1], device=model.device)
        log_pz_aug_org = torch.zeros(z.shape[:1], device=model.device)
        log_pz_aug_gen = torch.zeros(z.shape[:1], device=model.device)
        step_prob = torch.zeros(z.shape[:1], device=model.device)
        n_mcmc_steps = 0

        if learning_method == 'multilevel-nce':
            ## Forward pass
            ## ------------
            feat = model.gen_feat(x_distorted, x_restored, dirs, use_best=False, use_ddp=True)
            log_pzk = model.log_pzk(z_aug, feat, use_best=False, use_ddp=True)

            ## Calculate cross entropy loss (p(k|x_aug))
            ## -----------------------------------------
            ce_nll = F.cross_entropy(log_pzk, k)

            objective = ce_nll

        # elif learning_method == 'multilevel-cnce':
        #     z_aug_gen = z_aug + torch.randn_like(z_aug) * model.step_size[k][:, None, None, None]

        #     ## Forward pass
        #     ## ------------
        #     feat = model.gen_feat(x_distorted, x_restored, dirs, use_best=False, use_ddp=True)

        #     log_pzk_org = model.log_pzk(z_aug, feat, use_best=False, use_ddp=True)
        #     log_pzk_gen = model.log_pzk(z_aug_gen, feat, use_best=False, use_ddp=True)

        #     ## Calculate cross entropy loss (p(k|x_aug))
        #     ## -----------------------------------------
        #     ce_nll = F.cross_entropy(log_pzk_org, k)

        #     ## Calculate conditional contrastive noise estimation
        #     ## --------------------------------------------------
        #     log_pz_aug_org = torch.logsumexp(log_pzk_org, dim=1)
        #     log_pz_aug_gen = torch.logsumexp(log_pzk_gen, dim=1)
        #     log_pz_aug_diff = log_pz_aug_org - log_pz_aug_gen
        #     cd_nll = -F.logsigmoid(log_pz_aug_diff).mean()

        #     objective = cd_nll + ce_nll

        #     step_prob = torch.exp(-log_pz_aug_diff.detach()).clip(max=1.)
        #     n_mcmc_steps = GET_N_MCMC_STEPS(model.networks['nets'].step)

        #     log_pz_aug = lambda z_aug: torch.logsumexp(model.log_pzk(z_aug, feat, use_best=False) , dim=1)
        #     sampler = HMCSampler(log_pz_aug, z_aug.clone(), step_size=model.step_size[k])
        #     chain_log_prob_diff = 0
        #     step_prob = 0
        #     for _ in range(n_mcmc_steps // n_leaps):
        #         _, stats = sampler.step(mh_adjust=mh_adjust, n_leaps=n_leaps)
        #         chain_log_prob_diff = chain_log_prob_diff + stats['trans_log_prob_diff']
        #         step_prob += stats['step_prob'].float() / n_mcmc_steps
        #     z_aug_gen = sampler.x

        #     ## Forward pass
        #     ## ------------
        #     feat = model.gen_feat(x_distorted, x_restored, dirs, use_best=False, use_ddp=True)

        #     log_pzk_org = model.log_pzk(z_aug, feat, use_best=False, use_ddp=True)
        #     log_pzk_gen = model.log_pzk(z_aug_gen, feat, use_best=False, use_ddp=True)

        #     ## Calculate loss (Mixed CD: p(z_aug,k))
        #     ## -------------------------------------
        #     with torch.no_grad():
        #         ## Draw k gen
        #         k_gen = torch.multinomial(torch.softmax(log_pzk_gen, dim=1), 1, replacement=True)[:, 0]
        #         ## Calculate log p(k | z) for the acceptance rate
        #         log_pk_given_z_aug_org = torch.log_softmax(log_pzk_org, dim=1)[torch.arange(len(k)), k]
        #         log_pk_given_z_aug_gen = torch.log_softmax(log_pzk_gen, dim=1)[torch.arange(len(k_gen)), k_gen]
        #     log_pzk_org = log_pzk_org[torch.arange(len(k)), k]
        #     log_pzk_gen = log_pzk_gen[torch.arange(len(k_gen)), k_gen]

        #     cd_log_prob_diff = log_pzk_org - log_pzk_gen + chain_log_prob_diff + log_pk_given_z_aug_gen - log_pk_given_z_aug_org
        #     cd_nll = -F.logsigmoid(cd_log_prob_diff).mean()

        #     objective = cd_nll
        #     ce_nll = -log_pk_given_z_aug_org.detach()

        #     ## Adjust step size
        #     ## ----------------
        #     if adjust_step_size:
        #         step_size = model.step_size[k].clone()
        #         indices = step_prob > self.params['target_prob']
        #         HMCSampler.adjust_step_static_in_place(step_size, indices)
        #         model.step_size[k] = step_size

        elif learning_method == 'multilevel-cd-nce':
            with torch.no_grad():
                feat = model.gen_feat(x_distorted, x_restored, dirs, use_best=False)

            ## Generate adversarial samples (z_aug_gen) - MCMC (contrastive divergence)
            ## ------------------------------------------------------------------------
            n_mcmc_steps = GET_N_MCMC_STEPS(model.networks['net'].step)

            log_pz_aug = lambda z_aug: torch.logsumexp(model.log_pzk(z_aug, feat, use_best=False) , dim=1)
            sampler = HMCSampler(log_pz_aug, z_aug.clone(), step_size=model.step_size[k])
            chain_log_prob_diff = 0
            step_prob = 0
            for _ in range(n_mcmc_steps // n_leaps):
                _, stats = sampler.step(mh_adjust=mh_adjust, n_leaps=n_leaps)
                chain_log_prob_diff = chain_log_prob_diff + stats['trans_log_prob_diff']
                step_prob += stats['step_prob'].float() / n_mcmc_steps
            z_aug_gen = sampler.x

            ## Forward pass
            ## ------------
            feat = model.gen_feat(x_distorted, x_restored, dirs, use_best=False, use_ddp=True)

            log_pzk_org = model.log_pzk(z_aug, feat, use_best=False, use_ddp=True)
            log_pzk_gen = model.log_pzk(z_aug_gen, feat, use_best=False, use_ddp=True)

            ## Calculate cross entropy loss (p(k|x_aug))
            ## -----------------------------------------
            ce_nll = F.cross_entropy(log_pzk_org, k)

            ## Calculate loss (Marginal CD: p(x_aug))
            ## --------------------------------------
            log_pz_aug_org = torch.logsumexp(log_pzk_org, dim=1)
            log_pz_aug_gen = torch.logsumexp(log_pzk_gen, dim=1)
            cd_log_prob_diff = log_pz_aug_org - log_pz_aug_gen + chain_log_prob_diff
            cd_nll = -F.logsigmoid(cd_log_prob_diff).mean()

            objective = cd_nll + ce_nll

            ## Adjust step size
            ## ----------------
            if adjust_step_size:
                step_size = model.step_size[k].clone()
                indices = step_prob > self.params['target_prob']
                HMCSampler.adjust_step_static_in_place(step_size, indices)
                model.step_size[k] = step_size

        elif learning_method == 'multilevel-cond-cd-nce':
            feat = model.gen_feat(x_distorted, x_restored, dirs, use_best=False, use_ddp=True)

            ## Generate adversarial samples (z_aug_gen) - MCMC (contrastive divergence)
            ## ------------------------------------------------------------------------
            n_mcmc_steps = GET_N_MCMC_STEPS(model.networks['net'].step)

            feat_ = feat.detach()
            log_pz_aug = lambda z_aug: model.log_pzk(z_aug, feat_, use_best=False).gather(1, k[:, None])[:, 0]
            sampler = HMCSampler(log_pz_aug, z_aug.clone(), step_size=model.step_size[k])
            chain_log_prob_diff = 0
            step_prob = 0
            for _ in range(n_mcmc_steps // n_leaps):
                _, stats = sampler.step(mh_adjust=mh_adjust, n_leaps=n_leaps)
                chain_log_prob_diff = chain_log_prob_diff + stats['trans_log_prob_diff']
                step_prob += stats['step_prob'].float() / n_mcmc_steps
            z_aug_gen = sampler.x

            ## Forward pass
            ## ------------
            log_pzk_org = model.log_pzk(z_aug, feat, use_best=False, use_ddp=True)
            log_pzk_gen = model.log_pzk(z_aug_gen, feat, use_best=False, use_ddp=True)

            ## Calculate cross entropy loss (p(k|x_aug))
            ## -----------------------------------------
            ce_nll = F.cross_entropy(log_pzk_org, k)

            ## Calculate loss (Marginal CD: p(x_aug))
            ## --------------------------------------
            log_pz_aug_org = log_pzk_org.gather(1, k[:, None])[:, 0]
            log_pz_aug_gen = log_pzk_gen.gather(1, k[:, None])[:, 0]
            cd_log_prob_diff = log_pz_aug_org - log_pz_aug_gen + chain_log_prob_diff
            cd_nll = -F.logsigmoid(cd_log_prob_diff).mean()

            objective = cd_nll + ce_nll

            ## Adjust step size
            ## ----------------
            if adjust_step_size:
                step_size = model.step_size[k].clone()
                indices = step_prob > self.params['target_prob']
                HMCSampler.adjust_step_static_in_place(step_size, indices)
                model.step_size[k] = step_size

        ## Store log
        ## ---------
        log = dict(
            x_org=x_org,
            x_distorted=x_distorted,
            x_restored=x_restored,
            dirs=dirs,
            z=z.detach(),
            z_aug=z_aug.detach(),
            z_aug_gen=z_aug_gen.detach(),
            # k_gen=k_gen,

            ce_nll=ce_nll.detach(),
            cd_nll=cd_nll.detach(),
            log_pz_org=log_pz_aug_org.detach(),
            log_pz_gen=log_pz_aug_gen.detach(),
            log_pz_diff=(log_pz_aug_org - log_pz_aug_gen).detach(),
            step_prob=step_prob,

            objective=objective.detach(),

            n_mcmc_steps=n_mcmc_steps,
        )
        return objective, log

    ## Logging
    ## -------
    def _init_train_log_data(self):
        model = self.model
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
                                         for key in ('step', 'lr', 'objective', 'ce_nll', 'cd_nll',
                                                     'log_pz0', 'log_pz_org', 'log_pz_gen', 'log_pz_diff',
                                                     'step_prob', 'n_mcmc_steps')}
            logs['step_benchmark'] = []
            logs['nll_benchmark'] = []
            logs['correction_benchmark'] = []
            model.extra_data['train_log_data']['logs'] = logs
        train_log_data['logs'] = logs

        ## Reference log_p
        ## ---------------
        dataloader = torch.utils.data.DataLoader(
            model.data_module.valid_set,
            batch_size=self.batch_size,
            shuffle=False,
        )
        z = []
        pca_std = []
        n = 0
        n_samples = self.params['max_benchmark_samples']
        with EncapsulatedRandomState(0):
            for batch in dataloader:
                x_org, _, x_restored, dirs = model.process_batch(batch)
                if self.params['rand_rot_benchmark']:
                    dirs = model.rand_rotation(dirs)
                dirs = dirs[:, :model.n_dirs]
                z_ = model.project_x(x_org, x_restored, dirs)
                pca_std_ = dirs.flatten(2).norm(dim=2)
                z.append(z_)
                pca_std.append(pca_std_)
                n += z_.shape[0]
                if n > n_samples:
                    break
        z = torch.cat(z, dim=0)
        pca_std = torch.cat(pca_std, dim=0)
        z = z[:n_samples]
        pca_std = pca_std[:n_samples]

        log_pz = (-0.5 - torch.log(z.abs()) - 0.5 * np.log(2 * np.pi)).sum(1)
        log_pz_std = log_pz.std().item() / log_pz.shape[0] ** 0.5
        log_pz = log_pz.mean().item()
        self.header.append(f'Reference - oracle diagonal Gaussian: -log(p(z)): {-log_pz:.3f} ± {-log_pz_std:.3f}')

        std = pca_std.abs()
        log_pz = (-0.5 * (z / std).pow(2) - torch.log(std) - 0.5 * np.log(2 * np.pi)).sum(1)
        log_pz_std = log_pz.std().item() / log_pz.shape[0] ** 0.5
        log_pz = log_pz.mean().item()
        self.header.append(f'Reference - PCA: -log(p(z)): {-log_pz:.3f} ± {-log_pz_std:.3f}')

        ## Prepare images
        ## --------------
        imgs = {
            'batch_train': imgs_to_grid(torch.zeros((2, self.preview_width) + model.x_shape)),
            'batch_fixed': imgs_to_grid(torch.zeros((2, self.preview_width) + model.x_shape)),
            'batch_valid': imgs_to_grid(torch.zeros((2, self.preview_width) + model.x_shape)),
            'batch_fullv': imgs_to_grid(torch.zeros((2, self.preview_width) + model.x_shape)),
        }
        for field, batch in (('valid', self.valid_batch), ('fixed', self.fixed_batch)):
            with EncapsulatedRandomState(42):
                x_org, x_distorted, x_restored, dirs = model.process_batch(model.split_batch(batch, self.n_batch_accumulation)[0])
            imgs[f'x_org_{field}'] = imgs_to_grid(x_org)
            imgs[f'x_distorted_{field}'] = imgs_to_grid(model.pc_model.restoration_model.distortion_model.naive_restore(x_distorted))
            imgs[f'x_restored_{field}'] = imgs_to_grid(x_restored)

        train_log_data['imgs'] = imgs

        ## Prepare summary
        ## ---------------
        dirs = dirs[:, :model.n_dirs]
        summary = torchinfo.summary(model['net'].net, input_data=(x_distorted[:1], x_restored[:1], dirs[:1]), depth=10, verbose=0, calc_feat=True, device=model.device)
        # summary.formatting.verbose = 2
        summary = str(summary)
        train_log_data['summary_feat'] = summary

        z = model.project_x(x_org, x_restored, dirs)
        with torch.no_grad():
            feat = model.gen_feat(x_distorted, x_restored, dirs, use_best=False)
        summary = torchinfo.summary(model.networks['net'].net, input_data=z[:1], depth=10, verbose=0, device=model.device, feat=feat[:1])
        # summary.formatting.verbose = 2
        summary = str(summary)
        train_log_data['summary_cond_ebm'] = summary

        ## Prepare figures
        ## ---------------
        figs = {}

        figs['mixing_coeff'] = go.Figure(
            data=[go.Scatter(mode='lines', y=model.gen_interp.mix_factors[:, 0].cpu().numpy(), name='m1'),
                  go.Scatter(mode='lines', y=model.gen_interp.mix_factors[:, 1].cpu().numpy(), name='m2'),
            ],
            layout=go.Layout(height=250, width=250, margin=dict(t=0, b=20, l=20, r=0)))

        figs['lr'] = go.Figure(
            data=[go.Scatter(mode='lines', x=[], y=[])],
            layout=go.Layout(yaxis_title='lr', xaxis_title='step', yaxis_type='log', height=400, width=550, margin=dict(t=0, b=20, l=20, r=0)),
        )

        figs['objective'] = go.Figure(
            data=[go.Scatter(mode='lines', x=[], y=[], name=title) for title in ('Train', 'Fixed', 'Validation', 'Full validation')],
            layout=go.Layout(yaxis_title='Objective', xaxis_title='step', height=400, width=550, margin=dict(t=0, b=20, l=20, r=0)),
        )

        figs['ce_nll'] = go.Figure(
            data=[go.Scatter(mode='lines', x=[], y=[], name=title) for title in ('Train', 'Fixed', 'Validation')],
            layout=go.Layout(yaxis_title='CE NLL', xaxis_title='step', height=400, width=550, margin=dict(t=0, b=20, l=20, r=0)),
        )

        figs['cd_nll'] = go.Figure(
            data=[go.Scatter(mode='lines', x=[], y=[], name=title) for title in ('Train', 'Fixed', 'Validation')],
            layout=go.Layout(yaxis_title='CD NLL', xaxis_title='step', height=400, width=550, margin=dict(t=0, b=20, l=20, r=0)),
        )

        figs['log_pz0'] = go.Figure(
            data=[go.Scatter(mode='lines', x=[], y=[], name=title) for title in ('Train', 'Fixed', 'Validation')],
            layout=go.Layout(yaxis_title='Log p(z0 org)', xaxis_title='step', height=400, width=550, margin=dict(t=0, b=20, l=20, r=0)),
        )

        figs['log_pz_org'] = go.Figure(
            data=[go.Scatter(mode='lines', x=[], y=[], name=title) for title in ('Train', 'Fixed', 'Validation')],
            layout=go.Layout(yaxis_title='Log p(z org)', xaxis_title='step', height=400, width=550, margin=dict(t=0, b=20, l=20, r=0)),
        )

        figs['log_pz_gen'] = go.Figure(
            data=[go.Scatter(mode='lines', x=[], y=[], name=title) for title in ('Train', 'Fixed', 'Validation')],
            layout=go.Layout(yaxis_title='Log p(z gen)', xaxis_title='step', height=400, width=550, margin=dict(t=0, b=20, l=20, r=0)),
        )

        figs['log_pz_diff'] = go.Figure(
            data=[go.Scatter(mode='lines', x=[], y=[], name=title) for title in ('Train', 'Fixed', 'Validation')],
            layout=go.Layout(yaxis_title='Log p(z diff)', xaxis_title='step', height=400, width=550, margin=dict(t=0, b=20, l=20, r=0)),
        )

        figs['step_prob'] = go.Figure(
            data=[go.Scatter(mode='lines', x=[], y=[], name=title) for title in ('Train', 'Fixed', 'Validation')],
            layout=go.Layout(yaxis_title='Step probability', xaxis_title='step', height=400, width=550, margin=dict(t=0, b=20, l=20, r=0)),
        )

        figs['n_mcmc_steps'] = go.Figure(
            data=[go.Scatter(mode='lines', x=[], y=[])],
            layout=go.Layout(yaxis_title='# MCMC steps', xaxis_title='step', height=400, width=550, margin=dict(t=0, b=20, l=20, r=0)),
        )

        figs['step_size'] = go.FigureWidget(
            data=[go.Scatter(mode='lines', x=np.arange(self.model.n_levels), y=self.model.step_size.cpu().numpy())],
            layout=go.Layout(xaxis_title='k', yaxis_title='step_size',
                             height=400, width=450, margin=dict(t=0, b=20, l=20, r=0),
                             showlegend=False,
            ))

        figs['nll_benchmark'] = go.Figure(
            data=[go.Scatter(mode='lines', x=[], y=[])],
            layout=go.Layout(yaxis_title='Benchmark -log(p(z))', xaxis_title='step', height=400, width=550, margin=dict(t=0, b=20, l=20, r=0)),
        )

        figs['correction_benchmark'] = go.Figure(
            data=[go.Scatter(mode='lines', x=[], y=[])],
            layout=go.Layout(yaxis_title='Benchmark correction', xaxis_title='step', height=400, width=550, margin=dict(t=0, b=20, l=20, r=0)),
        )

        # colorscale = np.linspace(1, 0, 256) ** 0.7
        # colorscale = [f'rgb({c:d},{c:d},{c:d})' for c in (colorscale * 255).astype(int)]

        # colorscale = mpl.cm.Reds(np.linspace(0, 1, 256))
        # colorscale = [f'rgb({c[0]:d},{c[1]:d},{c[2]:d})' for c in (colorscale * 255).astype(int)]

        for field in ('fixed', 'valid'):
            figs[f'dist_{field}'] = []
            for _ in range(7):
                fig = go.Figure(
                    data=[
                        go.Heatmap(
                            z=np.zeros(self.dist_calc_z.shape),
                            x=self.dist_calc_z.ranges[0], y=self.dist_calc_z.ranges[1],
                            zmin=0, zmax=1.,
                            colorscale='reds',
                            showscale=False,
                        ),
                        go.Contour(
                            z=np.zeros(self.dist_calc_z.shape),
                            x=self.dist_calc_z.ranges[0], y=self.dist_calc_z.ranges[1],
                            autocontour=False,
                            contours=dict(
                                start=0.4, size=0.1, end=0.91,
                                coloring='lines',
                            ),
                            showscale=False,
                            colorscale=['black', 'black'],
                            line_width=2,
                        ),
                        go.Scatter(
                            mode='markers',
                            x=[], y=[],
                            marker_color='DodgerBlue',
                            marker_size=5,
                        ),
                        go.Scatter(
                            mode='markers',
                            x=[], y=[],
                            marker_color='DarkRed',
                            marker_size=5,
                        ),
                    ],
                    layout=go.Layout(height=self.dist_calc_z.shape[1], width=self.dist_calc_z.shape[0],
                                    xaxis_range=(self.dist_calc_z.ranges[0].min().item(), self.dist_calc_z.ranges[0].max().item()),
                                    yaxis_range=(self.dist_calc_z.ranges[1].min().item(), self.dist_calc_z.ranges[1].max().item()),
                                    margin=dict(t=0, b=0, l=0, r=0),
                                    xaxis_showticklabels=False,
                                    yaxis_showticklabels=False,
                                    yaxis_scaleanchor='x',
                                    yaxis_scaleratio=1,
                                    # yaxis_autorange='reversed',
                                    ),
                )
                figs[f'dist_{field}'].append(fig)

        train_log_data['figs'] = figs

        return train_log_data

    def cat_logs(self, logs):
        log = {}
        for key in logs[0].keys():
            if key in ('x_org', 'x_distorted', 'x_restored', 'dirs', 'z', 'z_aug', 'z_aug_gen', 'k_gen',
                       'log_pz_org', 'log_pz_gen', 'log_pz_diff', 'step_prob',
                       ):
                log[key] = torch.cat([log_[key] for log_ in logs], dim=0)
            elif key in ('objective', 'ce_nll', 'cd_nll'):
                total = 0.
                total_n = 0
                for log_ in logs:
                    n = log_['x_org'].shape[0]
                    total += log_[key] * n
                    total_n += n
                log[key] = total / total_n
            elif key in ('n_mcmc_steps',):
                log[key] = logs[0][key]
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
        dirs = log['dirs']
        z = log['z']
        z_aug = log['z_aug']
        z_aug_gen = log['z_aug_gen']

        ce_nll_full = log['ce_nll']
        cd_nll_full = log['cd_nll']
        log_pz_org_full = log['log_pz_org']
        log_pz_gen_full = log['log_pz_gen']
        log_pz_diff_full = log['log_pz_diff']
        step_prob_full = log['step_prob']

        objective = log['objective']
        n_mcmc_steps = log['n_mcmc_steps']

        ce_nll = ce_nll_full.mean().item()
        cd_nll = cd_nll_full.mean().item()
        log_pz_org = log_pz_org_full.mean().item()
        log_pz_gen = log_pz_gen_full.mean().item()
        log_pz_diff = log_pz_diff_full.mean().item()
        step_prob = step_prob_full.mean().item()

        objective = objective.item()

        x_org = self.sample_preview_width(x_org)
        x_distorted = self.sample_preview_width(x_distorted)
        x_restored = self.sample_preview_width(x_restored)
        dirs = self.sample_preview_width(dirs)
        z = self.sample_preview_width(z)
        z_aug = self.sample_preview_width(z_aug)
        z_aug_gen = self.sample_preview_width(z_aug_gen)

        dirs_ = dirs / dirs.flatten(2).norm(dim=2)[:, :, None, None, None]
        x_org_proj = x_restored + torch.einsum('nichw,ni->nchw', dirs_, z)
        x_aug_org_proj = x_restored + torch.einsum('nichw,ni->nchw', dirs_, z_aug)
        x_aug_gen_proj = x_restored + torch.einsum('nichw,ni->nchw', dirs_, z_aug_gen)

        with torch.no_grad():
            feat = model.gen_feat(x_distorted, x_restored, dirs, use_best=False)
            log_pz0_full = model.log_pzk(z, feat, use_best=False)[:, 0]
        log_pz0 = log_pz0_full.mean().item()

        x_distorted = model.pc_model.restoration_model.distortion_model.naive_restore(x_distorted)

        batch_img = imgs_to_grid(torch.stack((x_org, x_distorted, x_restored, x_org_proj, x_aug_org_proj, x_aug_gen_proj)))

        ## Update console message
        ## ----------------------
        self.set_msg(field, f'{field}: step:{step:7d};   cd_nll: {cd_nll:9.4g},   ce_nll: {ce_nll:9.4g},   log_pz_diff: {log_pz_diff:9.4g}')

        ## Store log data
        ## --------------
        logs = self.train_log_data['logs']
        logs[f'step_{field}'].append(step)
        logs[f'lr_{field}'].append(lr)
        logs[f'objective_{field}'].append(objective)
        logs[f'ce_nll_{field}'].append(ce_nll)
        logs[f'cd_nll_{field}'].append(cd_nll)
        logs[f'log_pz0_{field}'].append(log_pz0)
        logs[f'log_pz_org_{field}'].append(log_pz_org)
        logs[f'log_pz_gen_{field}'].append(log_pz_gen)
        logs[f'log_pz_diff_{field}'].append(log_pz_diff)
        logs[f'step_prob_{field}'].append(step_prob)
        logs[f'n_mcmc_steps_{field}'].append(n_mcmc_steps)

        figs = self.train_log_data['figs']
        if field in ('fixed',):
            figs['lr'].data[0].update(x=logs['step_fixed'], y=logs['lr_fixed'])
            figs['n_mcmc_steps'].data[0].update(x=logs['step_fixed'], y=logs['n_mcmc_steps_fixed'])
        for key in ('objective', 'ce_nll', 'cd_nll', 'log_pz0', 'log_pz_org', 'log_pz_gen', 'log_pz_diff', 'step_prob'):
            figs[key].data[('train', 'fixed', 'valid', 'fullv').index(field)].update(x=logs[f'step_{field}'], y=logs[f'{key}_{field}'])

        imgs = self.train_log_data['imgs']
        imgs[f'batch_{field}'] = batch_img

        ## Log to weights and biases
        ## -------------------------
        if self.wandb is not None:
            if field in ('fixed',):
                self.wandb.log({
                    'train_params/lr': lr,
                    'cd_params/n_mcmc_steps': n_mcmc_steps,
                }, commit=False)

            self.wandb.log({
                f'objective1/objective_{field}': objective,
                f'nll/ce_nll_{field}': ce_nll,
                f'nll/cd_nll_{field}': cd_nll,
                f'log_p/log_pz0_{field}': log_pz0,
                f'log_p/log_pz_org_{field}': log_pz_org,
                f'log_p/log_pz_gen_{field}': log_pz_gen,
                f'log_p2/log_pz_diff_{field}': log_pz_diff,
                f'cd_params/step_prob_{field}': step_prob,
            }, commit=False)

            if field not in ('train',):
                self.wandb.log({
                    f'imgs/batch_imgs_{field}': wandb.Image(batch_img),
                }, commit=False)

    def post_log_valid(self, train_log, fixed_log, valid_log):
        model = self.model

        ## Store log data
        ## --------------
        figs = self.train_log_data['figs']
        figs['step_size'].data[0].update(y=model.step_size.cpu().numpy())

        ## Log to weights and biases
        ## -------------------------
        if self.wandb is not None:
            self.wandb.log({
                'cd_params/step_size': wandb.Plotly(figs['step_size']),
                }, commit=False)

    def benchmark(self):
        model = self.model

        # pad_z = lambda z: torch.cat((z, torch.zeros((z.shape[0], model.n_dirs - z.shape[1]), device=z.device)), dim=1)

        for field, batch in (('fixed', self.fixed_batch), ('valid', self.valid_batch)):
            with EncapsulatedRandomState(42):
                x_org, x_distorted, x_restored, dirs = model.process_batch(batch)
                if self.params['rand_rot_benchmark']:
                    dirs = model.rand_rotation(dirs)
                dirs = dirs[:, :model.n_dirs]
                z = model.project_x(x_org, x_restored, dirs)
            with torch.no_grad():
                feat = model.gen_feat(x_distorted, x_restored, dirs, use_best=False)
            for i, fig in enumerate(self.train_log_data['figs'][f'dist_{field}']):
                log_pz_func = lambda z_aug: model.log_pz(z_aug, feat[i, None], use_best=False)  # pylint: disable=cell-var-from-loop
                log_p, correction, contoure_map = self.dist_calc_z(log_pz_func)
                dist = torch.exp(log_p + correction).cpu().numpy()
                dist = dist / (dist.sum() + np.exp(self.dist_calc_z.log_grid_space))

                fig.data[0].update(z=dist.T, zmin=0, zmax=dist.max().item())
                fig.data[1].update(z=contoure_map.T.cpu().numpy())
                fig.data[2].update(x=z[i, None, 0].cpu().numpy(), y=z[i, None, 1].cpu().numpy())



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

        corrections_list = []
        log_pz_list = []
        for batch in tqdm.tqdm(dataloader, dynamic_ncols=True, leave=False):
            x_org, x_distorted, x_restored, dirs = model.process_batch(batch)
            if self.params['rand_rot_benchmark']:
                dirs = model.rand_rotation(dirs)
            dirs = dirs[:, :model.n_dirs]
            z = model.project_x(x_org, x_restored, dirs)
            with torch.no_grad():
                feat = model.gen_feat(x_distorted, x_restored, dirs, use_best=False)
            for i in range(z.shape[0]):
                log_pz_func = lambda z: model.log_pz(z, feat[i, None], use_best=False)  # pylint: disable=cell-var-from-loop
                _, correction, _ = self.dist_calc_z(log_pz_func)
                with torch.no_grad():
                    log_pz = model.log_pz(z[i, None], feat[i, None], use_best=False)[0] + correction
                corrections_list.append(correction)
                log_pz_list.append(log_pz.item())
        corrections_list = torch.tensor(correction)
        log_pz_list = torch.tensor(log_pz)

        log_pz = log_pz_list.mean().item()
        nll = -log_pz
        correction = corrections_list.mean().item()

        step = model.networks['net'].step
        logs = self.train_log_data['logs']
        logs['step_benchmark'].append(step)
        logs['nll_benchmark'].append(nll)
        logs['correction_benchmark'].append(correction)

        figs = self.train_log_data['figs']
        figs['nll_benchmark'].data[0].update(x=logs['step_benchmark'], y=logs['nll_benchmark'])
        figs['correction_benchmark'].data[0].update(x=logs['step_benchmark'], y=logs['correction_benchmark'])

        self.set_msg('fullv', f'Benchmark: step:{step:7d};   nll: {nll:9.4g},   correction: {correction:9.4g}')

        return nll

    def log_html(self):
        if self.folder_manager is not None:
            model = self.model

            fields = model.params.copy()
            fields.update(self.train_log_data['general'])
            fields['header'] = '<br>\n'.join(self.header + list(self.status_msgs.values()))
            fields['now'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fields['summary_feat'] = '<div style="white-space:pre;">' + self.train_log_data['summary_feat'] + '</div>'
            fields['summary_cond_ebm'] = '<div style="white-space:pre;">' + self.train_log_data['summary_cond_ebm'] + '</div>'

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

                    {nll_benchmark_fig}
                    {correction_benchmark_fig}
                    <br/>

                    {ce_nll_fig}
                    {cd_nll_fig}
                    {log_pz0_fig}
                    <br/>
                    {log_pz_org_fig}
                    {log_pz_gen_fig}
                    {log_pz_diff_fig}
                    <br/>
                    {step_size_fig}
                    {step_prob_fig}
                    {n_mcmc_steps_fig}
                    {lr_fig}
                </div>

                <div>
                    <h2>Benchmarks</h2>
                    {dist_fixed_fig}
                    <br/>
                    {dist_valid_fig}
                </div>

                <div>
                    <h2>Model</h2>

                    <h3>Train data</h3>
                    {x_org_fixed_img}
                    {x_distorted_fixed_img}
                    {x_restored_fixed_img}
                    <br>
                    {x_org_valid_img}
                    {x_distorted_valid_img}
                    {x_restored_valid_img}
                </div>
                
                <div>
                    <h2>Networks</h2>
                    <h3>Features network</h3>
                    {summary_feat}<br>
                    <h3>Features network</h3>
                    {summary_cond_ebm}<br>
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
                'model/x_restored_fixed': wandb.Image(self.train_log_data['imgs']['x_restored_fixed']),
                'model/x_org_vaild': wandb.Image(self.train_log_data['imgs']['x_org_vaild']),
                'model/x_distorted_vaild': wandb.Image(self.train_log_data['imgs']['x_distorted_vaild']),
                'model/x_restored_valid': wandb.Image(self.train_log_data['imgs']['x_restored_valid']),
                'model/mixing_coeff': wandb.Image(self.train_log_data['figs']['mixing_coeff']),
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


## Distributions
## =============
class GaussianModel(nn.Module):
    def __init__(self, mean=0., std=1.):
        super().__init__()

        mean = torch.as_tensor(mean).float()
        std = torch.as_tensor(std).float()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x):
        dim = np.prod(x.shape[1:])
        mahal_dist_squared = (((x - self.mean) / self.std) ** 2).view(x.shape[0], -1).sum(dim=1)
        log_px = -mahal_dist_squared / 2 - dim / 2 * torch.log(2 * np.pi * (self.std ** 2))
        return log_px

    def draw_samples(self, shape):
        x = torch.randn(shape, device=self.mean.device) * self.std + self.mean
        return x


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


class AdaIN(nn.Module):
    def __init__(self, in_channels, feat_channels):
        super().__init__()
        self.instance_norm = nn.InstanceNorm1d(in_channels)
        self.feat_linear = nn.Linear(feat_channels, in_channels * 2)

    def forward(self,x, w):
        x = self.instance_norm(x)
        scale, bias = self.feat_linear(w).chunk(2, dim=1)
        return (1 + scale) * x + bias


class TelescopicClass(nn.Module):
    def __init__(self, base_net, n_levels):

        super().__init__()

        self.base_net = base_net
        self.n_levels = n_levels

    def forward(self, *args, **kwargs):
        log_p_diff = self.base_net(*args, **kwargs)
        log_p_diff = torch.cat((log_p_diff, torch.zeros_like(log_p_diff[:, :1])), dim=1)
        log_p = log_p_diff.flip([1]).cumsum(dim=1).flip([1])
        log_p = F.interpolate(log_p[:, None, :], self.n_levels, mode='linear', align_corners=False)[:, 0, :]
        return log_p


class CondEBMWrapper(nn.Module):
    def __init__(
            self,
            feat_net,
            cond_ebm_net,
            n_levels,
            pre_net=None,
            pad_base_size=None,
            pre_pad_base_size=None,
            input_offset=None,
            input_scale=None,
            z_scale=None,
        ):

        super().__init__()

        self.pre_net = pre_net
        self.feat_net = feat_net
        self.cond_ebm_net = TelescopicClass(cond_ebm_net, n_levels=n_levels)
        self.pad_base_size = pad_base_size
        self.pre_pad_base_size = pre_pad_base_size
        self.input_offset = input_offset
        self.input_scale = input_scale
        self.z_scale = z_scale

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

    def calc_feat(self, x_distorted, x_restored, dirs):
        if self.input_offset is not None:
            x_distorted = x_distorted - self.input_offset
            x_restored = x_restored - self.input_offset
        if self.input_scale is not None:
            x_distorted = x_distorted / self.input_scale
            x_restored = x_restored / self.input_scale
        dirs = dirs / dirs.flatten(2).norm(dim=2)[:, :, None, None, None]

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
        x = torch.cat((x, x_restored, dirs.flatten(1, 2)), dim=1)

        padding = self._get_padding(x, self.pad_base_size)
        if padding is not None:
            x = F.pad(x, tuple(padding))

        feat = self.feat_net(x)
        if padding is not None:
            feat = F.pad(feat, tuple(-padding))  # pylint: disable=invalid-unary-operand-type

        return feat

    def calc_ebm(self, z, feat):
        if self.z_scale is not None:
            z = z / self.z_scale
        log_p = self.cond_ebm_net(z, feat)
        return log_p

    def forward(self, *args, calc_feat=False, **kwrags):
        if calc_feat:
            return self.calc_feat(*args, **kwrags)
        else:
            return self.calc_ebm(*args, **kwrags)


## Networks
## ========
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


class ResNetA(nn.Module):
    ## Based upon https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    def __init__(
            self,
            in_channels=3,
            out_channels=1000,
            n_blocks_list=(2, 2, 2, 2),
            channels_list=(64, 128, 256, 512),
            in_stride=1,
            n_norm_groups=4,
        ):

        super().__init__()

        ch = in_channels
        ch2 = channels_list[0]
        layers = [
            nn.Conv2d(ch, ch2, kernel_size=3 + 2 * in_stride, stride=in_stride, padding=1 + in_stride),
            nn.ReLU(inplace=True),
        ]
        ch = ch2
        for i_level, (n_blocks, ch2) in enumerate(zip(n_blocks_list, channels_list)):
            for _ in range(n_blocks):
                layers += [
                    ShortcutBlock(nn.Sequential(
                            nn.Conv2d(ch, ch2, kernel_size=3, padding=1, bias=False),
                            nn.GroupNorm(n_norm_groups, ch2),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(ch2, ch2, kernel_size=3, padding=1, bias=False),
                            zero_weights(nn.GroupNorm(n_norm_groups, ch2)),
                        ),
                        shortcut=nn.Sequential(
                            nn.Conv2d(ch, ch2, kernel_size=1) if (ch2 != ch) else nn.Identity(),
                        ),
                    ),
                    nn.ReLU(inplace=True),
                ]
                ch = ch2
            if i_level != (len(n_blocks_list) - 1):
                layers += [
                    nn.AvgPool2d(2),
                ]
        layers += [
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(ch, out_channels),
        ]

        self.base = nn.Sequential(*layers)

    def forward(self, x):
        x = self.base(x)
        return x


class ResNetB(nn.Module):
    ## Based upon https://github.dev/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5
    def __init__(
            self,
            in_channels=3,
            out_channels=1000,
            n_blocks_list=(3, 4, 6, 3),
            channels_list=(64, 128, 256, 512),
            expansion=4,
            n_norm_groups=4,
        ):

        super().__init__()

        layers = []

        ch = in_channels
        ch_ = channels_list[0]
        layers += [
            nn.Conv2d(ch, ch_, kernel_size=7, stride=1, padding=3),
            nn.GroupNorm(n_norm_groups, ch_),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(3, stride=2, padding=1),
            nn.MaxPool2d(2, 2),
        ]
        ch = ch_
        for i_level, (n_blocks, ch_) in enumerate(zip(n_blocks_list, channels_list)):
            for i_block in range(n_blocks):
                stride = 2 if ((i_level != 0) and (i_block == 0)) else 1
                layers += [
                    ShortcutBlock(
                        nn.Sequential(
                            nn.Conv2d(ch, ch_, kernel_size=1, bias=False),
                            nn.GroupNorm(n_norm_groups, ch_),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(ch_, ch_, kernel_size=3, padding=1, bias=False, stride=stride),
                            nn.GroupNorm(n_norm_groups, ch_),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(ch_, ch_ * expansion, kernel_size=1, bias=False),
                            zero_weights(nn.GroupNorm(n_norm_groups, ch_ * expansion)),
                        ),
                        shortcut=nn.Sequential(
                            nn.Conv2d(ch, ch_ * expansion, kernel_size=1, stride=stride) if ((ch_ * expansion != ch) or stride != 1) else nn.Identity(),
                            nn.GroupNorm(n_norm_groups, ch_ * expansion),
                        ),
                    ),
                    # nn.ReLU(inplace=True),
                ]
                ch = ch_ * expansion
            # if i_level != (len(n_blocks) - 1):
            #     layers += [
            #         nn.AvgPool2d(2),
            #     ]
        layers += [
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(ch, out_channels),
        ]

        self.base = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.base(x)
        return x


## Half DDPM
class ResNet4(nn.Module):
    def __init__(
            self,
            in_channels=3,
            out_channels=None,
            channels_list=(128, 128, 256, 256, 512, 512, 512),
            downsample_list=(False, True, True, True, True, True, False),
            attn_list=(False, False, False, False, True, False, True),
            n_blocks=2,
            n_groups=8,
            attn_heads=1,
        ):

        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        ch = in_channels
        layers = []

        ch_ = channels_list[0]
        layers.append(nn.Conv2d(ch, ch_, 3, padding=1))
        ch = ch_

        for i_level in range(len(channels_list)):
            ch_ = channels_list[i_level]
            downsample = downsample_list[i_level]
            attn = attn_list[i_level]
            layers_ = []

            if downsample:
                layers.append(nn.Conv2d(ch, ch, 3, padding=1, stride=2))

            for _ in range(n_blocks):
                layers.append(ResBlock(ch, ch_, n_groups=n_groups))
                ch = ch_
                if attn:
                    layers_.append(Attention(ch, heads=attn_heads))

        layers.append(ResBlock(ch, ch, n_groups=n_groups))
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(ch, out_channels))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x


class CondMLP(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            feat_channels,
            hidden_channels=128,
            n_blocks=3,
        ):

        super().__init__()

        self.blocks = nn.ModuleList()
        self.adain_blocks = nn.ModuleList()
        ch = in_channels

        for _ in range(n_blocks - 1):
            block = nn.ModuleDict(dict(
                main = nn.Sequential(
                    nn.Linear(ch, hidden_channels),
                    # nn.ReLU(inplace=True),
                    nn.SiLU(inplace=True),
                ),
                adain = AdaIN(hidden_channels, feat_channels),
            ))
            self.blocks.append(block)
            ch = hidden_channels

        self.out_block = nn.Sequential(
            nn.Linear(ch, out_channels),
        )

    def forward(self, x, feat):
        for block in self.blocks:
            x = block['main'](x)
            x = block['adain'](x, feat)
        x = self.out_block(x)
        return x


## HMC Sampler
## ===========
class HMCSampler:
    def __init__(self, log_p_func, x, step_size=1):

        self._log_p_func = log_p_func

        self.x_shape = x.shape
        self.n_samples = self.x_shape[0]
        x = x.view(self.n_samples, -1)

        ## X, energy and grad
        ## ------------------
        self._x = x.detach().requires_grad_(True)
        self.log_p = None
        self.grad = None
        self._update_grad_and_prob()
        self._momentum = torch.empty_like(self.grad)

        ## Reference (stores the state before the step)
        ## --------------------------------------------
        self._x_ref = torch.empty_like(self._x)
        self._log_p_ref = torch.empty_like(self.log_p)
        self._grad_ref = torch.empty_like(self.grad)
        self._momentum_ref = torch.empty_like(self._momentum)

        self.step_size = torch.as_tensor(step_size).clone().to(self._x.device)
        if self.step_size.ndim == 0:
            self.step_size = self.step_size * torch.ones_like(self.log_p)

        self._last_stats = None

    @property
    def x(self):
        return self._x.detach().view(self.x_shape)

    def _update_grad_and_prob(self):
        with torch.enable_grad():
            x = self._x
            log_p = self._log_p_func(x.view(self.x_shape))
            self.grad = torch.autograd.grad(log_p.sum(), x)[0]
        self.log_p = log_p.detach()

    def init_step(self):
        self._momentum.normal_(0, 1)
        self._x_ref.copy_(self._x)
        self._log_p_ref.copy_(self.log_p)
        self._grad_ref.copy_(self.grad)
        self._momentum_ref.copy_(self._momentum)

    def leap(self):
        with torch.no_grad():
            self._momentum.addcmul_(self.step_size[:, None], self.grad, value=0.5)
            self._x.addcmul_(self.step_size[:, None], self._momentum)
            self._update_grad_and_prob()
            self._momentum.addcmul_(self.step_size[:, None], self.grad, value=0.5)

    def finalize_step(self, mh_adjust=True):
        log_trans_prob = -(self._momentum_ref ** 2).sum(dim=1) / 2
        log_trans_prob_inv = -(self._momentum ** 2).sum(dim=1) / 2

        trans_log_prob_diff = log_trans_prob - log_trans_prob_inv
        log_accept_prob = self.log_p - self._log_p_ref - trans_log_prob_diff
        step_prob = torch.exp(log_accept_prob.clamp(max=0))
        step_prob[torch.isnan(step_prob)] = 0
        accepted = torch.rand_like(step_prob) < step_prob

        if mh_adjust:
            self._x.data[~accepted] = self._x_ref[~accepted]
            self.log_p[~accepted] = self._log_p_ref[~accepted]
            self.grad[~accepted] = self._grad_ref[~accepted]
            trans_log_prob_diff = self.log_p - self._log_p_ref

        stats = {
            'step_prob': step_prob,
            'accepted': accepted,
            'trans_log_prob_diff': trans_log_prob_diff,
        }
        self._last_stats = stats

        return stats

    def step(self, n_leaps=1, mh_adjust=True):
        self.init_step()
        for _ in range(n_leaps):
            self.leap()
        stats = self.finalize_step(mh_adjust=mh_adjust)
        return self._x.detach(), stats

    @staticmethod
    def adjust_step_static_in_place(step_size, indices, step_adj_factor=1.1):
        step_size[indices] *= step_adj_factor
        step_size[~indices] /= step_adj_factor
        return step_size

    def adjust_step(self, target_prob=0.8, step_adj_factor=1.1):
        indices = self._last_stats['step_prob'] > target_prob
        self.adjust_step_static_in_place(self.step_size, indices, step_adj_factor=step_adj_factor)


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
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(self._net_ddp.parameters(), **optimizer_params)
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


class DistCalc:
    def __init__(self, log_p_func, ranges, batch_size=None, device='cpu'):

        self.log_p_func = log_p_func
        self.ranges = ranges

        self.grid = torch.stack(torch.meshgrid(*ranges, indexing='ij'), dim=0).to(device)
        self.shape = self.grid.shape[1:]
        self._flat_grid = self.grid.flatten(1).T
        self.log_grid_space = np.sum([np.log(range_[1] - range_[0]) for range_ in self.ranges])
        if batch_size is None:
            batch_size = self._flat_grid.shape[0]
        self._x_splits = torch.split(self._flat_grid, batch_size)

    def __call__(self, log_p_func=None):
        if log_p_func is None:
            log_p_func = self.log_p_func
        with torch.no_grad():
            log_p = torch.cat([log_p_func(x) for x in self._x_splits], dim=0)
        correction = -(torch.logsumexp(log_p, dim=0) + self.log_grid_space).item()

        indices = torch.argsort(log_p, descending=True)
        contoure_map = log_p.clone()
        contoure_map[indices] = torch.exp(log_p + correction + self.log_grid_space)[indices].cumsum(dim=0)

        log_p = log_p.view(*self.shape)
        contoure_map = contoure_map.view(*self.shape)

        return log_p, correction, contoure_map


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
