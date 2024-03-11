"""
Distributed learning engines based on torch.distributed

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import time
import torch
import torch.distributed as dist

from torch.nn import Module
from typing import Callable, Iterable, Optional

from ..ops import relocate_to_cuda
from ..utils import SyncedNumericalMeter

from .engines import State

class DistributedLearningEngine(State):
    r"""
    Base class for distributed learning engine
    仅支持单机多卡，不支持多机多卡。
    该类的使用方法见examples/distributed/mnist.py

    Arguments: 各参数的含义与engines.LearningEngine一致，不再赘述。

    [REQUIRED ARGS]
        net(Module): The network to be trained
        criterion(callable): Loss function
        train_loader(iterable): Dataloader for training set, with batch input in the
            format [INPUT_1, ..., INPUT_N, LABELS]. Each element should take one of 
            the following forms: Tensor, list[Tensor], dict[Tensor]

    [OPTIONAL ARGS]
        device(int or torch.device): CUDA device to be used for training.
            如果指定了设备，则仅在该设备上训练
        optim(str): Optimizer to be used. Choose between 'SGD' and 'Adam'
        optim_params(dict): Parameters for the selected optimizer
        optim_state_dict(dict): Optimizer state dict to be loaded
        lr_scheduler(bool): If True, use MultiStepLR as the learning rate scheduler
        lr_sched_params(dict): Parameters for the learning rate scheduler
        verbal(bool): If True, print statistics every fixed interval
        print_interval(int): Number of iterations to print statistics
        cache_dir(str): Directory to save checkpoints
        find_unused_parameters(bool): 默认为False，即每个节点都计算了所有参数的梯度。
            具体见：https://zhuanlan.zhihu.com/p/592515484

    [各方法的生命周期，根据实际需要进行覆盖]
        self._on_start()                        # 在所有的epoch开始前被调用
        for epoch
            self._on_start_epoch()              # 在每个epoch开始前被调用。通常用于sampler.set_epoch()
            for mini-batch
                self._on_start_iteration()      # 在每个iteration开始前被调用。通常在这里将数据迁移到指定的设备
                self._on_each_iteration()       # 前向传播、反向传播等过程。
                self._on_end_iteration()        # 执行完每个iteration后被调用。
            self._on_end_epoch()                # 执行完每个epoch后被调用。通常在这里保存checkpoint，执行lr_scheduler
        self._on_end()                          # 执行完所有的epoch后被调用。

        注：iteration指的是训练每个mini-batch的过程，epoch指的是训练整个数据集的过程

    [self._state中存储的状态信息]
        net: 模型
        optimizer: 优化器
        scaler: GradScalar实例，用于进行损失缩放
        epoch：当前是第几个epoch
        iteration：当前是第几个mini-batch
        lr_scheduler: 学习率调整器

		# NumericalMeter是对deque的封装，用于记录最近的print_interval次iteration产生的相关信息，以便进行日志输出
        running_loss(NumericalMeter)：每个iteration(mini-batch)的损失
        t_data(NumericalMeter)：加载一个mini-batch的数据所需的时间
        t_iteration(NumericalMeter)：执行一个mini-batch的时间（包括加载数据、推理、反向传播等的时间）

        # 下面这三个属性是针对每个iteration(mini-batch)的
        inputs: 当前iteration下，模型的输入
        targets：当前iteration下，样本的ground truth
        loss: 当前iteration下，损失函数值
    """
    def __init__(self,
            net: Module, criterion: Callable, train_loader: Iterable,
            device: Optional[int] = None, optim: str = 'SGD',
            optim_params: Optional[dict] = None, optim_state_dict: Optional[dict] = None,
            lr_scheduler: bool = False, lr_sched_params: Optional[dict] = None,
            verbal: bool = True, print_interval: int = 100, use_amp: bool = True,
            find_unused_parameters: bool = False, cache_dir: str = './checkpoints'
        ):

        super().__init__()
        if not dist.is_available():
            raise AssertionError("Torch not compiled with distributed package")
        if not dist.is_initialized():
            raise AssertionError("Default process group has not been initialized")

        self._dawn = time.time()

        # 获取当前进程的全局编号，从0开始，0表示master进程
        self._rank = dist.get_rank()
        # 总的进程数量，一般情况下一个进程管理一张卡
        self._world_size = dist.get_world_size()
        # 这里使用进程的全局编号获取设备，所以本代码只适用于`单机多卡`的情况。
        self._device = torch.device(device) if device is not None else torch.device(self._rank)
        # 设置当前进程默认使用的GPU，后面的xxx.cuda()就会将xxx迁移到该GPU上
        # NOTE Removing this line causes non-master subprocesses stuck at data loading
        torch.cuda.set_device(self._device)

        self._criterion = criterion if not isinstance(criterion, torch.nn.Module) \
            else criterion.cuda()
        self._train_loader = train_loader
        self._verbal = verbal
        self._print_interval = print_interval
        self._use_amp = use_amp
        self._cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # Relocate model to designated device
        net.cuda()

        # Initialize optimizer
        net_params = [p for p in net.parameters() if p.requires_grad]
        if optim_params is None:
            optim_params = {
                    'lr': 0.001,
                    'momentum': 0.9,
                    'weight_decay': 5e-4
            } if optim == 'SGD' else {'lr': 0.001, 'weight_decay': 5e-4}
        self._state.optimizer = eval(f'torch.optim.{optim}')(net_params, **optim_params)
        # Load optimzer state dict if provided
        if optim_state_dict is not None:
            self._state.optimizer.load_state_dict(optim_state_dict)
            # Relocate optimizer state to designated device
            for state in self._state.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

        # 自PyTorch1.9.0开始，DistributedDataParallel的device_ids参数最多只能传入1个设备，
        # 即仅支持1个进程对应1个设备，不支持1个进程对应多个设备，具体可见：https://github.com/pytorch/pytorch/releases/tag/v1.9.0
        # 这里没有指定output_device参数，默认output_device=device_ids[0]，在这里即为output_device=self._device.
        # （output_device参数的具体含义是什么？为什么与device_ids相同？不管了，等用到再说，多半用不到）
        self._state.net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[self._device],
            find_unused_parameters=find_unused_parameters
        )

        # Initialise gradient scaler
        self._state.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self._state.epoch = 0
        self._state.iteration = 0

        # Initialize learning rate scheduler
        lr_sched_params = {
                'milestones': [50,100],
                'gamma': 0.1
            } if lr_sched_params is None else lr_sched_params
        self._state.lr_scheduler = None if not lr_scheduler \
            else torch.optim.lr_scheduler.MultiStepLR(self._state.optimizer, **lr_sched_params)

        self._state.running_loss = SyncedNumericalMeter(maxlen=print_interval)
        # Initialize timers
        self._state.t_data = SyncedNumericalMeter(maxlen=print_interval)
        self._state.t_iteration = SyncedNumericalMeter(maxlen=print_interval)

    def __call__(self, n: int) -> None:
        self.epochs = n
        # Train for a specified number of epochs
        self._on_start()
        for _ in range(n):
            self._on_start_epoch()
            timestamp = time.time()
            for batch in self._train_loader:
                self._state.inputs = batch[:-1]
                self._state.targets = batch[-1]
                self._on_start_iteration()
                self._state.t_data.append(time.time() - timestamp)

                self._on_each_iteration()
                self._state.running_loss.append(self._state.loss.item())
                self._on_end_iteration()
                self._state.t_iteration.append(time.time() - timestamp)
                timestamp = time.time()
                
            self._on_end_epoch()
        self._on_end()

    def _on_start(self):
        pass

    def _on_end(self):
        pass

    def _on_start_epoch(self):
        self._state.epoch += 1
        # Force network mode
        self._state.net.train()
        # Update random seeds for sampler
        self._train_loader.sampler.set_epoch(self._state.epoch)

    def _on_end_epoch(self):
        # Save checkpoint in the master process
        if self._rank == 0:  # 在masker进程上保存checkpoint
            self.save_checkpoint()
        if self._state.lr_scheduler is not None:
            self._state.lr_scheduler.step()

    def _on_start_iteration(self):
        self._state.iteration += 1
        self._state.inputs = relocate_to_cuda(self._state.inputs, non_blocking=True)
        self._state.targets = relocate_to_cuda(self._state.targets, non_blocking=True)

    def _on_end_iteration(self):
        # Print stats in the master process
        if self._verbal and self._state.iteration % self._print_interval == 0:
            self._print_statistics()

    def _on_each_iteration(self):
        with torch.cuda.amp.autocast(enabled=self._use_amp):
            self._state.output = self._state.net(*self._state.inputs)
            self._state.loss = self._criterion(self._state.output, self._state.targets)
        self._state.scaler.scale(self._state.loss).backward()
        self._state.scaler.step(self._state.optimizer)
        self._state.scaler.update()
        self._state.optimizer.zero_grad(set_to_none=True)

    def _print_statistics(self):
        running_loss = self._state.running_loss.mean()
        t_data = self._state.t_data.sum() / self._world_size
        t_iter = self._state.t_iteration.sum() / self._world_size

        # Print stats in the master process
        if self._rank == 0:
            num_iter = len(self._train_loader)
            n_d = len(str(num_iter))
            print(
                "Epoch [{}/{}], Iter. [{}/{}], "
                "Loss: {:.4f}, "
                "Time[Data/Iter.]: [{:.2f}s/{:.2f}s]".format(
                self._state.epoch, self.epochs,
                str(self._state.iteration - num_iter * (self._state.epoch - 1)).zfill(n_d),
                num_iter, running_loss, t_data, t_iter
            ))
        self._state.t_iteration.reset()
        self._state.t_data.reset()
        self._state.running_loss.reset()

    def save_checkpoint(self) -> None:
        """Save a checkpoint of the model state"""
        checkpoint = {
            'iteration': self._state.iteration,
            'epoch': self._state.epoch,
            'model_state_dict': self._state.net.module.state_dict(),
            'optim_state_dict': self._state.optimizer.state_dict(),
            'scaler_state_dict': self._state.scaler.state_dict()
        }
        if self._state.lr_scheduler is not None:
            checkpoint['scheduler_state_dict'] = self._state.lr_scheduler.state_dict()
        # Cache the checkpoint
        torch.save(checkpoint, os.path.join(
            self._cache_dir,
            'ckpt_{:05d}_{:02d}.pt'.format(self._state.iteration, self._state.epoch)
        ))
