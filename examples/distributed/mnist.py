"""
An example on MNIST handwritten digits recognition
This script uses DistributedDataParallel

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from pocket.models import LeNet
from pocket.core import DistributedLearningEngine

def set_seed(seed=0, cuda_deterministic=False):
    """
    给不同的进程分配不同的、固定的随机数种子

    [OPTIONAL ARGS]
        seed(int): 随机数种子
        cuda_deterministic(bool): 是否使cuda计算结果具有确定性，默认为False，
            因为效率较低，通常不使用。

    """
    seed = seed + dist.get_rank()
    torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)

    # 使cuda结果在多次运行时是确定性的
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def main(rank, world_size):
    # 初始化进程组
    dist.init_process_group(
        backend="nccl",  # GPU通信的库
        init_method="env://",  # 表示使用读取环境变量的方法初始化互相通讯的进程
        world_size=world_size, # 总的进程数量，每个进程对应一张GPU
        rank=rank  # 当前进程的全局id，范围是[0, args.world_size-1)
    )

    # 设置随机数种子
    set_seed(42)

    # Initialize network
    net = LeNet()  # 就是普通的nn.Module模型即可
    # Initialize loss function
    criterion = torch.nn.CrossEntropyLoss()
    # Prepare dataset
    trainset = datasets.MNIST('../data', train=True, download=True,
        transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    )

    # Prepare sampler
    # num_replicas表示参与分布式训练的进程的数量
    # rank表示当前进程的全局编号
    # 默认shuffle=True
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, num_replicas=world_size, rank=rank
    )

    # Prepare dataloader
    # 指定了sampler参数就无需指定shuffle参数。
    # pin_memory=True表示使用锁页内存(Pinned Memory)，从而减少CPU拷贝内存的开销，
    # 锁页内存是相对分页内存而言的，具体可见https://zhuanlan.zhihu.com/p/561544545
    train_loader = DataLoader(
        trainset, batch_size=128,
        num_workers=2, pin_memory=True, sampler=train_sampler)
    # Intialize learning engine and start training
    engine = DistributedLearningEngine(
        net, criterion, train_loader,
        find_unused_parameters=True  # 在本例中，该参数可以为False
    )

    # 在这里可以调整engine的一些参数，比如engine默认使用MultiStepLR，
    # 可以将其改为StepLR，具体可见UPT的代码

    # Train the network for one epoch with default optimizer option
    # Checkpoints will be saved under ./checkpoints by default, containing 
    # saved model parameters, optimizer statistics and progress
    # 训练5个epoch
    engine(5)

    # Clean up 销毁进程组
    dist.destroy_process_group()

if __name__ == '__main__':

    # Number of GPUs to run the experiment with
    # 使用多少张GPU进行训练
    WORLD_SIZE = 1

    # 使用环境变量指定进程间的通讯方式
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8888"
    mp.spawn(main, nprocs=WORLD_SIZE, args=(WORLD_SIZE,))

    # Sample output
    """
    Epoch [1/5], Iter. [100/235], Loss: 2.2968, Time[Data/Iter.]: [3.31s/6.57s]
    Epoch [1/5], Iter. [200/235], Loss: 2.2767, Time[Data/Iter.]: [2.30s/5.07s]
    Epoch [2/5], Iter. [065/235], Loss: 2.2289, Time[Data/Iter.]: [3.13s/5.50s]
    Epoch [2/5], Iter. [165/235], Loss: 2.0091, Time[Data/Iter.]: [2.11s/4.99s]
    Epoch [3/5], Iter. [030/235], Loss: 1.0353, Time[Data/Iter.]: [3.21s/5.81s]
    Epoch [3/5], Iter. [130/235], Loss: 0.5111, Time[Data/Iter.]: [2.59s/5.80s]
    Epoch [3/5], Iter. [230/235], Loss: 0.4194, Time[Data/Iter.]: [2.32s/5.14s]
    Epoch [4/5], Iter. [095/235], Loss: 0.3574, Time[Data/Iter.]: [3.01s/5.64s]
    Epoch [4/5], Iter. [195/235], Loss: 0.3105, Time[Data/Iter.]: [2.39s/4.99s]
    Epoch [5/5], Iter. [060/235], Loss: 0.2800, Time[Data/Iter.]: [3.23s/6.19s]
    Epoch [5/5], Iter. [160/235], Loss: 0.2575, Time[Data/Iter.]: [2.44s/4.67s]
    """
