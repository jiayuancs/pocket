"""
Meters for the purpose of statistics tracking

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import time
import torch
import multiprocessing

from torch import Tensor
from collections import deque
from typing import Optional, Iterable, Any, List, Union, Tuple
from ..ops import to_tensor

__all__ = [
    'Meter', 'NumericalMeter', 'HandyTimer',
    'AveragePrecisionMeter', 'DetectionAPMeter'
]

def div(numerator: Tensor, denom: Union[Tensor, int, float]) -> Tensor:
    """Handle division by zero"""
    if type(denom) in [int, float]:
        if denom == 0:
            return torch.zeros_like(numerator)
        else:
            return numerator / denom
    elif type(denom) is Tensor:
        zero_idx = torch.nonzero(denom == 0).squeeze(1)
        denom[zero_idx] += 1e-8
        return numerator / denom
    else:
        raise TypeError("Unsupported data type ", type(denom))

class Meter:
    """
    Base class
    记录评价指标值的工具，基于双端队列deque，当想deque中添加超过maxlen个数的元素时，
    最左端的元素会被弹出。

    示例：
        # 最大长度为3
        Meter meter(3)

        meter.append(1)
        meter.append(2)
        meter.append(3)  # 此时容器的内容为：[1,2,3]

        # 再添加一个元素，超出最大长度，导致最左端的元素被弹出
        meter.append(1)  # 此时容器的内容为：[2,3,4]
    """
    def __init__(self, maxlen: Optional[int] = None) -> None:
        self._deque = deque(maxlen=maxlen)
        self._maxlen = maxlen

    def __len__(self) -> int:
        return len(self._deque)

    def __iter__(self) -> Iterable:
        return iter(self._deque)

    def __getitem__(self, i: int) -> Any:
        return self._deque[i]

    def __repr__(self) -> str:
        reprstr = self.__class__.__name__ + '('
        reprstr += 'maxlen='
        reprstr += str(self._maxlen)
        reprstr += ')'
        return reprstr

    def reset(self) -> None:
        """Reset the meter(清空)"""
        self._deque.clear()

    def append(self, x: Any) -> None:
        """Append an element"""
        self._deque.append(x)

    def sum(self):
        """Return the sum of all elements"""
        raise NotImplementedError

    def mean(self):
        """Return the mean"""
        raise NotImplementedError

    def max(self):
        """Return the minimum element"""
        raise NotImplementedError

    def min(self):
        """Return the maximum element"""
        raise NotImplementedError

    @property
    def items(self) -> List[Any]:
        """Return the content。
        返回一个列表，包含所有元素
        """
        return [item for item in self._deque]

class NumericalMeter(Meter):
    """
    Meter class with numerals as elements
    记录数值型的评价指标值的容器，仅支持存储int和float
    """
    VALID_TYPES = [int, float]
    
    def __init__(self, maxlen: Optional[int] = None) -> None:
        super().__init__(maxlen=maxlen)

    def append(self, x: Union[int, float]) -> None:
        if type(x) in self.VALID_TYPES:
            super().append(x)
        else:
            raise TypeError("Given element \'{}\' is not a numeral".format(x))

    def sum(self) -> Union[int, float]:
        if len(self._deque):
            return sum(self._deque)
        else:
            raise ValueError("Cannot take sum. The meter is empty.")

    def mean(self) -> float:
        if len(self._deque):
            return sum(self._deque) / len(self._deque)
        else:
            raise ValueError("Cannot take mean. The meter is empty.")

    def max(self) -> Union[int, float]:
        if len(self._deque):
            return max(self._deque)
        else:
            raise ValueError("Cannot take max. The meter is empty.")

    def min(self) -> Union[int, float]:
        if len(self._deque):
            return min(self._deque)
        else:
            raise ValueError("Cannot take min. The meter is empty.")

class HandyTimer(NumericalMeter):
    """
    A timer class that tracks a sequence of time
    """
    def __init__(self, maxlen: Optional[int] = None):
        super().__init__(maxlen=maxlen)

    def __enter__(self) -> None:
        self._timestamp = time.time()

    def __exit__(self, type, value, traceback) -> None:
        self.append(time.time() - self._timestamp)

class AveragePrecisionMeter:
    """
    Meter to compute average precision

    Arguments:
        num_gt(iterable): Number of ground truth instances for each class. When left
            as None, all positives are assumed to have been included in the collected
            results. As a result, full recall is guaranteed when the lowest scoring
            example is accounted for.
        algorithm(str, optional): AP evaluation algorithm
            '11P': 11-point interpolation algorithm prior to voc2010
            'INT': Interpolation algorithm with all points used in voc2010
            'AUC': Precisely as the area under precision-recall curve
        chunksize(int, optional): The approximate size the given iterable will be split
            into for each worker. Use -1 to make the argument adaptive to iterable size
            and number of workers
        precision(int, optional): Precision used for float-point operations. Choose
            amongst 64, 32 and 16. Default is 64
        output(tensor[N, K], optinoal): Network outputs with N examples and K classes
        labels(tensor[N, K], optinoal): Binary labels

    Usage:
        
    (1) Evalute AP using provided output scores and labels

        >>> # Given output(tensor[N, K]) and labels(tensor[N, K])
        >>> meter = pocket.utils.AveragePrecisionMeter(output=output, labels=labels)
        >>> ap = meter.eval(); map_ = ap.mean()

    (2) Collect results on the fly and evaluate AP

        >>> meter = pocket.utils.AveragePrecisionMeter()
        >>> # Compute output(tensor[N, K]) during forward pass
        >>> meter.append(output, labels)
        >>> ap = meter.eval(); map_ = ap.mean()
        >>> # If you are to start new evaluation and want to reset the meter
        >>> meter.reset()

    """
    def __init__(self, num_gt: Optional[Iterable] = None,
            algorithm: str = 'AUC', chunksize: int = -1,
            precision: int = 64,
            output: Optional[Tensor] = None,
            labels: Optional[Tensor] = None) -> None:
        self._dtype = eval('torch.float' + str(precision))
        self.num_gt = torch.as_tensor(num_gt, dtype=self._dtype) \
            if num_gt is not None else None
        self.algorithm = algorithm
        self._chunksize = chunksize
        
        is_none = (output is None, labels is None)
        if is_none == (True, True):
            self._output = torch.tensor([], dtype=self._dtype)
            self._labels = torch.tensor([], dtype=self._dtype)
        elif is_none == (False, False):
            self._output = output.detach().cpu().to(self._dtype)
            self._labels = labels.detach().cpu().to(self._dtype)
        else:
            raise AssertionError("Output and labels should both be given or None")

        self._output_temp = [torch.tensor([], dtype=self._dtype)]
        self._labels_temp = [torch.tensor([], dtype=self._dtype)]

    @staticmethod
    def compute_per_class_ap_as_auc(tuple_: Tuple[Tensor, Tensor]) -> Tensor:
        """
        Arguments: 
            tuple_(Tuple[Tensor, Tensor]): precision and recall
        Returns:
            ap(Tensor[1])
        """
        prec, rec = tuple_
        ap = 0
        max_rec = rec[-1]
        for idx in range(prec.numel()):
            # Stop when maximum recall is reached
            if rec[idx] >= max_rec:
                break
            d_x = rec[idx] - rec[idx - 1]
            # Skip when negative example is registered
            if d_x == 0:
                continue
            ap +=  prec[idx] * rec[idx] if idx == 0 \
                else 0.5 * (prec[idx] + prec[idx - 1]) * d_x
        return ap

    @staticmethod
    def compute_per_class_ap_with_interpolation(tuple_: Tuple[Tensor, Tensor]) -> Tensor:
        """
        Arguments:
            tuple_(Tuple[Tensor, Tensor]): precision and recall
        Returns:
            ap(Tensor[1])
        """
        prec, rec = tuple_
        ap = 0
        max_rec = rec[-1]
        for idx in range(prec.numel()):
            # Stop when maximum recall is reached
            if rec[idx] >= max_rec:
                break
            d_x = rec[idx] - rec[idx - 1]
            # Skip when negative example is registered
            if d_x == 0:
                continue
            # Compute interpolated precision
            max_ = prec[idx:].max()
            ap +=  max_ * rec[idx] if idx == 0 \
                else 0.5 * (max_ + torch.max(prec[idx - 1], max_)) * d_x
        return ap

    @staticmethod
    def compute_per_class_ap_with_11_point_interpolation(tuple_: Tuple[Tensor, Tensor]) -> Tensor:
        """
        Arguments:
            tuple_(Tuple[Tensor, Tensor]): precision and recall
        Returns:
            ap(Tensor[1])
        """
        # precision和recall
        prec, rec = tuple_
        dtype = rec.dtype
        ap = 0
        # 只需要选取当Recall >= 0, 0.1, 0.2, ..., 1共11个点时的Precision最大值，
        # 然后AP就是这11个Precision的平均值，mAP就是所有类别AP值的平均
        for t in torch.linspace(0, 1, 11, dtype=dtype):
            inds = torch.nonzero(rec >= t).squeeze()
            if inds.numel():
                ap += (prec[inds].max() / 11)
        return ap

    @classmethod            
    def compute_ap(cls, output: Tensor, labels: Tensor,
            num_gt: Optional[Tensor] = None,
            algorithm: str = 'AUC',
            chunksize: int = -1) -> Tensor:
        """
        Compute average precision under the classification setting. Scores of all 
        classes are retained for each sample.

        Arguments:
            output(Tensor[N, K])
            labels(Tensor[N, K])
            num_gt(Tensor[K]): Number of ground truth instances for each class
            algorithm(str): AP evaluation algorithm
            chunksize(int, optional): The approximate size the given iterable will be split
                into for each worker. Use -1 to make the argument adaptive to iterable size
                and number of workers
        Returns:
            ap(Tensor[K])
        """
        prec, rec = cls.compute_precision_and_recall(output, labels, 
            num_gt=num_gt)
        ap = torch.zeros(output.shape[1], dtype=prec.dtype)
        # Use the logic from pool._map_async to compute chunksize
        # https://github.com/python/cpython/blob/master/Lib/multiprocessing/pool.py
        # NOTE: Inappropriate chunksize will cause [Errno 24]Too many open files
        # Make changes with caution
        if chunksize == -1:
            chunksize, extra = divmod(
                    output.shape[1],
                    multiprocessing.cpu_count() * 4)
            if extra:
                chunksize += 1
       
        if algorithm == 'INT':
            algorithm_handle = cls.compute_per_class_ap_with_interpolation
        elif algorithm == '11P':
            algorithm_handle = cls.compute_per_class_ap_with_11_point_interpolation
        elif algorithm == 'AUC':
            algorithm_handle = cls.compute_per_class_ap_as_auc
        else:
            raise ValueError("Unknown algorithm option {}.".format(algorithm))

        with multiprocessing.get_context('spawn').Pool() as pool:
            for idx, result in enumerate(pool.imap(
                func=algorithm_handle,
                # NOTE: Use transpose instead of T for compatibility
                iterable=zip(prec.transpose(0,1), rec.transpose(0,1)),
                chunksize=chunksize
            )):
                ap[idx] = result
        
        return ap

    @staticmethod
    def compute_precision_and_recall(output: Tensor, labels: Tensor,
            num_gt: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Arguments:
            output(Tensor[N, K])
            labels(Tensor[N, K])
            num_gt(Tensor[K])
        Returns:
            prec(Tensor[N, K])
            rec(Tensor[N, K])
        """
        order = output.argsort(0, descending=True)
        tp = labels[
            order,
            torch.ones_like(order) * torch.arange(output.shape[1])
        ]
        fp = 1 - tp
        tp = tp.cumsum(0)
        fp = fp.cumsum(0)

        prec = tp / (tp + fp)
        rec = div(tp, labels.sum(0)) if num_gt is None \
            else div(tp, num_gt)

        return prec, rec

    def append(self, output: Tensor, labels: Tensor) -> None:
        """
        Add new results to the meter

        Arguments:
            output(tensor[N, K]): Network output with N examples and K classes
            labels(tensor[N, K]): Binary labels
        """
        if isinstance(output, torch.Tensor) and isinstance(labels, torch.Tensor):
            assert output.shape == labels.shape, \
                "Output scores do not match the dimension of labelss"
            self._output_temp.append(output.detach().cpu().to(self._dtype))
            self._labels_temp.append(labels.detach().cpu().to(self._dtype))
        else:
            raise TypeError("Arguments should both be torch.Tensor")

    def reset(self, keep_old: bool = False) -> None:
        """
        Clear saved statistics

        Arguments:
            keep_tracked(bool): If True, clear only the newly collected statistics
                since last evaluation
        """
        if not keep_old:
            self._output = torch.tensor([], dtype=self._dtype)
            self._labels = torch.tensor([], dtype=self._dtype)
        self._output_temp = [torch.tensor([], dtype=self._dtype)]
        self._labels_temp = [torch.tensor([], dtype=self._dtype)]

    def eval(self) -> Tensor:
        """
        Evaluate the average precision based on collected statistics

        Returns:
            torch.Tensor[K]: Average precisions for K classes
        """
        self._output = torch.cat([
            self._output,
            torch.cat(self._output_temp, 0)
        ], 0)
        self._labels = torch.cat([
            self._labels,
            torch.cat(self._labels_temp, 0)
        ], 0)
        self.reset(keep_old=True)

        # Sanity check
        if self.num_gt is not None:
            self.num_gt = self.num_gt.to(dtype=self._labels.dtype)
            faulty_cls = torch.nonzero(self._labels.sum(0) > self.num_gt).squeeze(1)
            if len(faulty_cls):
                raise AssertionError("Class {}: ".format(faulty_cls.tolist())+
                    "Number of true positives larger than that of ground truth")
        if len(self._output) and len(self._labels):
            return self.compute_ap(self._output, self._labels, num_gt=self.num_gt,
                algorithm=self.algorithm, chunksize=self._chunksize)
        else:
            print("WARNING: Collected results are empty. "
                "Return zero AP for all class.")
            return torch.zeros(self._output.shape[1], dtype=self._dtype)

class DetectionAPMeter:
    """
    A variant of AP meter, where network outputs are assumed to be class-specific.
    Different classes could potentially have different number of samples.

    注：对于HOI detection人物而言，仅实现了HICO-DET的评估方法，暂不支持V-COCO.

    Required Arguments:
        num_cls(int): Number of target classes. HOI的种类数量，对于HICO-DET而言，就是600.
    Optional Arguemnts:
        num_gt(iterable): Number of ground truth instances for each class. When left
            as None, all positives are assumed to have been included in the collected
            results. As a result, full recall is guaranteed when the lowest scoring
            example is accounted for.
            每个HOI类别的实例个数，对于HICO-DET数据集而言，输入的就应该是长度为600的列表，
            num_gt[i]表示第i个HOI类别对应的实例数量。
        algorithm(str, optional): A choice between '11P' and 'AUC'
            '11P': 11-point interpolation algorithm prior to voc2010
            'INT': Interpolation algorithm with all points used in voc2010
            'AUC': Precisely as the area under precision-recall curve
        nproc(int, optional): The number of processes used to compute mAP. Default: 20
            用于计算mAP的进程数量，默认是20
        precision(int, optional): Precision used for float-point operations. Choose
            amongst 64, 32 and 16. Default is 64
            浮点运算的计算精度，默认是float64
        output(list[tensor], optinoal): A collection of output scores for K classes
        labels(list[tensor], optinoal): Binary labels

        output和labels是模型输出的预测结果，可以不指定，后面调用append()方法进行添加。

    Usage:

    (1) Evalute AP using provided output scores and labels

        >>> # Given output(list[tensor]) and labels(list[tensor])
        >>> meter = pocket.utils.DetectionAPMeter(num_cls, output=output, labels=labels)
        >>> ap = meter.eval(); map_ = ap.mean()

    (2) Collect results on the fly and evaluate AP

        >>> meter = pocket.utils.DetectionAPMeter(num_cls)
        >>> # Get class-specific predictions. The following is an example
        >>> # Assume output(tensor[N, K]) and target(tensor[N]) is given
        >>> pred = output.argmax(1)
        >>> scores = output.max(1)
        >>> meter.append(scores, pred, pred==target)
        >>> ap = meter.eval(); map_ = ap.mean()
        >>> # If you are to start new evaluation and want to reset the meter
        >>> meter.reset()

    """
    def __init__(self, num_cls: int, num_gt: Optional[Tensor] = None,
            algorithm: str = 'AUC', nproc: int = 20,
            precision: int = 64,
            output: Optional[List[Tensor]] = None,
            labels: Optional[List[Tensor]] = None) -> None:
        if num_gt is not None and len(num_gt) != num_cls:
            raise AssertionError("Provided ground truth instances"
                "do not have the same number of classes as specified")

        self.num_cls = num_cls   # HOI类别个数
        self.num_gt = num_gt if num_gt is not None else \
            [None for _ in range(num_cls)]
        self.algorithm = algorithm
        self._nproc = nproc
        self._dtype = eval('torch.float' + str(precision))

        is_none = (output is None, labels is None)
        if is_none == (True, True):
            self._output = [torch.tensor([], dtype=self._dtype) for _ in range(num_cls)]
            self._labels = [torch.tensor([], dtype=self._dtype) for _ in range(num_cls)]
        elif is_none == (False, False):
            assert len(output) == len(labels), \
                "The given output does not have the same number of classes as labels"
            assert len(output) == num_cls, \
                "The number of classes in the given output does not match the argument"
            self._output = to_tensor(output, 
                input_format='list', dtype=self._dtype, device='cpu')
            self._labels = to_tensor(labels,
                input_format='list', dtype=self._dtype, device='cpu')
        else:
            raise AssertionError("Output and labels should both be given or None")

        self._output_temp = [[] for _ in range(num_cls)]  # self._output_temp[i]存储被预测为第i个类别的所有预测结果的置信度分数
        self._labels_temp = [[] for _ in range(num_cls)]  # self._output_temp[i]存储被预测为第i个类别的所有预测结果是否是true positive


    # Note: Python中@classmethod修饰的方法与@staticmethod修饰的方法都可以通过类名或类的实例调用。
    # 但是@classmethod修饰的方法内部能够访问类变量(类变量是定义在类中，而不是在类的实例中的变量)，
    # 而@staticmethod修饰的方法不能够访问类变量。
    @classmethod
    def compute_ap(cls, output: List[Tensor], labels: List[Tensor],
            num_gt: Iterable, nproc: int, algorithm: str = 'AUC') -> Tuple[Tensor, Tensor]:
        """
        Compute average precision under the detection setting. Only scores of the 
        predicted classes are retained for each sample. As a result, different classes
        could have different number of predictions.

        Arguments:
            output(list[Tensor]):
            labels(list[Tensor]):
            num_gt(iterable): Number of ground truth instances for each class
            nproc(int, optional): 进程数量
            algorithm(str): AP evaluation algorithm
        Returns:
            ap(Tensor[K])
            max_rec(Tensor[K])
        """
        ap = torch.zeros(len(output), dtype=output[0].dtype)
        max_rec = torch.zeros_like(ap)

        if algorithm == 'INT':
            algorithm_handle = \
                AveragePrecisionMeter.compute_per_class_ap_with_interpolation
        elif algorithm == '11P':
            algorithm_handle = \
                AveragePrecisionMeter.compute_per_class_ap_with_11_point_interpolation
        elif algorithm == 'AUC':
            algorithm_handle = \
                AveragePrecisionMeter.compute_per_class_ap_as_auc
        else:
            raise ValueError("Unknown algorithm option {}.".format(algorithm))

        # Avoid multiprocessing when the number of processes is fewer than two
        # 单进程计算mAP
        if nproc < 2:
            for idx in range(len(output)):  # 对每个类别
                ap[idx], max_rec[idx] = cls.compute_ap_for_each((
                    idx, list(num_gt)[idx],
                    output[idx], labels[idx],
                    algorithm_handle
                ))
            return ap, max_rec

        # 多进程计算mAP
        with multiprocessing.get_context('spawn').Pool(nproc) as pool:
            for idx, results in enumerate(pool.map(
                func=cls.compute_ap_for_each,
                iterable=[(idx, ngt, out, gt, algorithm_handle) 
                    for idx, (ngt, out, gt) in enumerate(zip(num_gt, output, labels))]
            )):
                ap[idx], max_rec[idx] = results

        return ap, max_rec

    @classmethod
    def compute_ap_for_each(cls, tuple_):
        """计算单个类别的AP"""
        # idx是当前的HOI类别编号(0~599)
        # num_gt是当前这个类别的实例数量
        # output是预测结果中，所有属于当前这个类别的预测结果的置信度
        # labels是预测结果中，所有属于当前这个类别的预测结果是否是true positive
        # algorithm是具体的mAP计算方法
        idx, num_gt, output, labels, algorithm = tuple_
        # Sanity check
        # 一个真实的实例只能对应一个预测实例，这部分是由pocket/utils/association.py保证的。
        if num_gt is not None and labels.sum() > num_gt:
            raise AssertionError("Class {}: ".format(idx)+
                "Number of true positives larger than that of ground truth")
        if len(output) and len(labels):
            # 计算precision和recall
            prec, rec = cls.compute_pr_for_each(output, labels, num_gt)
            # 交由具体的算法计算PR曲线下的面积
            return algorithm((prec, rec)), rec[-1]
        else:
            print("WARNING: Collected results are empty. "
                "Return zero AP for class {}.".format(idx))
            return 0, 0

    @staticmethod
    def compute_pr_for_each(output: Tensor, labels: Tensor,
            num_gt: Optional[Union[int, float]] = None) -> Tuple[Tensor, Tensor]:
        """
        计算单个类别的PR曲线上的所有点

        Arguments:
            output(Tensor[N])
            labels(Tensor[N]): Binary labels for each sample
            num_gt(int or float): Number of ground truth instances
        Returns:
            prec(Tensor[N])
            rec(Tensor[N])
        """
        order = output.argsort(descending=True)

        tp = labels[order]
        fp = 1 - tp
        tp = tp.cumsum(0)
        fp = fp.cumsum(0)

        prec = tp / (tp + fp)
        rec = div(tp, labels.sum().item()) if num_gt is None \
            else div(tp, num_gt)  # 即所有为正的样本中预测为正的比例

        return prec, rec

    def append(self, output: Tensor, prediction: Tensor, labels: Tensor) -> None:
        """
        Add new results to the meter
        添加新的预测结果。

        Arguments:
            output(tensor[N]): Output scores for each sample
                对于HOI detection任务而言，output[i]表示第i个预测的置信度分数（通常由人框、物框、交互等的置信度分数计算得到）
            prediction(tensor[N]): Predicted classes 0~(K-1)
                对于HOI detection任务而言，prediction[i]表示预测的第i个HOI的类别（对于HICO-DET数据集，类别编号从0到599）
            labels(tensor[N]): Binary labels for the predicted classes
                对于HOI detection任务而言，labels[i]=1表示第i个预测是true positive，
        """
        if isinstance(output, torch.Tensor) and \
                isinstance(prediction, torch.Tensor) and \
                isinstance(labels, torch.Tensor):
            prediction = prediction.long()
            unique_cls = prediction.unique()
            for cls_idx in unique_cls:  # 对每个类别
                sample_idx = torch.nonzero(prediction == cls_idx).squeeze(1)
                self._output_temp[cls_idx.item()] += output[sample_idx].tolist()
                self._labels_temp[cls_idx.item()] += labels[sample_idx].tolist()
        else:
            raise TypeError("Arguments should be torch.Tensor")

    def reset(self, keep_old: bool = False) -> None:
        """
        Clear saved statistics

        Arguments:
            keep_tracked(bool): If True, clear only the newly collected statistics since last evaluation
                如果为True，则仅清除self._output_temp和self._labels_temp；
                反之，则除了清除上述内容外，还清除self._output和self._labels
        """
        num_cls = len(self._output_temp)
        if not keep_old:
            self._output = [torch.tensor([], dtype=self._dtype) for _ in range(num_cls)]
            self._labels = [torch.tensor([], dtype=self._dtype) for _ in range(num_cls)]
        self._output_temp = [[] for _ in range(num_cls)]
        self._labels_temp = [[] for _ in range(num_cls)]

    def eval(self) -> Tensor:
        """
        Evaluate the average precision based on collected statistics

        Returns:
            torch.Tensor[K]: Average precisions for K classes
        """
        # self._output是上一次eval()时添加进来的数据，这里将所有数据拼接到一起
        self._output = [torch.cat([
            out1, torch.as_tensor(out2, dtype=self._dtype)
        ]) for out1, out2 in zip(self._output, self._output_temp)]

        self._labels = [torch.cat([
            tar1, torch.as_tensor(tar2, dtype=self._dtype)
        ]) for tar1, tar2 in zip(self._labels, self._labels_temp)]

        # 清空self._output_temp和self._labels_temp，以便用户调用append()添加其他预测结果
        self.reset(keep_old=True)

        # 对目前收集到的预测结果进行评估
        self.ap, self.max_rec = self.compute_ap(self._output, self._labels, self.num_gt,
            nproc=self._nproc, algorithm=self.algorithm)

        # 返回每个类别的AP，求平均即得到mAP
        return self.ap
