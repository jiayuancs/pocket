"""
HICODet dataset under PyTorch framework

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import json
import numpy as np

from typing import Optional, List, Callable, Tuple
from .base import ImageDataset, DataSubset
# from base import ImageDataset, DataSubset

class HICODetSubset(DataSubset):
    def __init__(self, *args) -> None:
        super().__init__(*args)
    def filename(self, idx: int) -> str:
        """Override: return the image file name in the subset"""
        return self._filenames[self._idx[self.pool[idx]]]
    def image_size(self, idx: int) -> Tuple[int, int]:
        """Override: return the size (width, height) of an image in the subset"""
        return self._image_sizes[self._idx[self.pool[idx]]]
    @property
    def anno_interaction(self) -> List[int]:
        """Override: Number of annotated box pairs for each interaction class"""
        num_anno = [0 for _ in range(self.num_interation_cls)]
        intra_idx = [self._idx[i] for i in self.pool]
        for idx in intra_idx:
            for hoi in self._anno[idx]['hoi']:
                num_anno[hoi] += 1
        return num_anno
    @property
    def anno_object(self) -> List[int]:
        """Override: Number of annotated box pairs for each object class"""
        num_anno = [0 for _ in range(self.num_object_cls)]
        anno_interaction = self.anno_interaction
        for corr in self._class_corr:
            num_anno[corr[1]] += anno_interaction[corr[0]]
        return num_anno
    @property
    def anno_action(self) -> List[int]:
        """Override: Number of annotated box pairs for each action class"""
        num_anno = [0 for _ in range(self.num_action_cls)]
        anno_interaction = self.anno_interaction
        for corr in self._class_corr:
            num_anno[corr[2]] += anno_interaction[corr[0]]
        return num_anno

class HICODet(ImageDataset):
    """
    Arguments:
        root(str): Root directory where images are downloaded to
        anno_file(str): Path to json annotation file
        transform(callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version
        target_transform(callable, optional): A function/transform that takes in the
            target and transforms it
        transforms (callable, optional): A function/transform that takes input sample 
            and its target as entry and returns a transformed version.
    """
    def __init__(self, root: str, anno_file: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None) -> None:
        super(HICODet, self).__init__(root, transform, target_transform, transforms)
        with open(anno_file, 'r') as f:
            anno = json.load(f)

        self.num_object_cls = 80       # 物体类别数量(对应COCO目标检测的80个类别)
        self.num_interation_cls = 600  # 动作和物体的组合数量
        self.num_action_cls = 117      # 动作数量
        self._anno_file = anno_file    # 标注文件路径

        # Load annotations
        self._load_annotation_and_metadata(anno)

    def __len__(self) -> int:
        """Return the number of images"""
        return len(self._idx)

    def __getitem__(self, i: int) -> tuple:
        """
        Arguments:
            i(int): Index to an image，范围是[0, len(HICODet))
        
        Returns:
            tuple[image, target]:
                默认情况下(没有应用transform对数据进行变换), image是PIL图片对象(RGB格式), target是标注信息字典。
                target字典各字段的含义如下：
                    "boxes_h": list[list[4]] 列表长度为N，表示当前图片中的第i个HOI的人框坐标，坐标格式为(x1, y1, x2, y2)
                    "boxes_o": list[list[4]] 列表长度为N，表示当前图片中的第i个HOI的物框坐标，坐标格式为(x1, y1, x2, y2)
                    "hoi":: list[N] 表示当前图片中的第i个HOI在整个数据集的600种HOI中的类别编号
                    "verb": list[N] 表示当前图片中的第i个HOI的动作类别编号
                    "object": list[N] 表示当前图片中的第i个HOI的物体类别编号
                其中N表示当前图片中的HOI个数，注意：一个human-object pair可能对应多个HOI，因为
                一个human-object pair可能对应多个动作类别。
        """
        intra_idx = self._idx[i]
        return self._transforms(
            self.load_image(os.path.join(self._root, self._filenames[intra_idx])), 
            self._anno[intra_idx]
            )

    def __repr__(self) -> str:
        """Return the executable string representation"""
        reprstr = self.__class__.__name__ + '(root=' + repr(self._root)
        reprstr += ', anno_file='
        reprstr += repr(self._anno_file)
        reprstr += ')'
        # Ignore the optional arguments
        return reprstr

    def __str__(self) -> str:
        """Return the readable string representation"""
        reprstr = 'Dataset: ' + self.__class__.__name__ + '\n'
        reprstr += '\tNumber of images: {}\n'.format(self.__len__())
        reprstr += '\tImage directory: {}\n'.format(self._root)
        reprstr += '\tAnnotation file: {}\n'.format(self._root)
        return reprstr

    @property
    def annotations(self) -> List[dict]:
        """
        是一个列表，每个元素是一个字典，对应一张图片的所有标注信息

        Returns: list[dict]
        """
        return self._anno

    @property
    def class_corr(self) -> List[Tuple[int, int, int]]:
        """
        Class correspondence matrix in zero-based index
        [
            [hoi_idx, obj_idx, verb_idx],
            ...
        ]

        Returns:
            list[list[3]]
        """
        return self._class_corr.copy()

    @property
    def object_n_verb_to_interaction(self) -> List[list]:
        """
        The interaction classes corresponding to an object-verb pair

        HICODet.object_n_verb_to_interaction[obj_idx][verb_idx] gives interaction class
        index if the pair is valid, None otherwise

        object_n_verb_to_interaction[i][j]表示第i个物体是否可以与第j个动作进行组合，如果可以，
        则值为该HOI组合在600种HOI中的编号，否则为None

        Returns:
            list[list[117]]
        """
        lut = np.full([self.num_object_cls, self.num_action_cls], None)
        for i, j, k in self._class_corr:
            lut[j, k] = i
        return lut.tolist()

    @property
    def object_to_interaction(self) -> List[list]:
        """
        The interaction classes that involve each object type
        每个物体类别可能对应的所有HOI类别编号

        Returns:
            list[list]
        """
        obj_to_int = [[] for _ in range(self.num_object_cls)]
        for corr in self._class_corr:
            obj_to_int[corr[1]].append(corr[0])
        return obj_to_int

    @property
    def object_to_verb(self) -> List[list]:
        """
        The valid verbs for each object type
        每个物体类别可能对应的所有动词索引编号

        Returns:
            list[list]

        实例：
            object_to_verb[i]表示第i个物体类别对应的所有可能的动作索引
        """
        obj_to_verb = [[] for _ in range(self.num_object_cls)]
        for corr in self._class_corr:
            obj_to_verb[corr[1]].append(corr[2])
        return obj_to_verb

    @property
    def anno_interaction(self) -> List[int]:
        """
        Number of annotated box pairs for each interaction class
        600种HOI出现的频率，其中有138种的频率小于10，记为rare set
        注意：一个human-object pair可能对应多个HOI类别。

        Returns:
            list[600]
        """
        return self._num_anno.copy()

    @property
    def anno_object(self) -> List[int]:
        """
        Number of annotated box pairs for each object class.
        每个物体类别对应的HOI标注个数（注意，一个human-object pair可能对应多个HOI标注，即对应多个动作）

        Returns:
            list[80]
        """
        num_anno = [0 for _ in range(self.num_object_cls)]
        for corr in self._class_corr:
            num_anno[corr[1]] += self._num_anno[corr[0]]
        return num_anno

    @property
    def anno_action(self) -> List[int]:
        """
        Number of annotated box pairs for each action class.
        每个动作类别对应的HOI标注个数（注意，一个human-object pair可能对应多个HOI标注，即对应多个动作）

        Returns:
            list[117]
        """
        num_anno = [0 for _ in range(self.num_action_cls)]
        for corr in self._class_corr:
            num_anno[corr[2]] += self._num_anno[corr[0]]
        return num_anno

    @property
    def objects(self) -> List[str]:
        """
        Object names. HICO-DET中的80种物体类别的名称列表

        Returns:
            list[str]
        """
        return self._objects.copy()

    @property
    def verbs(self) -> List[str]:
        """
        Verb (action) names

        Returns:
            list[str]
        """
        return self._verbs.copy()

    @property
    def interactions(self) -> List[str]:
        """
        Combination of verbs and objects

        Returns:
            list[str]

        示例：
            HICO-DET数据集中有600种HOI，因此这里返回的就是一个长度为600的列表，
            列表中的每个元素都是字符串，格式为'verb object'，例如：
            - 'wash toothbrush'
            - 'jump snowboard'
        """
        return [self._verbs[j] + ' ' + self.objects[i] 
            for _, i, j in self._class_corr]

    def split(self, ratio: float) -> Tuple[HICODetSubset, HICODetSubset]:
        """
        Split the dataset according to given ratio
        按给定的比例将数据集划分为两部分

        Arguments:
            ratio(float): The percentage of training set between 0 and 1
        Returns:
            train(Dataset): 占比为ratio
            val(Dataset): 占比为1-ratio
        """
        perm = np.random.permutation(len(self._idx))
        n = int(len(perm) * ratio)
        return HICODetSubset(self, perm[:n]), HICODetSubset(self, perm[n:])

    def filename(self, idx: int) -> str:
        """Return the image file name given the index"""
        return self._filenames[self._idx[idx]]

    def image_size(self, idx: int) -> Tuple[int, int]:
        """Return the size (width, height) of an image"""
        return self._image_sizes[self._idx[idx]]

    def _load_annotation_and_metadata(self, f: dict) -> None:
        """
        Arguments:
            f(dict): Dictionary loaded from {anno_file}.json
        """
        idx = list(range(len(f['filenames'])))
        # 'empty'字段记录了那些没有人框和物框的图片
        # 因为HICO-DET训练集中，有485张图片没有标记人框和物框，所以需要移除这些训练样本。
        # 在HICO-DET测试集中，有112张图片没有标记人框和物框，所以也需要移除这些测试样本。
        for empty_idx in f['empty']:
            idx.remove(empty_idx)

        # 统计600种HOI的频率
        num_anno = [0 for _ in range(self.num_interation_cls)]
        for anno in f['annotation']:
            for hoi in anno['hoi']:
                num_anno[hoi] += 1

        self._idx = idx            # 有效样本的索引
        # 600种HOI出现的频率，其中有138种的频率小于10，记为rare set
        # 注意：一个human-object pair可能对应多个HOI类别。
        self._num_anno = num_anno

        self._anno = f['annotation']            # 是一个列表，每个元素是一个字典，对应一张图片的所有标注信息，坐标格式为(x1, y1, x2, y2)
        self._filenames = f['filenames']        # 文件名列表
        self._image_sizes = f['size']           # 文件大小列表 [[w, h], ...]
        # 列表：[[hoi_id, object_id, verb_id], ...]，表示verb_id动词可以与object_id物体进行组合
        self._class_corr = f['correspondence']
        self._empty_idx = f['empty']            # 无效样本的索引，这些样本缺少标注框信息
        self._objects = f['objects']            # 80个物体类别名称
        self._verbs = f['verbs']                # 117个动作类别名称

if __name__ == '__main__':
    HICO_ROOT = "/mnt/c/Document/10Dataset/HICO-DET"

    hico_det_test = HICODet(
        root=os.path.join(HICO_ROOT, "hico_20160224_det/images/test2015"),
        anno_file=os.path.join(HICO_ROOT, "instances_test2015.json")
    )

    hico_det_train = HICODet(
        root=os.path.join(HICO_ROOT, "hico_20160224_det/images/train2015"),
        anno_file=os.path.join(HICO_ROOT, "instances_train2015.json")
    )

    from pocket.utils import draw_box_pairs
    import matplotlib.pyplot as plt
    image, annotation = hico_det_test[6]

    draw_box_pairs(image, annotation['boxes_h'], annotation['boxes_o'], width=4)

    plt.imshow(image)
    plt.show()
