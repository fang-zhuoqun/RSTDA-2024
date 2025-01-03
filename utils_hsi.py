import torch
import numpy as np
from sklearn import model_selection


def sample_gt(gt, train_size, args, mode='random'):
    """Extract a fixed percentage of samples from an array of labels.

    Args:
        gt: a 2D array of int labels
        percentage: [0, 1] float
    Returns:
        train_gt, test_gt: 2D arrays of int labels

    """
    # 返回标签中非0标签的位置
    indices = np.nonzero(gt)
    # X 对应非0标签坐标，y对应坐标标签
    X = list(zip(*indices))  # x,y features
    y = gt[indices].ravel()  # classes
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    if train_size > 1:
        train_size = int(train_size)
    train_label = []
    test_label = []
    random_state = 23 if args.seed is None else args.seed

    if mode == 'random':
        if train_size == 1:
            # random.shuffle() 函数将序列中的元素随机打乱
            # random.shuffle(X)
            train_indices = [list(t) for t in zip(*X)]
            # append() 方法向列表末尾追加元素。
            # [train_label.append(i) for i in gt[tuple(train_indices)]]
            # np.column_stack() 将矩阵按列合并
            # train_set = np.column_stack((train_indices[0], train_indices[1], train_label))
            train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
            test_gt = []
            test_set = []
        #     train_indices, test_indices = model_selection.train_test_split(X, train_size=1, random_state=random_state)
        elif train_size > 1:
        # select 180 samples from each class.
            np.random.seed(random_state)
            np.random.shuffle(X)
            m = int(max(y))
            train = {}
            train_indices = []
            test = {}
            test_indices = []
            for i in range(m):
                indices_whole = [value for index, value in enumerate(X) if gt[X[index]] == i + 1]
                # np.random.seed(random_state)
                # np.random.shuffle(indices_whole)
                train[i] = indices_whole[:train_size]
                test[i] = indices_whole[train_size:]

            for i in range(m):
                train_indices += train[i]
                test_indices += test[i]
            train_indices = [list(t) for t in zip(*train_indices)]
            test_indices = [list(t) for t in zip(*test_indices)]
            train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
            test_gt[tuple(test_indices)] = gt[tuple(test_indices)]
        else:
            train_indices, test_indices = model_selection.train_test_split(X, train_size=train_size, stratify=y,
                                                                           random_state=random_state)
            train_indices = [list(t) for t in zip(*train_indices)]
            test_indices = [list(t) for t in zip(*test_indices)]
            train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
            test_gt[tuple(test_indices)] = gt[tuple(test_indices)]

            [train_label.append(i) for i in gt[tuple(train_indices)]]
            train_set = np.column_stack((train_indices[0], train_indices[1], train_label))
            [test_label.append(i) for i in gt[tuple(test_indices)]]
            test_set = np.column_stack((test_indices[0], test_indices[1], test_label))

    elif mode == 'disjoint':
        # d = a.copy()  # 建立一个和a一样的d,d和a没有关系
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            mask = gt == c
            for x in range(gt.shape[0]):
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = np.count_nonzero(mask[x:, :])
                try:
                    ratio = first_half_count / second_half_count
                    if ratio > 0.9 * train_size and ratio < 1.1 * train_size:
                        break
                except ZeroDivisionError:
                    continue
            mask[:x, :] = 0
            train_gt[mask] = 0

        test_gt[train_gt > 0] = 0
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    # return train_gt, test_gt, train_set, test_set
    return train_gt, test_gt


def padding_hsi(img, gt, patch_size):
    # padding补边
    r = patch_size // 2 + 1
    # img = np.pad(img, ((r, r), (r, r), (0, 0)), 'symmetric')
    img = np.pad(img, ((r, r), (r, r), (0, 0)), 'constant', constant_values=(0, 0))
    gt = np.pad(gt, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
    return img, gt


class HyperX(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, args, transform=False):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX, self).__init__()
        self.transform = transform
        self.patch_size = args.patch_size
        self.data, self.label = padding_hsi(data, gt, self.patch_size)
        self.ignored_labels = [0]
        # self.data = data
        # self.label = gt

        # self.ignored_labels = set(hyperparams['ignored_labels'])
        if self.transform:
            self.flip_augmentation = True
            self.radiation_augmentation = True
        else:
            self.flip_augmentation = False
            self.radiation_augmentation = False
        # self.mixture_augmentation = hyperparams['mixture_augmentation']
        mask = np.ones_like(self.label)
        for l in self.ignored_labels:
            mask[self.label == l] = 0
        x_pos, y_pos = np.nonzero(mask)
        r = self.patch_size // 2
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos) if
                                 x > r and x < self.data.shape[0] - r and y > r and y < self.data.shape[1] - r])
        self.labels = [self.label[x, y] for x, y in self.indices]

    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1 / 25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise
    #
    # def mixture_noise(self, data, label, beta=1 / 25):
    #     alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
    #     noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
    #     data2 = np.zeros_like(data)
    #     for idx, value in np.ndenumerate(label):
    #         if value not in self.ignored_labels:
    #             l_indices = np.nonzero(self.labels == value)[0]
    #             l_indice = np.random.choice(l_indices)
    #             assert (self.labels[l_indice] == value)
    #             x, y = self.indices[l_indice]
    #             data2[idx] = self.data[x, y]
    #     return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size
        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        if self.flip_augmentation and self.patch_size > 1 and np.random.random() < 0.5:
            # Perform data augmentation (only on 2D patches)
            data, label = self.flip(data, label)
        if self.radiation_augmentation and np.random.random() < 0.5:
            data = self.radiation_noise(data)
        # if self.mixture_augmentation and np.random.random() < 0.5:
        #     data = self.mixture_noise(data, label)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')
        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        # Extract the center label if needed
        if self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
            label -= 1  # Labels start at 0 in the dataset
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]
            label -= 1  # Labels start at 0 in the dataset

        return data, label
