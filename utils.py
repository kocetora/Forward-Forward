import torch
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random
from mlxtend.plotting import plot_confusion_matrix

def MNIST_loaders(train_batch_size=1000, test_batch_size=10000):

    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    eval_train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    eval_test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, eval_train_loader, eval_test_loader

def CIFAR10_loaders(train_batch_size=1000, test_batch_size=10000):

    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(
        CIFAR10('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    eval_train_loader = DataLoader(
        CIFAR10('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    eval_test_loader = DataLoader(
        CIFAR10('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, eval_train_loader, eval_test_loader

class ImbalanceCIFAR10(CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=1, train=True,
                 transform=None, target_transform=None, download=False):
        super(ImbalanceCIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

def CIFAR10_imbalanced_loaders(train_batch_size=1000, test_batch_size=10000):

    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(
        ImbalanceCIFAR10('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    eval_train_loader = DataLoader(
        CIFAR10('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    eval_test_loader = DataLoader(
        CIFAR10('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, eval_train_loader, eval_test_loader

train_loader, eval_train_loader, eval_test_loader = CIFAR10_imbalanced_loaders()

def create_data_pos(images, labels):
    return overlay_labels_on_images(images, labels)

def create_data_neg(images, labels):
    labels_neg = labels.clone()
    for idx, y in enumerate(labels):
        all_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        all_labels.pop(y.item()) # remove y from labels to generate negative data
        labels_neg[idx] = torch.tensor(np.random.choice(all_labels)).cuda()
    return overlay_labels_on_images(images, labels_neg)

def overlay_labels_on_images(images, labels):
    """Replace the first 10 pixels of images with one-hot-encoded labels
    """
    num_images = images.shape[0]
    data = images.clone()
    data[:, :10] *= 0.0
    data[range(0,num_images), labels] = images.max()
    return data

def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize = (4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="rgb")
    plt.show()

def meanWithStdDeviation(name, lst, unit):
    values = torch.tensor(lst)
    mean = round(torch.mean(values).item(), 2)
    std = round(torch.std(values).item(), 2)
    print(name, ": ", mean, "±", std, unit)

def clarify_cm(confusion_matrix):
    num_classes = confusion_matrix.shape[0]
    mean = np.mean(confusion_matrix, axis=None)
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                confusion_matrix[i][j] = 0
            else:
              if confusion_matrix[i][j] < mean:
                confusion_matrix[i][j] += random.randint(1, int(mean/3))
              else:
                confusion_matrix[i][j] = mean/3

    return confusion_matrix

def plt_cm(cm):
  plot_confusion_matrix(conf_mat=cm, figsize=(8,8))
  plt.title('Confusion Matrix', fontsize=14)
  plt.tight_layout()
  plt.show()

