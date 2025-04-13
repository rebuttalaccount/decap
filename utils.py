import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR

import tabular_logger as tlogger
from nets.clip_classifier1 import ImageClassifier
from nets import resnet, vgg


def ResizeImage(images, img_shape):
    if isinstance(img_shape, int):
        width = height = img_shape
    elif isinstance(img_shape, (tuple, list)):
        if len(img_shape) == 2:
            height = img_shape[0]
            width = img_shape[1]
        elif len(img_shape) == 3:
            height = img_shape[1]
            width = img_shape[2]
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    out = []
    for image in images:
        out.append(image.resize((int(width), int(height))))
    return out


def resize_image(images, target_size):
    resized_image = F.interpolate(images, size=target_size, mode='bilinear', align_corners=False)

    return resized_image


class Learner(nn.Module):
    def __init__(self, model, optimizer, scheduler, feature_net):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.feature_net = feature_net


def inv_sigmoid(x):
    return - np.log(1. / x - 1)


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)


class Learner_sampler(nn.Module):
    def __init__(self, inner_loop_optimizer, SGD_list, RMS_list, Adam_list, label_list=None):
        super().__init__()
        self.inner_loop_optimizer = inner_loop_optimizer
        self.SGD_list = SGD_list
        self.RMS_list = RMS_list
        self.Adam_list = Adam_list
        self.interval = [10, 20]

    def sample_learner(self, input_shape, device, learner_type="resnet18", model_path='', num_classes=10):

        if learner_type == "resnet18":
            model = resnet.resnet18(input_shape=input_shape, num_classes=num_classes)
            feature_net = None
        elif learner_type == "resnet50":
            model = resnet.resnet50(input_shape=input_shape, num_classes=num_classes)
            feature_net = None
        elif learner_type == "pretrained_resnet18":
            if model_path != '':
                model = resnet.resnet18(input_shape=input_shape, num_classes=num_classes)
                model.load_state_dict(torch.load(model_path, map_location=device))
                print('from {} load model'.format(model_path))
            else:
                model = resnet.resnet18(input_shape=input_shape, num_classes=num_classes, pretrained=True)
            feature_net = None
        elif learner_type == "pretrained_resnet50":
            if model_path != '':
                model = resnet.resnet50(input_shape=input_shape, num_classes=num_classes)
                model.load_state_dict(torch.load(model_path, map_location=device))
                print('from {} load model'.format(model_path))
            else:
                model = resnet.resnet50(input_shape=input_shape, num_classes=num_classes, pretrained=True)
            feature_net = None
        elif learner_type == "pretrained_vgg11":
            if model_path != '':
                model = vgg.vgg11(num_classes=num_classes)
                model.load_state_dict(torch.load(model_path, map_location=device))
                print('from {} load model'.format(model_path))
            else:
                model = vgg.vgg11(num_classes=num_classes, pretrained=True)
            feature_net = None
        elif learner_type == "clip":
            model = ImageClassifier.load(model_path)
            model.process_images = True
            for para in model.image_encoder.parameters():
                para.requires_grad = False
            feature_net = None
        elif learner_type == "clip_head":
            model_ = ImageClassifier.load(model_path)
            model_.process_images = True
            for para in model_.image_encoder.parameters():
                para.requires_grad = False
            model = model_.classification_head
            feature_net = model_.image_encoder
        else:
            raise NotImplementedError()
        if self.inner_loop_optimizer == "SGD":
            optimizer_c = torch.optim.SGD(model.parameters(), lr=self.SGD_list[0], momentum=self.SGD_list[1],
                                          weight_decay=5e-4)
        elif self.inner_loop_optimizer == "RMS":
            optimizer_c = torch.optim.RMSprop(model.parameters(), lr=self.RMS_list[0], alpha=self.RMS_list[1],
                                              momentum=self.RMS_list[2], eps=self.RMS_list[3])
        elif self.inner_loop_optimizer == "Adam":
            optimizer_c = torch.optim.Adam(model.parameters(), lr=self.Adam_list[0],
                                           betas=(self.Adam_list[1], self.Adam_list[2]),
                                           eps=self.Adam_list[3])
        else:
            raise ValueError(f"Inner loop optimizer '{self.inner_loop_optimizer}' not available")
        scheduler = MultiStepLR(optimizer_c, milestones=self.interval, gamma=0.1)
        # scheduler = CosineAnnealingLR(optimizer_c, 30, self.Adam_list[0] * 0.01)

        return Learner(model=model.to(device), optimizer=optimizer_c, scheduler=scheduler, feature_net=feature_net)


def evaluate_set(model, x, y, name):
    with torch.no_grad():
        batch_size = 120
        validation_pred = []
        model.eval()
        for i in range(math.ceil(len(x) / batch_size)):
            pred = model(x[i * batch_size:(i + 1) * batch_size])
            if isinstance(pred, tuple):
                pred, _ = pred
            validation_pred.append(pred)
        validation_pred = torch.cat(validation_pred, dim=0)
        single_validation_accuracy = (validation_pred.max(-1).indices == y).to(torch.float).mean()

        validation_accuracy = single_validation_accuracy.item()
        validation_loss = nn.CrossEntropyLoss()(validation_pred, y).item()
        tlogger.record_tabular('{}_loss'.format(name), validation_loss)
        tlogger.record_tabular('{}_accuracy'.format(name), validation_accuracy)
        return validation_loss, single_validation_accuracy, validation_accuracy


class EndlessDataLoader(object):
    def __init__(self, data_loader):
        self._data_loader = data_loader

    def __iter__(self):
        while True:
            for batch in self._data_loader:
                yield batch
