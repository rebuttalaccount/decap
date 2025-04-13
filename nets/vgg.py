import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Linear(512, num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def freeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def Unfreeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = True


def make_layers(cfg):
    layers = []
    in_channels = 3
    for x in cfg:
        if x == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                       nn.BatchNorm2d(x),
                       nn.ReLU(inplace=True)]
            in_channels = x
    layers += [nn.AdaptiveAvgPool2d((1, 1))]
    return nn.Sequential(*layers)


# 224,224,3 -> 224,224,64 -> 112,112,64 -> 112,112,128 -> 56,56,128 -> 56,56,256 -> 28,28,256 -> 28,28,512
# 14,14,512 -> 14,14,512 -> 7,7,512
cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False, progress=True, num_classes=1000):
    model = VGG(make_layers(cfgs['A']))
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['vgg11_bn'], model_dir='./model_data',
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)

    if num_classes != 1000:
        model.classifier = nn.Sequential(nn.Linear(512, num_classes), )
    return model


def vgg13(pretrained=False, progress=True, num_classes=1000):
    model = VGG(make_layers(cfgs['B']))
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['vgg13_bn'], model_dir='./model_data',
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)

    if num_classes != 1000:
        model.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    return model


def vgg16(pretrained=False, progress=True, num_classes=1000):
    model = VGG(make_layers(cfgs['D']))
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['vgg16_bn'], model_dir='./model_data',
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)

    if num_classes != 1000:
        model.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    return model
