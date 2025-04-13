import torch
import copy

from tqdm import tqdm

import clip.clip as clip
import clip.utils as utils


class ImageEncoder(torch.nn.Module):
    def __init__(self, model="RN50", keep_lang=False,datasetname=""):
        super().__init__()

        self.model, self.train_preprocess, self.val_preprocess = clip.load(
            model, "cpu", jit=False)

        self.cache_dir = "./clip_classifier/{}".format(datasetname)

        if not keep_lang and hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)

    def save(self, filename):
        print(f'Saving image encoder to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image encoder from {filename}')
        return utils.torch_load(filename)


class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def save(self, filename):
        print(f'Saving classification head to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading classification head from {filename}')
        return utils.torch_load(filename)


class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head, process_images=True):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        self.process_images = process_images
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def forward(self, inputs):
        if self.process_images:
            inputs = self.image_encoder(inputs)
        outputs = self.classification_head(inputs)
        return outputs

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return utils.torch_load(filename)


def get_zeroshot_classifier(clip_model, label_list):
    template = ["a photo of a {}"]#["itap of a {}.","a bad photo of the {}.","a origami {}.","a photo of the large {}.",
               #"a {} in a video game.","art of the {}.","a photo of the small {}."]
    logit_scale = clip_model.logit_scale
    label_list = label_list
    device = "cpu"
    clip_model.eval()
    clip_model.to(device)

    print('Getting zeroshot weights.')
    with torch.no_grad():
        zeroshot_weights = []

        for classname in tqdm(label_list):
            texts = []
            for t in template:
                texts.append(t.format(classname))
                #texts.append(t)
            texts = clip.tokenize(texts).to(device)  # tokenize
            embeddings = clip_model.encode_text(texts)  # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= logit_scale.exp()

        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)
    # import ipdb
    # ipdb.set_trace(context=20)
    classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)
    # print(classification_head.weight)

    return classification_head
