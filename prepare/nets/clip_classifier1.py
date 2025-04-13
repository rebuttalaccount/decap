import torch
import clip.utils as utils


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
        
    def test():
        print("ok")

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    def freeze_backbone(self,):
        for para in self.image_encoder.parameters():
            para.requires_grad = False

    @classmethod
    def load(cls, filename):
        # print(f'Loading image classifier from {filename}')
        return utils.torch_load(filename)
    

