import os

from nets.clip_classifier import ImageEncoder, get_zeroshot_classifier, ImageClassifier

label_path = ""  # the path to cls_classes.txt
with open(label_path, 'r') as file:
    label_list = [line.strip().replace("_", " ") for line in file]
datasetname = "ucf101"
image_encoder = ImageEncoder("RN50", keep_lang=True)
classification_head = get_zeroshot_classifier(image_encoder.model, label_list=label_list)
delattr(image_encoder.model, 'transformer')
classifier = ImageClassifier(image_encoder, classification_head, process_images=False)

zeroshot_checkpoint = os.path.join("./clip_classifier", datasetname, 'zeroshot' + '.pt')
classifier.save(zeroshot_checkpoint)
