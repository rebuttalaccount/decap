import os
from PIL import Image

image_folder = 'path/to/dataset/test'
file_names = os.listdir(image_folder)
print(file_names)

txt_file = 'path/to/cls_classes.txt'
print("begin")

with open(txt_file, 'w') as f:
    for file_name in file_names:
        label = file_name.replace("_", " ")
        f.write(file_name + '\n')
