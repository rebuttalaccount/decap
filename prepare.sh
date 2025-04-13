#!/usr/bin/env bash

dataset="ucf101"
device=0


CUDA_VISIBLE_DEVICES=$device python prepare/txt_annotation.py

CUDA_VISIBLE_DEVICES=$device python prepare/get_zeroshot.py

CUDA_VISIBLE_DEVICES=$device python prepare/full.py $dataset full 5 20