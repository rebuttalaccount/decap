import os
import pickle
import random
import sys

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler, \
    AutoPipelineForText2Image, AutoPipelineForImage2Image


class EndlessDataLoader(object):
    def __init__(self, data_loader):
        self._data_loader = data_loader

    def __iter__(self):
        while True:
            for batch in self._data_loader:
                yield batch


def pretrained_model_download(model_id, img2img=False):
    # access_token=""
    # tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14"),
    # text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    # scheduler = EulerDiscreteScheduler.from_pretrained(model_id,
    #                                                      subfolder="scheduler")  # use_auth_token=access_token
    if img2img:
        if "turbo" or "xl" in model_id:
            pipe = AutoPipelineForImage2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
        else:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    else:
        if "turbo" or "xl" in model_id:
            pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
        else:
            pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe = pipe.to("cuda")
    # pipe.enable_attention_slicing()
    return pipe


def get_inputs_prompt(prompt, height=512, width=512, scaleguide=0, image=None,
                      num_images_per_prompt=20):
    num_inference_steps = 2
    if image is not None:
        return {"prompt": prompt, "guidance_scale": scaleguide, "num_inference_steps": num_inference_steps,
                "image": image, "num_images_per_prompt": num_images_per_prompt}
    else:
        return {"prompt": prompt, "height": height, "width": width, "guidance_scale": scaleguide,
                "num_inference_steps": num_inference_steps, "num_images_per_prompt": num_images_per_prompt}


def get_inputs_prompt_embs(prompt_embeds, height=512, width=512, scaleguide=0, num_images_per_prompt=20):
    num_inference_steps = 2

    return {"prompt_embeds": prompt_embeds, "height": height, "width": width,
            "num_images_per_prompt": num_images_per_prompt,
            "guidance_scale": scaleguide, "num_inference_steps": num_inference_steps}


def get_img(pipe, img_shape, prompt_embeds=None, prompt=None, scaleguide=0, image=None,
            num_images_per_prompt=1):
    if isinstance(img_shape, int):
        width = height = img_shape
    elif isinstance(img_shape, tuple):
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
    if prompt_embeds is not None:
        images = pipe(**get_inputs_prompt_embs(prompt_embeds=prompt_embeds, height=height,
                                               width=width, scaleguide=scaleguide,
                                               num_images_per_prompt=num_images_per_prompt)).images
    else:
        images = pipe(
            **get_inputs_prompt(prompt, height=height, width=width, scaleguide=scaleguide,
                                image=image, num_images_per_prompt=num_images_per_prompt)).images
    return images


def main(method, dataset, generate_num, batch_size=20):
    pipe = pretrained_model_download(
        'stabilityai/sdxl-turbo',
        # '/home/lthpc/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/ded79e214aa69e42c24d3f5ac14b76d568679cc2',
        False)
    pool_path = "./prompt_pool.pkl".format(dataset)
    learned_path = ''
    save_path = "./syn/{}".format(dataset)
    with open(pool_path, "rb") as file:
        Pool = pickle.load(file)
    # print(Pool)
    pool_length = 10000000
    for k, v in Pool.items():
        if len(v) < pool_length:
            pool_length = len(v)
    print(pool_length)
    if method == "full":
        for num in range(generate_num):
            for order in range(0, pool_length, batch_size):
                if order >= 0:
                    for k, v in Pool.items():
                        # if order==50 and num==0:
                        #    print(k)
                        prompt = v[order:order + batch_size]
                        imgs = get_img(pipe, 512, prompt=prompt)
                        for j, img in enumerate(imgs):
                            save_path_ = os.path.join(save_path, method, k, str(order + j))
                            # save_path1 = os.path.join(save_path, "full1", k)
                            if not os.path.exists(save_path_):
                                os.makedirs(save_path_)
                            # if not os.path.exists(save_path1):
                            #    os.makedirs(save_path1)
                            img = img.resize((224, 224))
                            img.save(
                                os.path.join(save_path_, "{}_{}.png".format(k, float(num * pool_length + order + j))))
                            # img.save(
                            #    os.path.join(save_path1, "{}_{}.png".format(k, int(num * pool_length + order + j))))

    if method == "ours":
        with open(learned_path, "rb") as file:
            learned_prompts = pickle.load(file)["prompts"]
            length = 20
        for num in range(generate_num):
            for order in range(0, length, batch_size):
                for k, v in learned_prompts.items():
                    prompt = v[order:order + batch_size]
                    imgs = get_img(pipe, 512, prompt=prompt)
                    for j, img in enumerate(imgs):
                        save_path_ = os.path.join(save_path, "ours", k)
                        if not os.path.exists(save_path_):
                            os.makedirs(save_path_)
                        img = img.resize((224, 224))
                        img.save(os.path.join(save_path_, "{}_{}.png".format(k, float(num * length + order + j))))
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    method = sys.argv[1]
    dataset = sys.argv[2]
    generate_num = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    main(method, dataset, generate_num, batch_size)
