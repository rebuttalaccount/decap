import pickle
import random

import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop, \
    InterpolationMode
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch.multiprocessing import Process, Queue
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

from nets.GA import GA
from nets.sdxl_turbo import StableDiffusionXLPipeline
from utils import *
import collections
from training_args import parse_args


def freeze_BN(model):
    for module in model.modules():
        # print(module)
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
    return model


def maybe_dictionarize(batch):
    if isinstance(batch, dict):
        return batch

    if len(batch) == 2:
        batch = {'images': batch[0], 'labels': batch[1]}
    elif len(batch) == 3:
        batch = {'images': batch[0], 'labels': batch[1], 'metadata': batch[2]}
    else:
        raise ValueError(f'Unexpected number of elements: {len(batch)}')

    return batch


def get_features(image_encoder, dataloader, device):
    all_data = collections.defaultdict(list)

    image_encoder = image_encoder.to(device)
    image_encoder.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = maybe_dictionarize(batch)
            features = image_encoder(batch['images'].to(device))

            all_data['features'].append(features.cpu())

            for key, val in batch.items():
                if key == 'images':
                    continue
                if hasattr(val, 'cpu'):
                    val = val.cpu()
                    all_data[key].append(val)
                else:
                    all_data[key].extend(val)

    for key, val in all_data.items():
        if torch.is_tensor(val[0]):
            all_data[key] = torch.cat(val).numpy()

    return all_data


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, image_encoder, dataloader, device):
        self.data = get_features(image_encoder, dataloader, device)

    def __len__(self):
        return len(self.data['features'])

    def __getitem__(self, idx):
        data = {k: v[idx] for k, v in self.data.items()}
        data['features'] = torch.from_numpy(data['features']).float()
        return data['features'], data['labels']


def get_dataloader(dataloader, is_train, args, device, image_encoder=None):
    if image_encoder is not None:
        feature_dataset = FeatureDataset(image_encoder, dataloader, device)
        new_dataloader = torch.utils.data.DataLoader(feature_dataset, batch_size=args.inner_batch_size,
                                                     shuffle=is_train)
    else:
        new_dataloader = dataloader
    return new_dataloader


def index_to_input(curriculum, classes, prompt_nums_per_class, num_classes):
    # shape:num_class*per_class_num
    curriculum = np.resize(curriculum, [num_classes, prompt_nums_per_class]).tolist()
    inputs = {classes[i]: curriculum[i] for i in range(len(classes))}
    # print(inputs)
    return inputs


class learner_dataGenerator(torch.utils.data.Dataset):
    def __init__(self, curriculum, label_list, root_dir, preprocess=None):
        self.curriculum = curriculum
        self.label_list = label_list
        self.image_paths = []
        self.labels = []
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        if preprocess is not None:
            self.preprocess = preprocess
        else:
            self.preprocess = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])

        for class_name, indexes in curriculum.items():
            for index in indexes:
                class_dir = os.path.join(root_dir, class_name, str(int(index)))
                count = 0
                for filename in os.listdir(class_dir):
                    if count < 5:
                        count += 1
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                            self.image_paths.append(os.path.join(class_dir, filename))
                            self.labels.append(label_list.index(class_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])
        label = self.labels[idx]

        img = self.preprocess(img)

        return img, label


class learner_dataGenerator_real(torch.utils.data.Dataset):
    def __init__(self, curriculum, label_list, root_dir, val_lines, preprocess=None):
        self.curriculum = curriculum
        self.label_list = label_list
        self.image_paths = []
        self.labels = []
        self.val_lines = val_lines
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        if preprocess is not None:
            self.preprocess = preprocess
        else:
            self.preprocess = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])

        for class_name, indexes in curriculum.items():
            for index in indexes:
                class_dir = os.path.join(root_dir, class_name, str(int(index)))
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                        self.image_paths.append(os.path.join(class_dir, filename))
                        self.labels.append(label_list.index(class_name))

    def __len__(self):
        return len(self.image_paths) + len(self.val_lines)

    def __getitem__(self, index):
        if index < len(self.image_paths):
            image = Image.open(self.image_paths[index])

            image = self.preprocess(image)
            return image, int(self.labels[index])
        else:
            annotation_path = self.val_lines[index - len(self.image_paths)].split(';')[1].strip()
            image = Image.open(annotation_path)
            image = self.preprocess(image)
            y = int(self.val_lines[index - len(self.image_paths)].split(';')[0])
            return image, y


class dataGenerator(torch.utils.data.Dataset):
    def __init__(self, preprocess, lines):
        self.lines = lines
        self.preprocess = preprocess

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        annotation_path = self.lines[index].split(';')[1].strip()
        image = Image.open(annotation_path)
        # image = image.resize((32, 32))
        image = self.preprocess(image)
        y = int(self.lines[index].split(';')[0])
        return image, y


def save_curriculum(ga, curriculum, fitness, pool, args, save_dir='curriculum', model=None):
    if not os.path.exists(os.path.join(tlogger.get_dir(), save_dir)):
        os.makedirs(os.path.join(tlogger.get_dir(), save_dir))
    inputs = index_to_input(ga.best_x, args.classes, args.prompt_nums_per_class, len(args.classes))
    prompts = {}
    for k, v in inputs.items():
        class_prompts = [pool[k][int(i)] for i in v]
        prompts[k] = class_prompts
    # print(prompts)
    outputs_dict = {"generation_best_Y": ga.generation_best_Y, "generation_best_X": ga.generation_best_X,
                    "all_history_FitV": ga.all_history_FitV, "all_history_Y": ga.all_history_Y,
                    "best_y": ga.best_y, "best_x": ga.best_x, "Chrom": ga.Chrom, "prompts": prompts}
    if model is not None:
        prompt_dict = model(curriculum, return_prompt_only=True)
        with open(os.path.join(tlogger.get_dir(), save_dir, 'prompt_dict.pkl'), 'wb') as file:
            pickle.dump(prompt_dict, file)
            tlogger.info("Saved:", os.path.join(tlogger.get_dir(), save_dir, 'prompt_dict.pkl'))
    else:
        with open(os.path.join(tlogger.get_dir(), save_dir, 'fitness{}.pkl'.format(fitness)), 'wb') as file:
            pickle.dump(outputs_dict, file)
            tlogger.info("Saved:", os.path.join(tlogger.get_dir(), save_dir, 'fitness{}.pkl'.format(fitness)))


def gpu_worker(gpu_id, input_queue, output_queue, completion_queue, learner_sampler_args, args, generator_args):
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)

    testset_x, testset_y, meta_x, meta_y, train_lines, train_preprocess = prepare_data(args)
    meta_x = meta_x.to(device)
    meta_y = meta_y.to(device)
    args.train_lines = train_lines
    args.train_preprocess = train_preprocess
    if device == torch.device(f'cuda:0'):
        print("meta-train dataset size :", meta_y.shape[0])
        print("test dataset size :", testset_y.shape[0])

    # generator = stable_diffusion(device=device, **generator_args)
    # generator.eval()  # Set to evaluation mode.

    learner_sampler = Learner_sampler(
        **learner_sampler_args
    )
    learner_sampler.interval = [8, 15]

    learner = learner_sampler.sample_learner(args.img_shape, device, learner_type=args.learner_type,
                                             model_path=args.model_path, num_classes=args.num_classes)

    if learner.feature_net is not None:
        meta_batch = math.ceil(meta_x.shape[0] / args.split)
        learner.feature_net.to(device)
        m_x = []
        with torch.no_grad():
            for i in range(0, meta_x.shape[0], meta_batch):
                batch_meta_x = meta_x[i:i + meta_batch]
                batch_meta_x = learner.feature_net(batch_meta_x)
                m_x.append(batch_meta_x)
        meta_x = torch.cat(m_x)
    del learner

    while True:
        inputs = input_queue.get()
        if inputs is None:
            break  # Exit loop and terminate process if None is received.

        outputs = [gpu_id]
        y = []
        for i in range(inputs.shape[0]):
            output = eval_fun(device, inputs[i], learner_sampler, meta_x=meta_x, meta_y=meta_y, args=args)
            y.append(output)
        outputs.append(y)
        output_queue.put(outputs)  # Send outputs back to the main process.
        completion_queue.put(1)  # Signal completion of an inference task.


# inner loop
def compute_learner(device, learner, curriculum, args, generate_num=None, callback=None, epoches=20,
                    iter_seed=42):
    learner.model.train()

    torch.manual_seed(iter_seed)

    inputs = index_to_input(curriculum, args.classes, args.prompt_nums_per_class, len(args.classes))

    epoch_loss = [0 for _ in range(epoches)]
    epoch_accuracy = [0 for _ in range(epoches)]
    min_loss = 1000000
    max_epoch_accuracy = 0
    learner.model.train()
    torch.cuda.empty_cache()
    # print(learner.optimizer)
    try:
        learner.model.freeze_backbone()
    except:
        learner.model = freeze_BN(learner.model)

    if not args.use_real_img:
        learner_dataset = learner_dataGenerator(inputs, args.label_list, args.root_dir, args.train_preprocess)
    else:
        learner_dataset = learner_dataGenerator_real(inputs, args.label_list, args.root_dir, args.train_lines,
                                                     args.train_preprocess)
    learner_data = torch.utils.data.DataLoader(learner_dataset, batch_size=args.inner_batch_size, shuffle=True,
                                               num_workers=4)
    learner_data = get_dataloader(learner_data, True, args, device, learner.feature_net)
    pbar = tqdm(total=args.epoches, desc="{} training epoch".format(device), postfix=dict, mininterval=0.3)
    for epoch in range(epoches):
        # if epoch == 25:
        #    learner.model.Unfreeze_backbone()

        for batch in learner_data:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            with torch.enable_grad():
                learner.optimizer.zero_grad()
                output = learner.model(x)
                loss = nn.CrossEntropyLoss()(output, y)
                loss.backward()
            learner.optimizer.step()

            epoch_loss[epoch] = epoch_loss[epoch] + loss.detach().cpu().item() * y.shape[0]
            epoch_accuracy[epoch] = epoch_accuracy[epoch] + (output.max(-1).indices == y).to(
                torch.float).mean().item() * y.shape[0]
        if epoch_loss[epoch] < min_loss:
            min_loss = epoch_loss[epoch]
            # new_params = copy.deepcopy(learner.model.state_dict())
            max_epoch_accuracy = epoch_accuracy[epoch]
        # print(np.array(epoch_loss)/len(learner_dataset))
        learner.scheduler.step()
        pbar.update(1)
    pbar.close()
    losses = min_loss / len(learner_dataset)
    accuracies = max_epoch_accuracy / len(learner_dataset)
    if callback is not None:
        callback(learner)

    # learner.model.load_state_dict(new_params)

    return learner, losses, accuracies


# outer_loop
def eval_fun(device, curriculum, learner_sampler, meta_x, meta_y, args, return_learner=False,
             generate_num=None):
    torch.manual_seed(args.seed)
    learner = learner_sampler.sample_learner(args.img_shape, device, learner_type=args.learner_type,
                                             model_path=args.model_path, num_classes=args.num_classes)

    torch.cuda.empty_cache()

    meta_x = meta_x.to(device)
    meta_y = meta_y.to(device)

    # inner
    learner, intermediate_losses, intermediate_accuracies = compute_learner(device=device,
                                                                            epoches=args.epoches,
                                                                            learner=learner, curriculum=curriculum,
                                                                            args=args, generate_num=generate_num,
                                                                            iter_seed=args.seed)

    torch.cuda.empty_cache()
    learner.model.eval()
    accuracy = torch.Tensor([0])
    if args.meta_loss_type == 'average':
        meta_loss = []
        meta_batch = math.ceil(meta_y.shape[0] / args.split)
        for i in range(0, meta_y.shape[0], meta_batch):
            batch_meta_x = meta_x[i:i + meta_batch]
            batch_meta_y = meta_y[i:i + meta_batch]
            # if learner.feature_net is not None:
            #     batch_meta_x = learner.feature_net(batch_meta_x)
            pred = learner.model(batch_meta_x).detach()
            m_loss = nn.CrossEntropyLoss()(pred, batch_meta_y)
            meta_loss.append(m_loss.item())
            accuracy += torch.mean(
                (torch.argmax(F.softmax(pred, dim=-1), dim=-1) == batch_meta_y).type(torch.FloatTensor)) / args.split
        meta_loss = sum(meta_loss) / len(meta_loss)
    elif args.meta_loss_type == 'accuracy':
        meta_loss = []
        for i in range(args.split):
            batch_meta_x = meta_x[
                           i * meta_y.shape[0] // args.split:i * meta_y.shape[0] // args.split + meta_x.shape[
                               0] // args.split]
            batch_meta_y = meta_y[
                           i * meta_y.shape[0] // args.split:i * meta_y.shape[0] // args.split + meta_y.shape[
                               0] // args.split]
            # if learner.feature_net is not None:
            #     batch_meta_x = learner.feature_net(batch_meta_x)
            pred = learner.model(batch_meta_x).detach()
            accuracy += torch.mean(
                (torch.argmax(F.softmax(pred, dim=-1), dim=-1) == batch_meta_y).type(torch.FloatTensor)) / args.split
            meta_loss = 100 - accuracy.item() * 100
    elif args.meta_loss_type == 'eval':
        meta_loss, _, accuracy = evaluate_set(learner.model, meta_x, meta_y, "val")
    else:
        raise NotImplementedError()

    # num_parameters = sum(p.numel() for p in learner.model.parameters())
    # tlogger.info("TrainingLearnerParameters", num_parameters)
    # tlogger.info("optimizer", type(learner.optimizer).__name__)
    tlogger.info('meta_training_loss', meta_loss)
    tlogger.info('meta_training_accuracy', accuracy.item())
    tlogger.info('training_accuracies', intermediate_accuracies)
    tlogger.info('training_losses', intermediate_losses)
    if return_learner:
        return meta_loss, learner
    else:
        return meta_loss


def _convert_to_rgb(image):
    return image.convert('RGB')


def _transform(n_px: int, is_train: bool):
    normalize = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                 (0.26862954, 0.26130258, 0.27577711))
    if is_train:
        return Compose([
            RandomResizedCrop(n_px, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
    else:
        return Compose([
            Resize(n_px, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(n_px),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])


def prepare_data(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.learner_type == "clip" or "clip_head":
        train_preprocess = _transform(224, True)
        test_preprocess = _transform(224, False)
    else:
        train_preprocess = test_preprocess = None
    # Load dataset
    with open(args.val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    with open(args.test_annotation_path, encoding='utf-8') as f:
        test_lines = f.readlines()
    with open(args.train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(val_lines)
    np.random.seed(None)
    val_dataset = dataGenerator(test_preprocess, val_lines)
    test_dataset = dataGenerator(test_preprocess, test_lines)
    data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=True, drop_last=True,
                                              num_workers=4, pin_memory=False)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, drop_last=True,
                                            num_workers=4, pin_memory=False)

    testset_x, testset_y = zip(*test_data)
    testset_x = torch.cat(testset_x)
    testset_y = torch.cat(testset_y)
    meta_x, meta_y = zip(*data_loader)
    meta_x = torch.cat(meta_x)
    meta_y = torch.cat(meta_y)

    return testset_x, testset_y, meta_x, meta_y, train_lines, train_preprocess


def get_label_list(args):
    if args.dataset == 'cifar10':
        label_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif args.dataset == 'stl10':
        label_list = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
    elif args.dataset == 'animal10':
        label_list = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']
    elif args.dataset == 'imagenette':
        label_list = ['tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn',
                      'garbage truck', 'gas pump', 'golf ball', 'parachute']
    elif args.dataset == 'imagewoof':
        label_list = ['Shih-Tzu', 'Rhodesian ridgeback', 'beagle', 'English foxhound', 'Border terrier',
                      'Australian terrier', 'golden retriever', 'Old English sheepdog', 'Samoyed', 'dingo']
    else:
        label_path = args.label_path
        with open(label_path, 'r') as file:
            label_list = [line.strip().replace("_", " ") for line in file]
    return label_list


def main(args):
    if not os.path.exists(os.path.join(tlogger.get_dir(), args.save_dir)):
        os.makedirs(os.path.join(tlogger.get_dir(), args.save_dir))

    label_list = get_label_list(args)
    args.label_list = label_list
    if args.label_path_to_opt is None:
        args.classes = label_list
    else:
        with open(args.label_path_to_opt, 'r') as file:
            classes = [line.strip().replace("_", " ") for line in file]
        args.classes = classes

    with open(args.pool_path, "rb") as file:
        pool = pickle.load(file)
    pool_size = len(pool[label_list[0]])
    print("pool size:", pool_size)
    args.num_classes = len(label_list)

    generator_args = {'generator': args.model_id,
                      'generator_batch': args.generator_batch,
                      'noise_size': args.img_shape,
                      'label_list': label_list,
                      'guidance_scale_learn': args.guidance_scale_learn,
                      'resize': args.resize,
                      'scale': args.scale,
                      'pool': pool,
                      'clip_clean': args.clip_clean,
                      'num_classes': len(label_list),
                      'prompt_nums_per_class': args.prompt_nums_per_class,
                      'save_dir': args.save_dir,
                      'separate': args.separate,
                      'common_prompt_nums': args.common_prompt_nums}

    SGD_list = [args.inner_loop_lr, args.inner_loop_momentum]
    RMS_list = [args.inner_loop_lr, args.adam_beta1, args.inner_loop_momentum, args.adam_epsilon]
    Adam_list = [args.inner_loop_lr, args.adam_beta1, args.adam_beta2, args.adam_epsilon]
    learner_sampler_args = {"inner_loop_optimizer": args.inner_loop_optimizer,
                            "SGD_list": SGD_list,
                            "RMS_list": RMS_list,
                            "Adam_list": Adam_list}

    num_gpus = torch.cuda.device_count()
    processes = []
    input_queues = []
    output_queue = Queue()
    completion_queue = Queue()  # Queue to track completion.

    for gpu_id in range(num_gpus):
        input_queue = Queue()
        p = Process(target=gpu_worker,
                    args=(gpu_id, input_queue, output_queue, completion_queue, learner_sampler_args, args,
                          generator_args))
        p.start()
        processes.append(p)
        input_queues.append(input_queue)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    def ga_algorithm(learned_dict=None):
        if learned_dict is not None:
            init_X = learned_dict.get("best_x", None)
            Chrom = learned_dict.get("Chrom", None)
            init_X = np.array([init_X, [128 for _ in range(len(init_X))]])
        else:
            init_X = None
            Chrom = None
        if not args.separate:
            n_dim = args.prompt_nums_per_class * len(args.classes)
            lb = [0 for _ in range(n_dim)]
            ub = [pool_size - 1 for _ in range(n_dim)]
        else:
            n_dim = args.prompt_nums_per_class * len(label_list) + args.common_prompt_nums
            lb, ub = [], []
            for i in range(n_dim):
                if i < args.prompt_nums_per_class * len(label_list):
                    lb.append(0)
                    ub.append(127)
                else:
                    lb.append(127)
                    ub.append(pool_size - 1)
        tlogger.info("optimization dim:", n_dim)
        if args.guidance_scale_learn:
            lb = lb + [0]
            ub = ub + [1]
            precision = n_dim * [1] + [1e-3]
            ga = GA(func=eval_fun, n_dim=n_dim + 1, size_pop=args.per_gpu_popsize * num_gpus, init_X=init_X,
                    max_iter=args.maxiter, prob_mut=0.002, lb=lb, ub=ub, precision=precision)
        else:
            ga = GA(func=eval_fun, n_dim=n_dim, size_pop=args.per_gpu_popsize * num_gpus, init_X=init_X,
                    max_iter=args.maxiter, prob_mut=0.002, lb=lb, ub=ub, precision=1)
        # if Chrom is not None:
        #    ga.Chrom = Chrom

        best_x, best_y = None, None

        for i in range(args.maxiter):
            X = ga.get_X()
            for queue_order, input_queue in enumerate(input_queues):
                inputs = X[queue_order * args.per_gpu_popsize: (queue_order + 1) * args.per_gpu_popsize]
                input_queue.put(inputs)
            num_completed = 0
            while num_completed < num_gpus:
                completion_queue.get()  # Blocks until a GPU signals completion.
                num_completed += 1
            outputs = []
            while True:
                output = output_queue.get()
                outputs.append(output)
                if len(outputs) == num_gpus:
                    break
            outputs = sorted(outputs, key=lambda x: x[0])
            outputs = [item[1] for item in outputs]
            outputs = [item for sublist in outputs for item in sublist]
            outputs = np.array(outputs)
            ga.set_Y(outputs)
            best_x, best_y = ga.one_iter()
            print("\n\033[1;31;44m {} iteration best fitness is {}\033[0m".format((i + 1), best_y))
            if (i + 1) % args.validation_period == 0:
                save_path = args.save_dir + "/{}iteration".format((i + 1))
                save_curriculum(ga, best_x, best_y, pool, args, save_dir=save_path)
        tlogger.record_tabular("best fitness ", best_y)
        for input_queue in input_queues:
            input_queue.put(None)
        for p in processes:
            p.join()
        return best_x, best_y

    if args.learned_dict != '':
        with open(args.learned_dict, 'rb') as file:
            learned_dict = pickle.load(file)
        print(learned_dict["best_x"].shape)
    else:
        learned_dict = None
    xbest, fbest = ga_algorithm(learned_dict)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    if args.learner_type == "clip" or "clip_head":
        zeroshot_checkpoint = os.path.join("./clip_classifier", args.dataset, 'zeroshot' + '.pt')
        # if not os.path.exists(zeroshot_checkpoint):
        #    image_encoder = ImageEncoder("RN50", keep_lang=True)
        #    classification_head = get_zeroshot_classifier(image_encoder.model, label_list=get_label_list(args))
        #    delattr(image_encoder.model, 'transformer')
        #    classifier = ImageClassifier(image_encoder, classification_head, process_images=False)
        #    classifier.save(zeroshot_checkpoint)
        # else:
        #    print("already exists CLIP")
        args.model_path = zeroshot_checkpoint
    # prepare_data_outputs = prepare_data(args)
    main(args)
    # print('Training is over! The test accuracy is:', result)
