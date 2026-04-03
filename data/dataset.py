import os
import sys
import re
import six
import math
import lmdb
import torch
import random

from augmentation.weather import Fog, Snow, Frost, Rain, Shadow
from augmentation.warp import Curve, Distort, Stretch
from augmentation.geometry import Rotate, Perspective, Shrink, TranslateX, TranslateY
from augmentation.pattern import VGrid, HGrid, Grid, RectGrid, EllipseGrid
from augmentation.noise import GaussianNoise, ShotNoise, ImpulseNoise, SpeckleNoise
from augmentation.blur import GaussianBlur, DefocusBlur, MotionBlur, GlassBlur, ZoomBlur
from augmentation.camera import Contrast, Brightness, JpegCompression, Pixelate
from augmentation.process import Posterize, Solarize, Invert, Equalize, AutoContrast, Sharpness, Color

from natsort import natsorted
from PIL import Image
import PIL.ImageOps
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
from torch.utils.data import random_split

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data.distributed import DistributedSampler


class Batch_Balanced_Dataset(object):

    def __init__(self, opt):

        log = open(f'{opt.saved_path}/{opt.exp_name}/log_dataset.txt', 'a')
        dashed_line = '-' * 80
        print(dashed_line)
        log.write(dashed_line + '\n')
        print(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}')
        log.write(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}\n')
        assert len(opt.select_data) == len(opt.batch_ratio)

        _AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=opt)
        self.data_loader_list = []
        self.dataloader_iter_list = []
        batch_size_list = []
        Total_batch_size = 0
        for selected_d, batch_ratio_d in zip(opt.select_data, opt.batch_ratio):
            _batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)
            print(dashed_line)
            log.write(dashed_line + '\n')
            if selected_d == 'chinese':
                _dataset, _dataset_log = hierarchical_c_dataset(root=opt.train_data, opt=opt, select_data=[selected_d])
            else:
                _dataset, _dataset_log = hierarchical_dataset(root=opt.train_data, opt=opt, select_data=[selected_d])
            total_number_dataset = len(_dataset)
            log.write(_dataset_log)
            number_dataset = int(total_number_dataset * float(opt.total_data_usage_ratio))
            dataset_split = [number_dataset, total_number_dataset - number_dataset]

            indices = range(total_number_dataset)

            _dataset, _ = [Subset(_dataset, indices[offset - length:offset])
                           for offset, length in zip(_accumulate(dataset_split), dataset_split)]

            selected_d_log = f'num total samples of {selected_d}: {total_number_dataset} x {opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n'
            selected_d_log += f'num samples of {selected_d} per batch: {opt.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}'
            print(selected_d_log)
            log.write(selected_d_log + '\n')
            batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            train_sampler = DistributedSampler(_dataset)
            _data_loader = torch.utils.data.DataLoader(
                _dataset, batch_size=_batch_size,
                sampler=train_sampler,
                num_workers=int(opt.workers),
                collate_fn=_AlignCollate, pin_memory=True)

            print(f'Total number of samples in _dataset: {len(_dataset)}')
            print(f'_batch_size: {_batch_size}')
            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))

        Total_batch_size_log = f'{dashed_line}\n'
        batch_size_sum = '+'.join(batch_size_list)
        Total_batch_size_log += f'Total_batch_size: {batch_size_sum} = {Total_batch_size}\n'
        Total_batch_size_log += f'{dashed_line}'
        opt.batch_size = Total_batch_size

        print(Total_batch_size_log)
        log.write(Total_batch_size_log + '\n')
        log.close()

    def get_batch(self):
        balanced_batch_images = []
        balanced_batch_texts = []
        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image, text, _ = next(data_loader_iter)
                #print(f'Dataset {i}: Fetched image and text')
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, text, _ = next(self.dataloader_iter_list[i])
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except Exception as e:
                pass
        balanced_batch_images = torch.cat(balanced_batch_images, 0)

        return balanced_batch_images, balanced_batch_texts


def hierarchical_dataset(root, opt, select_data='/', val=False):
    """ select_data='/' contains all sub-directory of root directory """
    dataset_list = []
    if (select_data == '/' and opt.select_data != None):
        select_data = opt.select_data
    dataset_log = f'dataset_root:    {root}\t dataset: {select_data[0]}'
    print(dataset_log)
    dataset_log += '\n'
    for dirpath, dirnames, filenames in os.walk(root+'/'):
        if not dirnames:
            select_flag = False
            for selected_d in select_data:
                if selected_d in dirpath.split('/') + ['/']:
                    select_flag = True
                    break

            if select_flag:
                dataset = LmdbDataset(dirpath, opt)
                sub_dataset_log = f'sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}'
                dataset_log += f'{sub_dataset_log}\n'
                dataset_list.append(dataset)
    if val:
        dataset = ChineseDataset(root, opt, val)
        dataset_list.append(dataset)
    concatenated_dataset = ConcatDataset(dataset_list)
    print(dataset_log)
    return concatenated_dataset, dataset_log


class LmdbDataset(Dataset):

    def __init__(self, root, opt):

        self.root = root
        self.opt = opt

        print(f"-------root: {root}")
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False, map_size=1024*1024*1024*1024)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)
        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples
            if self.opt.data_filtering_off:
                self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
            else:

                self.filtered_index_list = []
                for index in range(self.nSamples):
                    index += 1  # lmdb starts with 1
                    label_key = 'label-%09d'.encode() % index
                    label = txn.get(label_key).decode('utf-8')
                    if len(label) > self.opt.batch_max_length:
                        continue
                    self.filtered_index_list.append(index)
        #
                self.nSamples = len(self.filtered_index_list)

                
    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                if self.opt.rgb:
                    img = Image.open(buf).convert('RGB')  # for color image
                else:
                    img = Image.open(buf).convert('L')

            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                if self.opt.rgb:
                    img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                else:
                    img = Image.new('L', (self.opt.imgW, self.opt.imgH))
                label = '[dummy_label]'


        return (img, label)

def hierarchical_c_dataset(root, opt, select_data='/'):
    """ select_data='/' contains all sub-directory of root directory """
    dataset_list = []
    dataset_log = f'dataset_root:    {root}\t dataset: {select_data[0]}'
    print(dataset_log)
    dir_path = os.path.join(root,select_data[0])
    dataset = ChineseDataset(dir_path, opt)
    dataset_list.append(dataset)

    concatenated_dataset = ConcatDataset(dataset_list)
    return concatenated_dataset, dataset_log

def readlines(filename):
    res = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in lines:
            res.append(i.strip())
    dic = {}
    for i in res:
        p = i.split(' ')
        dic[p[0]] = p[1:]
    return dic


class ChineseDataset(Dataset):

    def __init__(self, root, opt, val=False):

        self.root = root
        self.opt = opt
        self.imgpath = root + '/images'
        if val:
            self.root = root.replace('evaluation', 'training') + '/chinese'
            self.imgpath = self.root + '/images'
            self.imgs_labels = readlines(self.root + '/data_test.txt')
        else:
            self.imgs_labels = readlines(root + '/data_train.txt')

        self.imgfiles = [i for i, j in self.imgs_labels.items()]

        self.nSamples = len(self.imgs_labels)


    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        try:
            img_name = self.imgfiles[index]
            if self.opt.rgb:
                img = Image.open(os.path.join(self.imgpath, img_name)).convert('RGB')  # for color image
            else:
                img = Image.open(os.path.join(self.imgpath, img_name)).convert('L')

            label = self.imgs_labels[img_name]
        except IOError:
            print(f'Corrupted image for {index}')
            # make dummy image and dummy label for corrupted image.
            if self.opt.rgb:
                img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
            else:
                img = Image.new('L', (self.opt.imgW, self.opt.imgH))
            label = '[dummy_label]'

        return (img, label)


class RawDataset(Dataset):

    def __init__(self, root, opt):
        self.opt = opt
        self.image_path_list = []
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                _, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                    self.image_path_list.append(os.path.join(dirpath, name))

        self.image_path_list = natsorted(self.image_path_list)
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        try:
            if self.opt.rgb:
                img = Image.open(self.image_path_list[index]).convert('RGB')  # for color image
            else:
                img = Image.open(self.image_path_list[index]).convert('L')

        except IOError:
            print(f'Corrupted image for {index}')
            # make dummy image and dummy label for corrupted image.
            if self.opt.rgb:
                img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
            else:
                img = Image.new('L', (self.opt.imgW, self.opt.imgH))

        return (img, self.image_path_list[index])


def isless(prob=0.5):
    return np.random.uniform(0,1) < prob

class DataAugment(object):
    '''
    Supports with and without data augmentation 
    '''
    def __init__(self, opt):
        self.opt = opt

        if not opt.eval:
            self.process = [Posterize(), Solarize(), Invert(), Equalize(), AutoContrast(), Sharpness(), Color()]
            self.camera = [Contrast(), Brightness(), JpegCompression(), Pixelate()

]

            self.pattern = [VGrid(), HGrid(), Grid(), RectGrid(), EllipseGrid()]

            self.noise = [GaussianNoise(), ShotNoise(), ImpulseNoise(), SpeckleNoise()]
            self.blur = [GaussianBlur(), DefocusBlur(), MotionBlur(), GlassBlur(), ZoomBlur()]
            self.weather = [Fog(), Snow(), Frost(), Rain(), Shadow()]

            self.noises = [self.blur, self.noise, self.weather]
            self.processes = [self.camera, self.process]

            self.warp = [Curve(), Distort(), Stretch()]
            self.geometry = [Rotate(), Perspective(), Shrink()]

            self.isbaseline_aug = False
            # rand augment
            if self.opt.isrand_aug:
                self.augs = [self.process, self.camera, self.noise, self.blur, self.weather, self.pattern, self.warp, self.geometry]
            # semantic augment
            elif self.opt.issemantic_aug:
                self.geometry = [Rotate(), Perspective(), Shrink()]
                self.noise = [GaussianNoise()]
                self.blur = [MotionBlur()]
                self.augs = [self.noise, self.blur, self.geometry]
                self.isbaseline_aug = True
            # pp-ocr augment
            elif self.opt.islearning_aug:
                self.geometry = [Rotate(), Perspective()]
                self.noise = [GaussianNoise()]
                self.blur = [MotionBlur()]
                self.warp = [Distort()]
                self.augs = [self.warp, self.noise, self.blur, self.geometry]
                self.isbaseline_aug = True
            # scatter augment
            elif self.opt.isscatter_aug:
                self.geometry = [Shrink()]
                self.warp = [Distort()]
                self.augs = [self.warp, self.geometry]
                self.baseline_aug = True
            # rotation augment
            elif self.opt.isrotation_aug:
                self.geometry = [Rotate()]
                self.augs = [self.geometry]
                self.isbaseline_aug = True

        self.scale = False if opt.Transformer else True

    def __call__(self, img):
        '''
            Must call img.copy() if pattern, Rain or Shadow is used
        '''
        if self.opt.eval or isless(self.opt.intact_prob):
            pass
        elif self.opt.isrand_aug or self.isbaseline_aug:
            img = self.rand_aug(img)
        # individual augment can also be selected
        elif self.opt.issel_aug:
            img = self.sel_aug(img)

        img = transforms.ToTensor()(img)

        if self.opt.rgb:
            img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        else:

            img = transforms.Normalize(mean=[0.5], std=[0.5])(img)
            
        return img

    def rand_aug(self, img):
        # 扁平化 self.augs
        flat_augs = [aug for sublist in self.augs for aug in sublist]

        augs = np.random.choice(flat_augs, self.opt.augs_num, replace=False)
        for aug in augs:
            mag = np.random.randint(0, 3) if self.opt.augs_mag is None else self.opt.augs_mag
            if isinstance(aug, (Rain, Grid)):
                img = aug(img.copy(), mag=mag)
            else:
                img = aug(img, mag=mag)

        return img

    def sel_aug(self, img):

        prob = 1.
        if self.opt.process:
            mag = np.random.randint(0, 3)
            index = np.random.randint(0, len(self.process))
            op = self.process[index]
            img = op(img, mag=mag, prob=prob)

        if self.opt.noise:
            mag = np.random.randint(0, 3)
            index = np.random.randint(0, len(self.noise))
            op = self.noise[index]
            img = op(img, mag=mag, prob=prob)

        if self.opt.blur:
            mag = np.random.randint(0, 3)
            index = np.random.randint(0, len(self.blur))
            op = self.blur[index]
            img = op(img, mag=mag, prob=prob)

        if self.opt.weather:
            mag = np.random.randint(0, 3)
            index = np.random.randint(0, len(self.weather))
            op = self.weather[index]
            if isinstance(op, Rain):
                img = op(img.copy(), mag=mag, prob=prob)
            else:
                img = op(img, mag=mag, prob=prob)

        if self.opt.camera:
            mag = np.random.randint(0, 3)
            index = np.random.randint(0, len(self.camera))
            op = self.camera[index]
            img = op(img, mag=mag, prob=prob)

        if self.opt.pattern:
            mag = np.random.randint(0, 3)
            index = np.random.randint(0, len(self.pattern))
            op = self.pattern[index]
            img = op(img.copy(), mag=mag, prob=prob)

        iscurve = False
        if self.opt.warp:
            mag = np.random.randint(0, 3)
            index = np.random.randint(0, len(self.warp))
            op = self.warp[index]
            if isinstance(op, Curve):
                iscurve = True
            img = op(img, mag=mag, prob=prob)

        if self.opt.geometry:
            mag = np.random.randint(0, 3)
            index = np.random.randint(0, len(self.geometry))
            op = self.geometry[index]
            if isinstance(op, Rotate):
                img = op(img, iscurve=iscurve, mag=mag, prob=prob)
            else:
                img = op(img, mag=mag, prob=prob)

        return img



class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False, opt=None):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.opt = opt

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        if self.opt.eval_img:
            images, labels, img_paths = zip(*batch)
        else:
            images, labels = zip(*batch)
            img_paths = None

        pil = transforms.ToPILImage()
        aug_transform = DataAugment(self.opt)

        image_tensors = [aug_transform(image.resize((self.opt.imgW, self.opt.imgH), Image.BICUBIC)) for image in images]
        image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels, img_paths




class ImgDataset(Dataset):

    def __init__(self, root, opt):

        with open(root) as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        self.data_list = [x.strip() for x in content] 

        self.leng_index = [0] * 26
        self.leng_index.append(0)
        text_length = 1
        for i, line in enumerate(self.data_list):
            label_text = line.split(' ')[0]
            if i > 0 and len(label_text) != text_length:
                self.leng_index[text_length] = i
            text_length = len(label_text)
        
        self.nSamples = len(self.data_list)
        self.batch_size = opt.batch_size
        self.opt = opt

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        sample_ = self.data_list[index].split(' ')
        label = sample_[0]
        img_path = sample_[1]

        try:
            if self.opt.rgb:
                img = Image.open(img_path).convert('RGB')  # for color image
            else:
                img = Image.open(img_path).convert('L')
        except IOError:
            print('Corrupted read image for ', img_path)
            randi = random.randint(0, self.nSamples-1)
            return self.__getitem__(randi)

        if not self.opt.sensitive:
            label = label.lower() 

        return img, label, img_path

