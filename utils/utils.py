import os

import torch
import numpy as np
import argparse
from PIL import Image
from transformers import BertTokenizer, GPT2Tokenizer
from strsimpy.normalized_levenshtein import NormalizedLevenshtein

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TokenLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, opt):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        self.SPACE = '[s]'
        self.GO = '[GO]'

        self.list_token = [self.GO, self.SPACE]
        char_dic = os.getcwd() + '/data/dict/merged_char.txt'  # '/data/char_std_5990.txt' '/data/bert_base_chinese_vocab.txt'
        self.character = open(char_dic, 'r', encoding='utf-8').readlines()
        self.character = ''.join([ch.strip('\n') for ch in self.character] + ['卍'])
        self.character = self.list_token + list(self.character)
        # self.character = self.list_token + list(opt.character)

        self.dict = {word: i for i, word in enumerate(self.character)}
        self.batch_max_length = opt.batch_max_length + len(self.list_token)
        self.bpe_tokenizer = GPT2Tokenizer.from_pretrained(os.getcwd() + "/data/dict/gpt2")
        self.wp_tokenizer = BertTokenizer.from_pretrained(
            os.getcwd() + "/data/dict/bert_base_chinese_vocab.txt")  # ("bert-base-uncased")
        self.normalized_levenshtein = NormalizedLevenshtein()

    def encode(self, text):
        """ convert text-label into text-index.
        """
        batch_text = torch.LongTensor(len(text), self.batch_max_length).fill_(self.dict[self.GO])
        for i, t in enumerate(text):
            if isinstance(t, list):
                txt = [self.dict[self.GO]] + list(map(int, t)) + [self.dict[self.SPACE]]
                batch_text[i][:len(txt)] = torch.LongTensor(txt)
            else:
                txt = [self.GO] + list(t) + [self.SPACE]
                txt = [self.dict[char] if char in self.dict else self.dict['卍'] for char in txt]
                batch_text[i][:len(txt)] = torch.LongTensor(txt)  # batch_text[:, 0] = [GO] token
        return batch_text.to(device)

    def char_encode(self, text, return_length=False):
        """ convert text-label into text-index.
        """
        batch_len = torch.LongTensor(len(text), 2).fill_(self.dict[self.GO])
        batch_text = torch.LongTensor(len(text), self.batch_max_length).fill_(self.dict[self.GO])
        for i, t in enumerate(text):
            length = len(t)
            batch_len[i][1] = torch.LongTensor([length])  # batch_text[:, 0] = [GO] token
            if isinstance(t, list):
                txt = [self.dict[self.GO]] + list(map(int, t)) + [self.dict[self.SPACE]]
                batch_text[i][:len(txt)] = torch.LongTensor(txt)
            else:
                txt = [self.GO] + list(t) + [self.SPACE]
                txt = [self.dict[char] for char in txt]
                batch_text[i][:len(txt)] = torch.LongTensor(txt)

        if return_length:
            text_length = [len(i) for i in text]
            return batch_len.to(device), batch_text.to(device), torch.Tensor(text_length).to(device)

        return batch_len.to(device), batch_text.to(device)

    def char_decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts

    def bpe_encode(self, text, return_length=False):
        batch_text = torch.LongTensor(len(text), self.batch_max_length).fill_(self.dict[self.GO])
        for i, t in enumerate(text):
            if isinstance(t, list):
                t = ''.join([self.character[int(ch)] for ch in t])
            token = self.bpe_tokenizer(t)['input_ids']
            txt = [1] + token + [2]
            batch_text[i][:len(txt)] = torch.LongTensor(txt)

        if return_length:
            text_length = [len(self.bpe_tokenizer.tokenize(i)) for i in text]
            return batch_text.to(device), torch.Tensor(text_length).to(device)

        return batch_text.to(device)

    def bpe_decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            tokenstr = self.bpe_tokenizer.decode(text_index[index, :])
            texts.append(tokenstr)
        return texts

    def wp_encode(self, text, return_length=False):
        modify_text = []
        for t in text:
            if isinstance(t, list):
                t = ''.join([self.character[int(ch)] for ch in t])
            modify_text.append(t)
        wp_target = self.wp_tokenizer(modify_text, padding='max_length', max_length=self.batch_max_length,
                                      truncation=True, return_tensors="pt")

        if return_length:
            text_length = [len(self.wp_tokenizer.tokenize(i)) for i in text]
            return wp_target["input_ids"].to(device), torch.Tensor(text_length).to(device)

        return wp_target["input_ids"].to(device)

    def wp_decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            tokenstr = self.wp_tokenizer.decode(text_index[index, :])
            tokenlist = tokenstr.split()
            texts.append(''.join(tokenlist))
        return texts


class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


import matplotlib.pyplot as plt


def get_device(verbose=True):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if verbose:
        print("Device:", device)
    return device


def draw_mul_loss(loss, save_path, interval):
    # draw multiple loss curves. `loss` is a list of sequences.
    labels = ['char_loss', 'bpe_loss', 'wp_loss', 'fusion_loss', 'sum_loss']
    # sample by interval
    sampled = [l[::interval] for l in loss]
    # determine x axis length as the shortest series to avoid mismatch
    if len(sampled) == 0:
        return
    x_len = min(len(s) for s in sampled)
    x = list(range(x_len))
    fig, ax = plt.subplots()
    for i, series in enumerate(sampled):
        label = labels[i] if i < len(labels) else f'loss_{i}'
        ax.plot(x[:len(series[:x_len])], series[:x_len], label=label)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_title('Loss')
    ax.legend()
    fig.savefig(f'{save_path}/loss.jpg')


def draw_one_loss(loss, save_path, interval, label):
    # iter = np.linspace(0, len(loss)*interval, len(loss))
    new_loss = loss[::interval]
    iter = [i for i in range(len(new_loss))]
    fig, ax = plt.subplots()
    # print(iter,loss[::interval],len(loss))
    ax.plot(iter, loss[::interval], label=f'{label}_loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_title(f'{label}_Loss')
    ax.legend()
    fig.savefig(f'{save_path}/{label}_loss.jpg')


def draw_mul_acc(acc, save_path, interval):
    # draw multiple accuracy curves. `acc` is a list of sequences.
    labels = ['char_accuracy', 'bpe_accuracy', 'wp_accuracy', 'fusion_accuracy']
    plt.cla()
    sampled = [a[::interval] for a in acc]
    if len(sampled) == 0:
        return
    x_len = min(len(s) for s in sampled)
    x = list(range(x_len))
    fig, ax = plt.subplots()
    for i, series in enumerate(sampled):
        label = labels[i] if i < len(labels) else f'acc_{i}'
        ax.plot(x[:len(series[:x_len])], series[:x_len], label=label)
    ax.set_xlabel('epoch')
    ax.set_ylabel('accuracy')
    ax.set_title('Accuracy')
    ax.legend()
    fig.savefig(f'{save_path}/accuracy.jpg')


def draw_one_acc(acc, save_path, interval, label):
    # iter = np.linspace(0, len(acc)*interval, len(acc))
    plt.cla()
    new_acc = acc[::interval]
    iter = [i for i in range(len(new_acc))]
    fig, ax = plt.subplots()
    ax.plot(iter, acc[::interval], label=f'{label}_accuracy')
    ax.set_xlabel('epoch')
    ax.set_ylabel('accuracy')
    ax.set_title(f'{label}_accuracy')
    ax.legend()
    fig.savefig(f'{save_path}/{label}_accuracy.jpg')


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def get_args(is_train=True):
    parser = argparse.ArgumentParser(description='STR')

    # for test
    parser.add_argument('--eval_data', help='path to evaluation dataset')
    parser.add_argument('--benchmark_all_eval', action='store_true', help='evaluate 10 benchmark evaluation datasets')
    parser.add_argument('--calculate_infer_time', action='store_true', help='calculate inference timing')
    parser.add_argument('--flops', action='store_true', help='calculates approx flops (may not work)')

    # for train
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--train_data', required=is_train, help='path to training dataset')
    parser.add_argument('--valid_data', required=is_train, help='path to validation dataset')
    parser.add_argument('--test_data', required=is_train, help='path to test dataset')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers. Use -1 to use all cores.',
                        default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=300000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=2000, help='Interval between each validation')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")
    parser.add_argument('--saved_path', default='./saved_models', help="path to save")
    # fine-tuning / optimizer flags removed (unused)
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    """ Data processing """
    parser.add_argument('--select_data', type=str, default='MJ-ST',
                        help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    parser.add_argument('--batch_ratio', type=str, default='0.5-0.5',
                        help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--batch_max_length', type=int, default=120, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=128, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')

    """ Model Architecture """
    parser.add_argument('--Transformer', type=str, required=True, help='Transformer stage. mgp-str|char-str')

    choices = ["mgp_str_base_patch4_3_32_128", "mgp_str_tiny_patch4_3_32_128",
               "mgp_str_small_patch4_3_32_128", "char_str_base_patch4_3_32_128",
               "swin_small_patch4_window7_224", "swin_small_patch4_window7_224_fusion",
               "swin_small_patch4_window7_224_lister"]
    parser.add_argument('--TransformerModel', default='', help='Which mgp_str transformer model', choices=choices)
    parser.add_argument('--Transformation', type=str, default='', help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='',
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default='', help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default='', help='Prediction stage. None|CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=3,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    # selective augmentation
    # can choose specific data augmentation
    parser.add_argument('--issel_aug', action='store_true', help='Select augs')
    # selective augmentation probability removed (unused)
    parser.add_argument('--pattern', action='store_true', help='Pattern group')
    parser.add_argument('--warp', action='store_true', help='Warp group')
    parser.add_argument('--geometry', action='store_true', help='Geometry group')
    parser.add_argument('--weather', action='store_true', help='Weather group')
    parser.add_argument('--noise', action='store_true', help='Noise group')
    parser.add_argument('--blur', action='store_true', help='Blur group')
    parser.add_argument('--camera', action='store_true', help='Camera group')
    parser.add_argument('--process', action='store_true', help='Image processing routines')

    # use cosine learning rate decay
    parser.add_argument('--scheduler', action='store_true', help='Use lr scheduler')

    parser.add_argument('--intact_prob', type=float, default=0.5, help='Probability of not applying augmentation')
    parser.add_argument('--isrand_aug', action='store_true', help='Use RandAug')
    parser.add_argument('--augs_num', type=int, default=3, help='Number of data augment groups to apply. 1 to 8.')
    parser.add_argument('--augs_mag', type=int, default=None,
                        help='Magnitude of data augment groups to apply. None if random.')

    # (removed some experimental/unused augmentation flags)

    # orig paper uses this for fast benchmarking (flag removed, not used)

    # local_rank removed (not referenced)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    # mask train flags removed (unused)
    
    parser.add_argument('--use_fpn', action='store_true', help='Whether to use FPN')
    parser.add_argument('--use_stages', action='store_true', help='Whether to use STAGES')
    parser.add_argument('--use_sk', action='store_true', help='Whether to use SK Unit')

    # for eval
    parser.add_argument('--eval_img', action='store_true', help='eval imgs dataset')
    parser.add_argument('--range', default=None, help="start-end for example(800-1000)")
    parser.add_argument('--model_dir', default='')
    parser.add_argument('--demo_imgs', default='')

    args = parser.parse_args()

    # control whether use SK after Patch Merging in SWIN-STR/modules/models/swin_transformer.py:398
    if args.use_sk:
        os.environ['use_SK'] = 'True'

    return args


# module end
