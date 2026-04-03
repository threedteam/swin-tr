import os
import time
import random
import string
import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np
import copy

from utils.utils import Averager, TokenLabelConverter, get_args, draw_one_loss, draw_one_acc, draw_mul_loss, draw_mul_acc
from data.dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from modules.model import Model
from test import validation
import utils.utils_dist as utils

lister_loss = utils.ListerLoss()

import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

import torch

# evaluation statistics utilities
from utils import results_statistics

import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(opt):
    """ dataset preparation """
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')
        # see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L130

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    opt.eval = False
    # Balanced loading for multiple training datasets
    train_dataset = Batch_Balanced_Dataset(opt)

    log = open(f'{opt.saved_path}/{opt.exp_name}/log_dataset.txt', 'a')

    val_opt = copy.deepcopy(opt)
    val_opt.eval = True
    
    if opt.sensitive:
        opt.data_filtering_off = True
    # Prepare a batch of images and labels into a model-ready format
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=val_opt)
    valid_dataset, _ = hierarchical_dataset(root=opt.valid_data, opt=val_opt, select_data=["inno"])
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
    
    # Create test dataset (same settings as validation but using test path)
    test_opt = copy.deepcopy(opt)
    test_opt.eval = True
    AlignCollate_test = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=test_opt)
    test_dataset, _ = hierarchical_dataset(root=opt.test_data, opt=test_opt, select_data=["inno"])
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size,
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_test, pin_memory=True)
    
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()
        
    """ model configuration """
    converter = TokenLabelConverter(opt)  # TokenLabelConverter: converts between text and indices for STR
        
    opt.num_class = len(converter.character)

    # Get padding/GO token index
    padding_token_index = converter.dict[converter.GO]
    print(f"DEBUG: Padding/GO token index is: {padding_token_index}")

    if opt.rgb:
        opt.input_channel = 3

    model = Model(opt)

    # Write model architecture to model.txt
    # print(model)
    with open(f'{opt.saved_path}/{opt.exp_name}/model.txt', 'w') as model_file:
        model_file.write(str(model))

    # data parallel for multi-GPU
    model.to(device)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu], find_unused_parameters=True)
    
    model.train()
    
    # Prepare for potential checkpoint loading. We'll try to support both legacy
    # files that contain only state_dict and new-style checkpoints with optimizer/scheduler.
    ckpt = None
    if opt.saved_model != '':
        print(f'loading pretrained model/checkpoint from {opt.saved_model}')
        try:
            ckpt = torch.load(opt.saved_model, map_location='cpu')
            # If ckpt is a dict and contains 'model' or similar, extract state_dict
            if isinstance(ckpt, dict):
                if 'model' in ckpt:
                    model.load_state_dict(ckpt['model'], strict=True)
                elif 'state_dict' in ckpt:
                    model.load_state_dict(ckpt['state_dict'], strict=True)
                else:
                    # assume it's a raw state_dict
                    try:
                        model.load_state_dict(ckpt, strict=True)
                    except Exception:
                        # if loading fails, try matching keys (remove module. prefix)
                        sd = {k.replace('module.', ''): v for k, v in ckpt.items()}
                        model.load_state_dict(sd, strict=False)
            else:
                # legacy: ckpt is a state_dict
                model.load_state_dict(ckpt, strict=False)
            print('Loaded model weights.')
        except Exception as e:
            print(f'Failed to fully load checkpoint model weights: {repr(e)}')

    """ setup loss """
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0

    # loss averager
    loss_avg = Averager()
    traincharLoss = []
    trainbpeLoss = []
    trainwpLoss = []
    trainfusionLoss = [0]
    trainsumLoss = []

    valchar_accuracy = []
    valbpe_accuracy = []
    valwp_accuracy = []
    valfusion_accuracy = []
    valsumLoss = []

    # Test set accuracy trackers
    testchar_accuracy = []
    testbpe_accuracy = []
    testwp_accuracy = []
    testfusion_accuracy = []
    testsumLoss = []

    # Checkpoint saving settings
    save_per_epoch = 1  # save every N validation cycles
    save_count = 0  # number of saved checkpoints
    # save_count_max = 4  # maximum number of saved checkpoints (disabled)

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))

    # setup optimizer
    scheduler = None
    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
        #optimizer = optim.RMSprop(filtered_parameters,lr=opt.lr, eps=opt.eps)

    if opt.scheduler:
        # Example: LambdaLR schedule (custom)
        M = 0.1  # constant M
        q = 0.5  # exponent q
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: M * (epoch + 1) ** (-q))

    if ckpt is not None and isinstance(ckpt, dict):
        if 'optimizer' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer'])
                print('Loaded optimizer state from checkpoint.')
            except Exception as e:
                print(f'Could not load optimizer state: {repr(e)}')
        if 'scheduler' in ckpt and scheduler is not None and ckpt['scheduler'] is not None:
            try:
                scheduler.load_state_dict(ckpt['scheduler'])
                print('Loaded scheduler state from checkpoint.')
            except Exception as e:
                print(f'Could not load scheduler state: {repr(e)}')
        if 'iteration' in ckpt:
            try:
                start_iter = int(ckpt['iteration'])
                iteration = start_iter
                print(f'Resuming from iteration {start_iter} (from checkpoint)')
            except Exception:
                pass

    """ final options """
    # print(opt)
    with open(f'{opt.saved_path}/{opt.exp_name}/opt.txt', 'w') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        #print(opt_log)
        opt_file.write(opt_log)
        total_params = int(sum(params_num))
        total_params = f'Trainable network params num : {total_params:,}'
        print(total_params)
        opt_file.write(total_params)

    """ start training """
    start_iter = 0
    first_draw = False
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            first_draw = True
            print(f'continue to train, start_iter: {start_iter}')
            # If resuming, load prior metric logs and trim incomplete entries
            data_train = results_statistics.divide_results(f'{opt.saved_path}/{opt.exp_name}/data_train.txt', column=5)
            data_train = data_train[:(start_iter//opt.valInterval)]
            [traincharLoss, trainbpeLoss, trainwpLoss, trainfusionLoss, trainsumLoss] = data_train.astype(float).T.tolist()
            results_statistics.write_file(f'{opt.saved_path}/{opt.exp_name}/data_train.txt', data_train)

            data_val = results_statistics.divide_results(f'{opt.saved_path}/{opt.exp_name}/data_val.txt', column=5)
            data_val = data_val[:(start_iter//opt.valInterval)]
            [valchar_accuracy, valbpe_accuracy, valwp_accuracy, valfusion_accuracy, valsumLoss] = data_val.astype(float).T.tolist()
            results_statistics.write_file(f'{opt.saved_path}/{opt.exp_name}/data_val.txt', data_val)

            data_test = results_statistics.divide_results(f'{opt.saved_path}/{opt.exp_name}/data_test.txt', column=5)
            data_test = data_test[:(start_iter//opt.valInterval)]
            [testchar_accuracy, testbpe_accuracy, testwp_accuracy, testfusion_accuracy, testsumLoss] = data_test.astype(float).T.tolist()
            results_statistics.write_file(f'{opt.saved_path}/{opt.exp_name}/data_test.txt', data_test)

            data_test_non_osr = results_statistics.divide_results(f'{opt.saved_path}/{opt.exp_name}/data_test_non_osr.txt', column=4)
            data_test_non_osr = data_test_non_osr[:(start_iter//opt.valInterval)]
            [testchar_non_osr_accuracy, testbpe_non_osr_accuracy, testwp_non_osr_accuracy, testfusion_non_osr_accuracy] = data_test_non_osr.astype(float).T.tolist()
            results_statistics.write_file(f'{opt.saved_path}/{opt.exp_name}/data_test_non_osr.txt', data_test_non_osr)
        except:
            pass

    start_time = time.time()
    best_accuracy = -1
    iteration = start_iter

    # print("LR",scheduler.get_last_lr()[0])
        
    while(True):
        # train part
        image_tensors, labels = train_dataset.get_batch()
        image = image_tensors.to(device)

        # Debug: print basic image statistics on the first iteration
        if iteration == start_iter + 1:  # only print once
            print(f"🚑 DEBUG IMAGE STATS:")
            print(f"Shape: {image.shape}")
            print(f"Max Value: {image.max().item()}")
            print(f"Min Value: {image.min().item()}")
            print(f"Mean Value: {image.mean().item()}")
        
        len_target, char_target = converter.char_encode(labels)
        bpe_target = converter.bpe_encode(labels)
        wp_target = converter.wp_encode(labels)
        char_preds, bpe_preds, wp_preds, fusion_preds = model(image)

        char_loss = criterion(char_preds.view(-1, char_preds.shape[-1]), char_target.contiguous().view(-1))
        bpe_pred_cost = criterion(bpe_preds.view(-1, bpe_preds.shape[-1]), bpe_target.contiguous().view(-1))
        wp_pred_cost = criterion(wp_preds.view(-1, wp_preds.shape[-1]), wp_target.contiguous().view(-1))
        fusion_loss = criterion(fusion_preds.view(-1, fusion_preds.shape[-1]), char_target.contiguous().view(-1))
        
        cost = 0.7*fusion_loss+ 0.1*char_loss + 0.1*bpe_pred_cost + 0.1*wp_pred_cost

        model.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        optimizer.step()

        loss_avg.add(cost)

        if utils.is_main_process() and (iteration + 1) % (save_per_epoch*opt.valInterval) == 0:
            # Save a checkpoint dict (model + optimizer + scheduler + iteration) for reliable resuming
            ckpt_to_save = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict() if 'optimizer' in locals() else None,
                'scheduler': scheduler.state_dict() if scheduler is not None else None,
                'iteration': iteration + 1,
                'best_accuracy': best_accuracy,
            }
            torch.save(ckpt_to_save, f'{opt.saved_path}/{opt.exp_name}/iter_{iteration+1}.pth')
            save_count += 1
           
        # validation part
        if utils.is_main_process() and ((iteration + 1) % opt.valInterval == 0 or iteration == 0): # To see training progress, we also conduct validation when 'iteration == 0' 
            if scheduler is not None and (iteration + 1)/opt.valInterval == 10:
                scheduler.step()
                print("LR",scheduler.get_last_lr()[0])
            elapsed_time = time.time() - start_time

            with open(f'{opt.saved_path}/{opt.exp_name}/log_train.txt', 'a') as log:
                model.eval()
                with torch.no_grad():
                    
                    valid_loss, current_accuracys, char_preds, confidence_score, labels, infer_time, length_of_data, accur_numbers, sum_image_paths, sum_char_preds_str, sum_labels, sum_confidence_score_list, sum_logit, pred_result = validation(
                        model, criterion, valid_loader, converter, opt)
                
                    char_accuracy = current_accuracys[0]
                    bpe_accuracy = current_accuracys[1]
                    wp_accuracy = current_accuracys[2]
                    final_accuracy = current_accuracys[3]
                    cur_best = max(char_accuracy, bpe_accuracy, wp_accuracy, final_accuracy)

                    test_loss, test_current_accuracys, test_char_preds, test_confidence_score, test_labels, test_infer_time, test_length_of_data, test_accur_numbers, test_sum_image_paths, test_sum_char_preds_str, test_sum_labels, test_sum_confidence_score_list, test_sum_logit, test_pred_result = validation(
                        model, criterion, test_loader, converter, opt)
                    print('Running test set evaluation...')

                    test_char_accuracy = test_current_accuracys[0]
                    test_bpe_accuracy = test_current_accuracys[1]
                    test_wp_accuracy = test_current_accuracys[2]
                    test_final_accuracy = test_current_accuracys[3]


                valchar_accuracy.append(round(char_accuracy,4))
                valbpe_accuracy.append(round(bpe_accuracy,4))
                valwp_accuracy.append(round(wp_accuracy,4))
                valfusion_accuracy.append(round(final_accuracy,4))
                valsumLoss.append(round(valid_loss.cpu().item(),4))

                testchar_accuracy.append(round(test_char_accuracy,4))
                testbpe_accuracy.append(round(test_bpe_accuracy,4))
                testwp_accuracy.append(round(test_wp_accuracy,4))
                testfusion_accuracy.append(round(test_final_accuracy,4))

                traincharLoss.append(round(char_loss.cpu().item(), 4))
                trainbpeLoss.append(round(bpe_pred_cost.cpu().item(), 4))
                trainwpLoss.append(round(wp_pred_cost.cpu().item(), 4))
                trainfusionLoss.append(round(fusion_loss.cpu().item(),4))
                trainsumLoss.append(round(cost.cpu().item(), 4))

                model.train()

                loss_log = f'[{iteration+1}/{opt.num_iter}], Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}, At: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}'
                loss_avg.reset()
                current_model_log = f'{"char_accuracy":17s}: {char_accuracy:0.3f}, {"bpe_accuracy":17s}: {bpe_accuracy:0.3f}, {"wp_accuracy":17s}: {wp_accuracy:0.3f}, {"fused_accuracy":17s}: {final_accuracy:0.3f}'

                if cur_best >= best_accuracy:
                    best_accuracy = cur_best
                    torch.save(model.state_dict(), f'{opt.saved_path}/{opt.exp_name}/best_accuracy.pth')
                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}'
                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                print(loss_model_log)
                log.write(loss_model_log + '\n')
                # show some predicted results
                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                for gt, pred, confidence in zip(labels[:5], char_preds[:5], confidence_score[:5]):
                    if isinstance(gt, list):
                       gt = ''.join([converter.character[int(ch)] for ch in gt])
                    if opt.Transformer:
                        pred = pred[:pred.find('[s]')]
                    elif 'Attn' in opt.Prediction:
                        gt = gt[:gt.find('[s]')]
                        pred = pred[:pred.find('[s]')]


                    predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}'
                print(predicted_result_log)
                log.write(predicted_result_log + '\n')
                # Write metrics to files
                def write_line(path: str, data: list):
                    with open(path, 'a') as file:
                        line = ''
                        for d in data:
                            line += str(d) + '\t'
                        file.write(line[:-1] + '\n')
                write_line(f'{opt.saved_path}/{opt.exp_name}/data_train.txt', [traincharLoss[-1], trainbpeLoss[-1], trainwpLoss[-1], trainfusionLoss[-1], trainsumLoss[-1]])
                
                write_line(f'{opt.saved_path}/{opt.exp_name}/data_val.txt', [valchar_accuracy[-1], valbpe_accuracy[-1], valwp_accuracy[-1], valfusion_accuracy[-1], round(valid_loss.cpu().item(),4)])
                write_line(f'{opt.saved_path}/{opt.exp_name}/data_test.txt', [testchar_accuracy[-1], testbpe_accuracy[-1], testwp_accuracy[-1], testfusion_accuracy[-1], round(test_loss.cpu().item(),4)])

        if (iteration + 1) == opt.num_iter:
            print('end the training')
            # sys.exit()
            break
        iteration += 1
        
        # Plot metrics every 10 epochs and also on the first resume
        if utils.is_main_process() and ((iteration + 1) % (10*opt.valInterval) == 0 or first_draw):
            # After plotting on first resume, clear the flag
            first_draw = False
                
            save_figure_path = f'{opt.saved_path}/{opt.exp_name}'
            
            # Record per-epoch data; sampling interval set to 1 (was opt.valInterval)
            draw_one_loss(traincharLoss, save_figure_path, 1, 'char')
            draw_one_loss(trainbpeLoss, save_figure_path, 1, 'bpe')
            draw_one_loss(trainwpLoss, save_figure_path, 1, 'wp')
            draw_one_loss(trainsumLoss, save_figure_path, 1, 'sum')
            draw_one_loss(trainfusionLoss,save_figure_path, 1, 'fusion')
            # Record per-epoch data; sampling interval set to 1 (was opt.valInterval)
            # Search for parameter names like 'trainsumLoss' to locate related changes
            draw_mul_loss([traincharLoss, trainbpeLoss, trainwpLoss, trainfusionLoss, trainsumLoss], save_figure_path, 1)

            draw_one_acc(valchar_accuracy, save_figure_path, 1, 'char')
            draw_one_acc(valbpe_accuracy, save_figure_path,1, 'bpe')
            draw_one_acc(valwp_accuracy, save_figure_path, 1, 'wp')
            if len(valfusion_accuracy) > 0:
                draw_one_acc(valfusion_accuracy,save_figure_path,1,'fusion')
            draw_one_loss(valsumLoss, save_figure_path, 1, 'sum')
            
            # Save combined accuracy plots for validation and test sets
            draw_mul_acc([valchar_accuracy, valbpe_accuracy, valwp_accuracy, valfusion_accuracy], save_figure_path, 1)
            draw_mul_acc([testchar_accuracy, testbpe_accuracy, testwp_accuracy, testfusion_accuracy], save_figure_path, 1)
           
            # Record train and validation loss curve
            # trainSumLoss = trainsumLoss[::opt.valInterval] # 原方法
            trainSumLoss = trainsumLoss
            valSumLoss = valsumLoss[:len(trainSumLoss)]
            iter = [i for i in range(len(valSumLoss))]
            fig, ax = plt.subplots()
            ax.plot([i for i in range(len(trainSumLoss))], trainSumLoss, label='train_loss')
            ax.plot([i for i in range(len(valSumLoss))], valSumLoss, label='val_loss')
            ax.set_xlabel('epoch')
            ax.set_ylabel('loss')
            ax.set_title('Loss')
            ax.legend()
            fig.savefig(f'{save_figure_path}/train_and_valid_loss.jpg')

            loss_excel = {'epoch':[], 'trainLoss':[], 'valLoss': [], 'subLoss': []}
            acc_excel = {'epoch':[], 'charAcc':[], 'bpeAcc': [], 'wpAcc': [], 'fusionAcc': []}

            loss_excel['epoch'] = iter
            loss_excel['trainLoss'] = trainSumLoss
            loss_excel['valLoss'] = valSumLoss
            loss_excel['subLoss'] = [trainSumLoss[i] - valSumLoss[i] for i in range(len(valSumLoss))]

            acc_excel['epoch'] = [i for i in range(len(valchar_accuracy))]
            acc_excel['charAcc'] = valchar_accuracy
            acc_excel['bpeAcc'] = valbpe_accuracy
            acc_excel['wpAcc'] = valwp_accuracy
            acc_excel['fusionAcc'] = valfusion_accuracy
            

if __name__ == '__main__':

    opt = get_args()

    if not opt.exp_name:
        opt.exp_name = f'{opt.TransformerModel}' if opt.Transformer else f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'

    opt.exp_name += f'-Seed{opt.manualSeed}'

    os.makedirs(f'{opt.saved_path}/{opt.exp_name}', exist_ok=True)

    """ vocab / character number configuration """
    if opt.sensitive:
        # Explicitly exclude whitespace characters (spaces, newlines, etc.)
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).
    
    utils.init_distributed_mode(opt)

    # print(opt)
    
    """ Seed and GPU setting """
    
    seed = opt.manualSeed + utils.get_rank()
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Common PyTorch performance flags
    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    
    train(opt)

