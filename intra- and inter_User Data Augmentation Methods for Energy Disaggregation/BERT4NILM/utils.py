from tkinter.messagebox import NO
import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def get_user_input(args):
    if torch.cuda.is_available():
        #args.device = 'cuda:' + input('Input GPU ID: ')
        args.device = 'cuda:0' 
    else:
        args.device = 'cpu'

    dataset_code = {'r': 'redd_lf', 'u': 'uk_dale', 'refit': 'REFIT'}
    # args.dataset_code = dataset_code[input(
    #     'Input r for REDD, u for UK_DALE, refit for REFIT: ')]
    args.dataset_code = dataset_code['refit']

    if args.dataset_code == 'redd_lf':
        app_dict = {
            'r': ['refrigerator'],
            'w': ['washer_dryer'],
            'm': ['microwave'],
            'd': ['dishwasher'],
        }
        # args.appliance_names = app_dict[input(
        #     'Input r, w, m or d for target appliance: ')]
        args.appliance_names = app_dict['m']
    elif args.dataset_code == 'uk_dale':
        app_dict = {
            'k': ['kettle'],
            'f': ['fridge'],
            'w': ['washing_machine'],
            'm': ['microwave'],
            'd': ['dishwasher'],
        }
        args.appliance_names = app_dict[input(
            'Input k, f, w, m or d for target appliance: ')]
        
    elif args.dataset_code == 'REFIT':
        app_dict = {
            'k': ['kettle'],
            'w': ['washingmachine'],
            'm': ['microwave'],
            'd': ['dishwasher'],
            'e': ['electromobile']
        }
        app_indicies_dict = {
            'k': {
                'normalize':[2,3,4,5,6,7,8,9,11,12,13],
                #'train': [1], 
                'train': [2,3,4,5,6,7,8,], 
                'validation': [9,11,12,13],
                'test':[2]
                #'test':[17,19,20,21]
            },
            'w': {
                'normalize':[1,2,3,5,6,7,8,9,10,11,13,15,16,17],
                #'train': [1], 
                'train': [1,2,3,5,6,7,8,9,10,11], 
                'validation': [13,15,16,17],
                'test':[2]
                #'test':[18,19,20,21]
            },
            'm': {
                'normalize':[2,3,4,5,6,8,9,10,11,12,15,17],
                #'train': [1], 
                'train': [2,3,4,5,6,8,9,10], 
                'validation': [11,12,15,17],
                # 'test':[2]
                'test':[18,19,20]
            },
            'd': {
                'normalize':[1,2,3,5,6,7,9,10,11,13,15],
                'train': [1,2,3,5,6,7,9], 
                #'train': [1], 
                'validation': [10,11,13,15],
                'test':[2]
                #'test':[16,18,20,21]
                #'train': [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 15],
                #'validate': [22,23,24,25],
            },
            
        }
        #key = input('Input (k)ettle, (w)ashing machine, (m)icrowave, (d)ishwahser or (e)lectromobile for target appliance: ')
        key = args.data_key
        args.appliance_names = app_dict[key]
        args.appliance_indicies = app_indicies_dict[key]
        args.dataset_name=app_dict[key]
    #args.num_epochs = int(input('Input training epochs: '))
    args.num_epochs =0




def set_template(args):
    args.output_size = len(args.appliance_names)
    if args.dataset_code == 'redd_lf':
        args.window_stride = 120
        args.house_indicies = [1, 2, 3, 4, 5, 6]

        args.cutoff = {
            'aggregate': 6000,
            'refrigerator': 400,
            'washer_dryer': 3500,
            'microwave': 1800,
            'dishwasher': 1200,
        }

        args.threshold = {
            'refrigerator': 50,
            'washer_dryer': 20,
            'microwave': 200,
            'dishwasher': 10
        }

        args.min_on = {
            'refrigerator': 10,
            'washer_dryer': 300,
            'microwave': 2,
            'dishwasher': 300
        }

        args.min_off = {
            'refrigerator': 2,
            'washer_dryer': 26,
            'microwave': 5,
            'dishwasher': 300
        }

        args.c0 = {
            'refrigerator': 1e-6,
            'washer_dryer': 0.001,
            'microwave': 1.,
            'dishwasher': 1.
        }

    elif args.dataset_code == 'uk_dale':
        args.window_stride = 240
        args.house_indicies = [1, 2, 3, 4, 5]
        
        args.cutoff = {
            'aggregate': 6000,
            'kettle': 3100,
            'fridge': 300,
            'washing_machine': 2500,
            'microwave': 3000,
            'dishwasher': 2500
        }

        args.threshold = {
            'kettle': 2000,
            'fridge': 50,
            'washing_machine': 20,
            'microwave': 200,
            'dishwasher': 10
        }

        args.min_on = {
            'kettle': 2,
            'fridge': 10,
            'washing_machine': 300,
            'microwave': 2,
            'dishwasher': 300
        }

        args.min_off = {
            'kettle': 0,
            'fridge': 2,
            'washing_machine': 26,
            'microwave': 5,
            'dishwasher': 300
        }

        args.c0 = {
            'kettle': 1.,
            'fridge': 1e-6,
            'washing_machine': 0.01,
            'microwave': 1.,
            'dishwasher': 1.
        }
        
    elif args.dataset_code == 'REFIT':
        args.window_stride = 240
        # args.house_indicies = [1, 2, 3, 4, 5]
        args.sampling = '10s'
        
        args.cutoff = {
            'aggregate': 6000,
            'kettle': 3100,
            'washingmachine': 2500,
            'microwave': 3000,
            'dishwasher': 2500,
            'electromobile': 200
        }

        args.threshold = {
            'kettle': 2000,
            'washingmachine': 20,
            'microwave': 200,
            'dishwasher': 10,
            'electromobile': 10
        }

        args.min_on = {
            'kettle': 2,
            'washingmachine': 300,
            'microwave': 2,
            'dishwasher': 300,
            'electromobile': 10
        }

        args.min_off = {
            'kettle': 0,
            'washingmachine': 26,
            'microwave': 5,
            'dishwasher': 300,
            'electromobile': 0
        }

        args.c0 = {
            'kettle': 1.,
            'washingmachine': 0.01,
            'microwave': 1.,
            'dishwasher': 1.,
            'electromobile': 1.
        }     
        
    args.optimizer = 'adam'
    # args.lr = 1e-4
    args.lr = 1e-5
    args.enable_lr_schedule = False
    args.batch_size =64
    # args.batch_size = 128


def acc_precision_recall_f1_score(pred, status):
    assert pred.shape == status.shape

    pred = pred.reshape(-1, pred.shape[-1])
    status = status.reshape(-1, status.shape[-1])
    accs, precisions, recalls, f1_scores = [], [], [], []

    for i in range(status.shape[-1]):
        tn, fp, fn, tp = confusion_matrix(status[:, i], pred[:, i], labels=[
                                          0, 1]).ravel()
        acc = (tn + tp) / (tn + fp + fn + tp)
        precision = tp / np.max((tp + fp, 1e-9))
        recall = tp / np.max((tp + fn, 1e-9))
        f1_score = 2 * (precision * recall) / \
            np.max((precision + recall, 1e-9))

        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    return np.array(accs), np.array(precisions), np.array(recalls), np.array(f1_scores)


def validate_relative_absolute_error(pred, label):
    assert pred.shape == label.shape
  
    pred = pred.reshape(-1, pred.shape[-1])
    label = label.reshape(-1, label.shape[-1])
    temp = np.full(label.shape, 1e-9)
    relative, absolute, sum_err = [], [], []

    for i in range(label.shape[-1]):
        
        relative_error = np.mean(np.nan_to_num(np.abs(label[:, i] - pred[:, i]) / np.max(
            (label[:, i], pred[:, i], temp[:, i]), axis=0)))
        absolute_error = np.mean(np.abs(label[:, i] - pred[:, i]))

        relative.append(relative_error)
        absolute.append(absolute_error)

    return np.array(relative), np.array(absolute)

def relative_absolute_error(pred, label):
    assert pred.shape == label.shape

    pred = pred.reshape(-1, pred.shape[-1])
    label = label.reshape(-1, label.shape[-1])
    temp = np.full(label.shape, 1e-9)
    relative, absolute, sum_err = [], [], []

    for i in range(label.shape[-1]):
        relative_error = np.mean(np.nan_to_num(np.abs(label[:, i] - pred[:, i]) / np.max(
            (label[:, i], pred[:, i], temp[:, i]), axis=0)))
        absolute_error = np.mean(np.abs(label[:, i] - pred[:, i]))

        relative.append(relative_error)
        absolute.append(absolute_error)

    return np.array(relative), np.array(absolute)


def mean_absolute_error(output, target):

    return np.mean(np.abs(output - target))


def signal_aggregate_error(output, target):

    return np.abs(np.sum(output) - np.sum(target))/np.sum(target)