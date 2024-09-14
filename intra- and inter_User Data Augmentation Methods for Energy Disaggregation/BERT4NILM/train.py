from dataset import *
from dataloader import *
from trainer import *
from config import *
from utils import *
from model import BERT4NILM
import logging
import argparse
import torch


def train(args, export_root=None, resume=True):
    # args.validation_size = 0
    # if args.dataset_code == 'redd_lf':
    #     args.house_indicies = [2, 3, 4, 5, 6]
    #     dataset = REDD_LF_Dataset(args)
    # elif args.dataset_code == 'uk_dale':
    #     args.house_indicies = [1, 3, 4, 5]
    #     dataset = UK_DALE_Dataset(args)
    # elif args.dataset_code == 'REFIT':
        
    #     args.house_indicies = args.appliance_indicies['normalize']
    #     dataset = REFIT_Dataset(args)
    # x_mean, x_std = dataset.get_mean_std()
    # stats = (x_mean, x_std)
    # logging.info('x_mean {}, x_std {}'.format( x_mean, x_std))

    #stats = (478.9757834992137,  807.3386296352045) # washingmachine
    stats = (464.38038583754525, 787.8320975714753) # microwave
    #stats = (472.83127363711964, 804.5241999588686)#kettle
    #stats = (475.6770257193361, 788.2688078230458) #dishwasher
    #logging.info("x_mean:{},x_std:{}".format(x_mean,x_std))
    args.validation_size = 0
    if args.dataset_code == 'redd_lf':
        args.house_indicies = [2, 3, 4, 5, 6]
        dataset = REDD_LF_Dataset(args)
    elif args.dataset_code == 'uk_dale':
        args.house_indicies = [3, 4, 5]
        dataset = UK_DALE_Dataset(args,stats)
    elif args.dataset_code == 'REFIT':
        args.house_indicies = [args.train_house]
        # args.house_indicies = args.appliance_indicies['train']
        dataset = REFIT_Dataset(args,stats)
        print(stats)

    
    model = BERT4NILM(args)

    if export_root == None:
        folder_name = '-'.join(args.appliance_names)
        export_root = 'experiments/' + args.dataset_code + '/' + folder_name

    dataloader = NILMDataloader(args, dataset, bert=True)
    train_loader = dataloader.get_dataloaders()
    args.validation_size = 0
    if args.dataset_code == 'redd_lf':
        args.house_indicies = [2, 3, 4, 5, 6]
        dataset1 = REDD_LF_Dataset(args)
    elif args.dataset_code == 'uk_dale':
        args.house_indicies = [1]
        dataset1 = UK_DALE_Dataset(args)
    elif args.dataset_code == 'REFIT':
        args.house_indicies = [args.test_name]
        # args.house_indicies = args.appliance_indicies['validation']
        dataset1 = REFIT_Dataset(args, stats)
    dataloader1 = NILMDataloader(args, dataset1, bert=False)
    val_loader = dataloader1.get_dataloaders()

    trainer = Trainer(args, model, train_loader,
                    val_loader, stats, export_root)
    if args.num_epochs > 0:
        if resume:
            try:
                model.load_state_dict(torch.load(os.path.join(
                    export_root, 'best_acc_modeldata-raw.pth'), map_location='cpu'))
                # model.load_state_dict(torch.load(os.path.join(
                #     export_root, 'best_acc_model{}.pth'.format(args.data_na)), map_location='cpu'))
                print('Successfully loaded previous model, continue training...')
                print(os.path.join(
                    export_root, 'best_acc_model{}.pth'.format(args.data_na)))
            except FileNotFoundError:
                print('Failed to load old model, continue training new model...')
        for param in model.parameters():
           param.requires_grad = False
        # #model.transformer_blocks[-2].requires_grad_(True)
        model.transformer_blocks[-1].requires_grad_(True)
        model.deconv.requires_grad_(True)
        model.linear1.requires_grad_(True)
        model.linear2.requires_grad_(True)
        
           
        trainer.train()
    else:
        #model.load_state_dict(torch.load(os.path.join(export_root, 'best_acc_modeldata-raw.pth'), map_location='cpu'))
        model.load_state_dict(torch.load(os.path.join(export_root, 'best_acc_model{}_{}_{}.pth'.format(args.epo,args.iter,args.data_na)), map_location='cpu'))
        # model.load_state_dict(torch.load(os.path.join(export_root,  'best_acc_msodel{}.pth'.format(args.data_na)), map_location='cpu'))
        # print(os.path.join(export_root, 'best_acc_model{}_{}_{}fine.pth'.format(args.epo,args.iter,args.data_na)))
    args.validation_size = 0
    if args.dataset_code == 'redd_lf':
        args.house_indicies = [1]
        dataset2 = REDD_LF_Dataset(args, stats)
    elif args.dataset_code == 'uk_dale':
        args.house_indicies = [2]
        dataset2 = UK_DALE_Dataset(args, stats)
    elif args.dataset_code == 'REFIT':
        # args.house_indicies =args.appliance_indicies['test']
        args.house_indicies = [args.test_name]
        dataset2 = REFIT_Dataset(args, stats)

    dataloader2 = NILMDataloader(args, dataset2, bert=False)
    test_loader = dataloader2.get_dataloaders()
    rel_err, abs_err, acc, prec, recall, f1 = trainer.test(test_loader,dataset2)
    
    print('Mean Accuracy:', acc)
    print('Mean F1-Score:', f1)
    print('SAE:', rel_err)
    print('Mean Absolute Error:', abs_err)
    logging.info('Mean Accuracy {:.2f}, Mean F1-Score {:.2f},SAE {:.2f}, Mean Absolute Error {:.2f}'.format(
            * acc, *f1, rel_err, abs_err))

def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    

torch.set_default_tensor_type(torch.DoubleTensor)
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--dataset_code', type=str,
                    default='redd_lf', choices=['redd_lf', 'uk_dale'])
parser.add_argument('--validation_size', type=float, default=0.2)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--house_indicies', type=list, default=[1, 2, 3, 4, 5])
parser.add_argument('--appliance_names', type=list,
                    default=['microwave', 'dishwasher'])
parser.add_argument('--sampling', type=str, default='6s')
parser.add_argument('--cutoff', type=dict, default=None)
parser.add_argument('--threshold', type=dict, default=None)
parser.add_argument('--min_on', type=dict, default=None)
parser.add_argument('--min_off', type=dict, default=None)
parser.add_argument('--window_size', type=int, default=480)
parser.add_argument('--window_stride', type=int, default=120)
parser.add_argument('--normalize', type=str, default='mean',
                    choices=['mean', 'minmax'])
parser.add_argument('--denom', type=int, default=2000)
parser.add_argument('--model_size', type=str, default='gru',
                    choices=['gru', 'lstm', 'dae'])
parser.add_argument('--output_size', type=int, default=1)
parser.add_argument('--drop_out', type=float, default=0.1)
parser.add_argument('--mask_prob', type=float, default=0.25)
parser.add_argument('--device', type=str, default='cuda',
                    choices=['cpu', 'cuda'])
parser.add_argument('--optimizer', type=str,
                    default='adam', choices=['sgd', 'adam', 'adamw'])
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0.)
parser.add_argument('--momentum', type=float, default=None)
parser.add_argument('--decay_step', type=int, default=100)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--c0', type=dict, default=None)
parser.add_argument('--dataset_name', type=str, default='data')
parser.add_argument('--data_na', type=str, default='data')
parser.add_argument('--test_name', type=int, default=2)
parser.add_argument('--data_key', type=str, default='k')
parser.add_argument('--train_house', type=int, default=2)
parser.add_argument('--iter', type=int, default=1)
parser.add_argument('--epo', type=int, default=1)

args = parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(filename='log/training_data-{}.log'.format(args.data_na), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    fix_random_seed_as(args.seed)
    get_user_input(args)
    set_template(args)
    
    train(args)
