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
    
    x_mean=475.6770257193361
    x_std=788.2688078230458
    
    

    

    
    model = BERT4NILM(args)
    model2 = BERT4NILM(args)

    
    if args.num_epochs > 0:
        if resume:
            try:
                model.load_state_dict(torch.load('experiments/REFIT/dishwasher/best_acc_modeldata-raw.pth'))

                model2.load_state_dict(torch.load('experiments/REFIT/dishwasher/best_acc_modeldata-rawfine_1.pth'))
                
            except FileNotFoundError:
                print('Failed to load old model, continue training new model...')
        # 检查模型是否结构相同  
    assert isinstance(model, type(model2)), "The models should be of the same type"  
    assert len(list(model.parameters())) == len(list(model2.parameters())), "The models should have the same number of parameters"  
  
# 遍历模型的每一层，比较权重和偏置  
    all_weights_equal = True  
    for ((name1, param1), (name2, param2)) in zip(model.named_parameters(), model2.named_parameters()):  
        if name1 != name2:  
            raise ValueError(f"The parameter names do not match: {name1} vs {name2}")
        if torch.equal(param1.data, param2.data):
            print(f"The weights of {name1} are  equal.")   
        if not torch.equal(param1.data, param2.data):  
            print(f"The weights of {name1} are not equal.")  
            all_weights_equal = False  
             
  
    if all_weights_equal:  
        print("All weights of the models are equal.")  
    else:  
        print("Some weights of the models are not equal.")
    
        
           
       
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
parser.add_argument('--test_name', type=int, default=20)
parser.add_argument('--data_key', type=str, default='k')
parser.add_argument('--train_house', type=int, default=20)

args = parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(filename='training_{}.log'.format(args.data_na), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    fix_random_seed_as(args.seed)
    get_user_input(args)
    set_template(args)
    
    train(args)
