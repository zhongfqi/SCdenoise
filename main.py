#-*- coding: UTF-8 -*-
import argparse
import random
import torch.backends.cudnn as cudnn
from utils import *
from SCdenoise import SCdenoise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--embedding_size', type=int, default=256, help='embedding_size')
    parser.add_argument('--cls_coeff', type=float, default=1.0, help="regularization coefficient for classification loss")
    parser.add_argument('--mse_coeff', type=float, default=1.0,
                        help="regularization coefficient for source mse loss")
    parser.add_argument('--BNM_coeff', type=float, default=0.2, help="regularization coefficient for BNM loss")
    parser.add_argument('--DA_coeff', type=float, default=0.05, help="regularization coefficient for domain alignment loss")
    parser.add_argument('--epoch_th', type=int, default=1000, help='epoch_th')
    parser.add_argument('--num_iterations', type=int, default=1000, help="num_iterations")
    parser.add_argument('--alpha', type=float, default=0.1, metavar='ALPHA',
                        help='regularization coefficient for VAT loss')
    parser.add_argument('--xi', type=float, default=10.0, metavar='XI',
                        help='hyperparameter of VAT')
    parser.add_argument('--eps', type=float, default=1.0, metavar='EPS',
                        help='hyperparameter of VAT')
    parser.add_argument('--ip', type=int, default=1, metavar='IP',
                        help='hyperparameter of VAT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='4', help="device id to run")
    parser.add_argument('--seed', type=int, default=666, help="random seed")
    parser.add_argument('--result_path', type=str, default='./results/')
    parser.add_argument('--dataset_path', type=str, default='./dataset/')
    parser.add_argument('--dataset', type=str, default='simulated_drop1')
    parser.add_argument('--source_name', type=str, default='batch2')
    parser.add_argument('--target_name', type=str, default='batch1')


    args = parser.parse_args()
    print(args)
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.set_num_threads(2)
    cudnn.deterministic = True
    cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # Convert CSV to H5AD
    csv2h5ad(inPath='./dataset/csv/', outPath=args.dataset_path)
    # read data and preprocess
    data = sc.read_h5ad(os.path.join(args.dataset_path, args.dataset+'.h5ad'))
    source_data, target_data = split_input(data, args.source_name, args.target_name)
    _, source_data = pre_proccess(source_data)
    _, target_data = pre_proccess(target_data)
    # SCdenoise
    SCdenoise(args, source_data, target_data)

