# main.py
import argparse
from trainer import Trainer

def get_args():
    parser = argparse.ArgumentParser(description='Executer')
    parser.add_argument('--name', type=str, default='base', help='Model selection', required=False)
    parser.add_argument('--mode', type=str, choices=['rand', 'static', 'non-static', 'multichannel'], default='rand')
    parser.add_argument('--ck_path', type=str, default=r'C:\Users\Simon\ongoing_projects\torch_study\1_CNN\3_packaging\checkpoint\\')  # ?
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--path', type=str, default=r'C:\Users\Simon\ongoing_projects\torch_study\1_CNN\1_reference_code')
    parser.add_argument('--embedding_dim', type=int, default=300)
    parser.add_argument('--n_filters', type=int, default=100)
    parser.add_argument('--filter_sizes', type=list, default=[3,4,5])
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--cv_num', type=int, default=10)
    parser.add_argument('--l2_constraint', type=int, default=3)
    parser.add_argument('--cv', action='store_true')  # 해당인자에 True가 저장
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = get_args()
    trainer = Trainer(args)
    trainer.train()

    