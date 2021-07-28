import argparse
from agent import train

parser = argparse.ArgumentParser(description='Implement EfficientDet Pytorch - JeongwooLee1124, kimkj38, hy21116')
parser.add_argument('--train', action='store_true')
parser.add_argument('--eval', action='store_true')
## Train arguments
parser.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
parser.add_argument('--dataset', type=str, default='VOC', help='VOC or COCO')

parser.add_argument('--num_workers', type=int, default=12, help='num_workers of dataloader')
parser.add_argument('--batch_size', type=int, default=12, help='The number of images per batch among all devices')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--optim', type=str, default='adamw', help='select optimizer for training')

parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
parser.add_argument('--save_interval', type=int, default=500, help='Number of steps between saving')
parser.add_argument('--es_min_delta', type=float, default=0.0,
                    help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
parser.add_argument('--es_patience', type=int, default=0,
                    help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')

parser.add_argument('--data_path', type=str, default='', help='the root folder of dataset')
parser.add_argument('--log_path', type=str, default='logs/')
parser.add_argument('--saved_path', type=str, default='logs/')
parser.add_argument('--load_weights', type=str, default=None,
                    help='whether to load weights from a checkpoint, set None to initialize, set \'last\' to load last checkpoint')

args = parser.parse_args()

if __name__ == '__main__':
    if args.train:
        train(args)

