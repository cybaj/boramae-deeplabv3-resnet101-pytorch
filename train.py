import torch
import os

import argparse
import datetime
import yaml

from tqdm import tqdm

from model import get_unet5, get_unet, deeplabv3plus_resnet101
from angio4 import ANGIODataset
from trainer import Trainer
from preprocessing import preprocess_resize, preprocess_PIL_not_crop

from torchsummary import summary

# target image size
_HEIGHT = 512
_WIDTH = 512

# scale factor
_MIN_SCALE = 1.0
_MAX_SCALE = 2.0

def get_parameters(model, bias=None):
    for name, param in model.named_parameters():
        yield param

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def main():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-g', '--gpu', type=int, required=True, help='gpu id')
    parser.add_argument('--resume', help='checkpoint path')

    parser.add_argument(
        '--max-iteration', type=int, default=100000, help='max iteration'
    )
    parser.add_argument(
        '--lr', type=float, default=1.0e-10, help='learning rate',
    )
    parser.add_argument(
        '--weight-decay', type=float, default=0.0005, help='weight decay',
    )
    parser.add_argument(
        '--momentum', type=float, default=0.99, help='momentum',
    )
    parser.add_argument(
            '--trainset-index', type=int, default=0, help='trainsubset index : [88, 44, 22, 11]',
    )
    parser.add_argument(
        '--infer_dir', type=str, default='../inference/test', help='inference output directory',
    )
    parser.add_argument(
        '--infer_target_dir', type=str, default='../inference/targets', help='inference output directory',
    )
    parser.add_argument(
        '--infer_arch', type=str, default='FCN', help='arch name',
    )
    parser.add_argument(
        '--infer_subset', type=str, default='noname', help='inference output directory',
    )
    args = parser.parse_args()

    args.model = 'deeplabv3plus'
    
    # PATH
    now = datetime.datetime.now()
    here = os.path.dirname(os.path.abspath(__file__))

    args.out = os.path.join(here, 'logs', now.strftime('%Y%m%d_%H%M%S.%f'))

    os.makedirs(args.out)
    with open(os.path.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    # GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()

    # SEED
    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)
    
    # DATASET LOADER
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    
    # train_angio_dataset = ANGIODataset('train', transform=preprocess_PIL_not_crop(_HEIGHT, _WIDTH, _MIN_SCALE, _MAX_SCALE), logdir=args.out)
    train_angio_dataset = ANGIODataset('train', index=args.trainset_index, logdir=args.out)
    valid_angio_dataset = ANGIODataset('valid', logdir=args.out)
    infer_angio_dataset = ANGIODataset('infer', logdir=args.out, infer_target_dir=args.infer_target_dir)

    train_loader = torch.utils.data.DataLoader(train_angio_dataset,
                                                batch_size=3, shuffle=True, collate_fn=collate_fn, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_angio_dataset,
                                                batch_size=1, shuffle=True, **kwargs)
    infer_loader = torch.utils.data.DataLoader(infer_angio_dataset,
                                                batch_size=1, **kwargs)


    # RESUME
    model = deeplabv3plus_resnet101(num_classes=2, output_stride=8, pretrained_backbone=True)
    # summary(model, (3, 128, 128))

    start_epoch = 0
    start_iteration = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        # vgg16 = torchfcn.models.VGG16(pretrained=True)
        # model.copy_params_from_vgg16(vgg16)
        pass
    if cuda:
        model = model.cuda()

    
    # OPTIMIZER
    optim = torch.optim.SGD(
    [
        {'params': get_parameters(model),
         'lr': args.lr * 2, 'weight_decay': 0},
    ],
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay)
    
    # TRAIN
    trainer = Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=valid_loader, 
        infer_loader=infer_loader,
        out=args.out,
        max_iter=args.max_iteration,
        interval_validate=4000,
        resume=args.resume,
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()
    
if __name__ == "__main__":
    main()    
