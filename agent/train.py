import torch
from models import EfficientDet
from loss import FocalLoss
from datasets import get_data


def train(args):
    # Model
    model = EfficientDet()
    # model.initialize()
    # model.cuda()

    # Loss, Optimizer, LR scheduler
    criterion = FocalLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    # Load Dataset
    train_loader, test_loader = get_data(args.dataset, args.data_path, args.batch_size)

    # Start Training
    for epoch in range(args.num_epochs):
        print('\n')
        print('Starting epoch {} / {}'.format(epoch, args.num_epochs))

        # Training.
        model.train()
        total_loss = 0.0
        total_batch = 0

        for idx, sample in enumerate(train_loader):
            x = sample['img'].to('cuda')
            labels = sample['annot'].to('cuda')

            features, regression, classification, anchors = model(x)
            cls_loss, reg_loss = criterion(classification, regression, anchors, labels)
            cls_loss = cls_loss.mean()
            reg_loss = reg_loss.mean()
            loss = cls_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 현재 loss 출력
            if idx % print_freq == 0:
                print('Epoch [%d/%d], Iter [%d/%d], LR: %.6f, Loss: %.4f, Average Loss: %.4f'
                      % (epoch, num_epochs, idx, len(train_loader), lr, loss_this_iter, total_loss / float(total_batch)))

