from torch.utils.data import DataLoader
from datasets import COCODataset, DefaultTrainTransform, DefaultValidTransform, collate_fn

def train(args):
    # Model
    model = EfficientDet()
    model.initialize()
    model.cuda()

    # Loss, Optimizer, LR scheduler
    criterion = FocalLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nestrov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    # Load Dataset
    train_set = COCODataset(args.data_path, set_name='train2017', transform=DefaultTrainTransform(size=512))
    val_set = COCODataset(args.data_path, set_name='val2017', transform=DefaultValidTransform(size=512))

    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              collate_fn=collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False, drop_last=True,
                            collate_fn=collate_fn, num_workers=args.num_workers)

    # Start Training
    for epoch in range(args.num_epochs):
        print('\n')
        print('Starting epoch {} / {}'.format(epoch, args.num_epochs))

        # Training.
        yolo.train()
        total_loss = 0.0
        total_batch = 0

        for idx, sample in enumerate(train_loader):
            x = sample['img'].to('cuda')
            labels = sample['annot'].to('cuda')

            preds = yolo(x)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 현재 loss 출력
            if idx % print_freq == 0:
                print('Epoch [%d/%d], Iter [%d/%d], LR: %.6f, Loss: %.4f, Average Loss: %.4f'
                      % (epoch, num_epochs, idx, len(train_loader), lr, loss_this_iter, total_loss / float(total_batch)))

