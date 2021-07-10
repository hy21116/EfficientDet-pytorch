import torch
from datasets import get_data

def train():
    train_loader, test_loader = get_data()

    # model
    model = EfficientDet()
    model.initialize()
    model.cuda()

    # Loss, Optimizer, LR scheduler
    criterion = FocalLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nestrov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    # Load Dataset
    train_loader, test_loader = getData()

    for epoch in range(num_epochs):
        print('\n')
        print('Starting epoch {} / {}'.format(epoch, num_epochs))

        # Training.
        yolo.train()
        total_loss = 0.0
        total_batch = 0

        for idx, (x, labels) in enumerate(train_loader):
            x = x.to('cuda')
            labels = labels.to('cuda')

            preds = yolo(x)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 현재 loss 출력
            if idx % print_freq == 0:
                print('Epoch [%d/%d], Iter [%d/%d], LR: %.6f, Loss: %.4f, Average Loss: %.4f'
                      % (epoch, num_epochs, idx, len(train_loader), lr, loss_this_iter, total_loss / float(total_batch)))

