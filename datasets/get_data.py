from voc import VOCdataset
from coco import COCODataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from augmentation import DefaultTrainTransform, DefaultValidTransform
from collate_fn import collate_fn


def get_data(dataset, data_path, batch_size, num_workers=16):
    print('Load {} datasets...'.format(dataset))
    if dataset == 'VOC':
        train_dataset = VOCdataset(data_path, transforms=transforms.ToTensor(), mode='train')
        test_dataset = VOCdataset(data_path, transforms=transforms.ToTensor(), mode='test')

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    elif dataset == 'COCO':
        train_set = COCODataset(data_path, set_name='train2017', transform=DefaultTrainTransform(size=512))
        test_set = COCODataset(data_path, set_name='val2017', transform=DefaultValidTransform(size=512))

        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True,
                                  collate_fn=collate_fn, num_workers=num_workers)
        test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, drop_last=True,
                                 collate_fn=collate_fn, num_workers=num_workers)

    return train_loader, test_loader
