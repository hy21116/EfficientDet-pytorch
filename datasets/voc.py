import os
import xmltodict  # xmltodict 설치 필요
import os.path as pth
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class VOCdataset(Dataset):
    def __init__(self, dataset_dir='./datasets', mode='train', resize=448, transforms=None, feature_size=7,
                 num_bboxes=2, num_classes=20):
        self.dataset_dir = dataset_dir
        self.mode = mode
        self.resize_factor = resize
        self.transform = transforms
        self.feature_size = feature_size
        self.num_bboxes = num_bboxes
        self.num_classes = num_classes
        self.data_list = self.get_infos(mode=self.mode)
        self.image_f = './{}/VOCdevkit/VOC2007/JPEGImages/{}'

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        record = self.data_list[idx]
        img_id = record['image_id']
        bboxs = record['bboxs']
        labels = record['labels']

        img = Image.open(pth.join(self.dataset_dir, self.image_f.format(self.mode, img_id))).convert('RGB')
        img = img.resize((self.resize_factor, self.resize_factor))
        img = np.array(img)

        if self.transform:
            image = self.transform(img)

        if self.mode == 'train':
            target = self.encode(bboxs, labels)
            return image, target
        else:
            return image

    def get_infos(self, mode='train'):

        annot_f = '{}/VOCdevkit/VOC2007/Annotations'

        classes = ['person', 'bird', 'cat', 'cow', 'dog', 'horse',
                   'sheep', 'aeroplane', 'bicycle', 'boat', 'bus', 'car',
                   'motorbike', 'train', 'bottle', 'chair', 'dining table',
                   'potted plant', 'sofa', 'tv/monitor']

        annot_dir = pth.join(self.dataset_dir, annot_f.format(mode))
        result = []
        for ano in [pth.join(annot_dir, ano) for ano in os.listdir(annot_dir)]:
            f = open(ano)  # xml 파일 하나씩 읽어들임
            info = xmltodict.parse(f.read())['annotation']
            image_id = info['filename']
            image_size = np.asarray(tuple(map(int, info['size'].values()))[:2], np.int16)
            w, h = image_size
            box_objects = info['object']
            labels = []
            bboxs = []
            for obj in box_objects:
                try:
                    labels.append(classes.index(obj['name'].lower()))  # 0~19 사이
                    bboxs.append(tuple(map(int, obj['bndbox'].values())))
                except:
                    pass
            bboxs = np.asarray(bboxs, dtype=np.float64)
            try:
                bboxs[:, [0, 2]] /= w
                bboxs[:, [1, 3]] /= h
            except:
                pass
            if bboxs.shape[0] or mode == 'test':
                result.append({'image_id': image_id, 'image_size': image_size, 'bboxs': bboxs, 'labels': labels})
        return result

    def encode(self, bboxs, labels):
        S = self.feature_size
        B = self.num_bboxes
        N = 5 * B + self.num_classes
        cell_size = 1.0 / float(S)

        box_xy = (bboxs[:, 2:] + bboxs[:, :2]) / 2.0
        box_wh = (bboxs[:, 2:] - bboxs[:, :2])
        target = np.zeros((S, S, N))
        for b in range(bboxs.shape[0]):  # ground truth 박스 수만큼 반복
            xy, wh, label = box_xy[b], box_wh[b], labels[b]
            ij = np.ceil(xy / cell_size) - 1.0  # ceil: 소수점있으면 무조건 올림 4.1 -> 5
            i, j = map(int, ij)  # i,j: cell 번호 (0~6)
            top_left = ij * cell_size  # 각 cell의 좌상단 좌표
            xy_norm = (xy - top_left) / cell_size

            for k in range(B):  # 한 cell당 두개의 박스
                a = 5 * k
                target[j, i, a:a + 2] = xy_norm
                target[j, i, a + 2:a + 4] = wh
                target[j, i, a + 4] = 1.0
            target[j, i, 5 * B + label] = 1.0
        return target
