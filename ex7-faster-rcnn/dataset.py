from torchvision.datasets.voc import VOCDetection
import torch

class MyVOCDataSet(VOCDetection):
    def __init__(self, year, root, image_set, download, transform):
        super().__init__(root, year, image_set, download, transform)
        self.categories = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]

    def __getitem__(self, index):
        image, targets = super().__getitem__(index)
        old_h = int(targets['annotation']['size']['height'])
        old_w = int(targets['annotation']['size']['width'])
        _, new_h, new_w = image.shape

        targets = targets['annotation']['object']
        labels = []
        bboxes = []
        output = {}
        for target in targets:
            label = target['name']
            labels.append(self.categories.index(label))
            bbox = target['bndbox']
            x_min = int(float(bbox['xmin'])/old_w * new_w)
            y_min = int(float(bbox['ymin'])/old_h * new_h)
            x_max = int(float(bbox['xmax'])/old_w * new_w)
            y_max = int(float(bbox['ymax'])/old_h * new_h)
            bbox = [x_min, y_min, x_max, y_max]
            bboxes.append(bbox)
        output['boxes'] = torch.FloatTensor(bboxes)
        output['labels'] = torch.LongTensor(labels)
        return image, output

    def __len__(self):
        return super().__len__()