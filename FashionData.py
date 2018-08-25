import os
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image, ImageOps
import random
random.seed(1024)

default_transform = transforms.Compose([transforms.ToTensor()])

def train_eval_loader(path, resize, transform):
    img = Image.open(path).convert('RGB')
    width, height = img.size
    random_scale = random.uniform(0.7, 0.9)

    scale = resize / float(max(width, height)) * random_scale
    scaled_width = int(width * scale)
    scaled_height = int(height * scale)

    resized_img = img.resize((scaled_width, scaled_height))

    top_pad = random.randint(0, resize-scaled_height)
    down_pad = resize-scaled_height-top_pad
    left_pad = random.randint(0, resize-scaled_width)
    right_pad = resize-scaled_width-left_pad

    padded_image = ImageOps.expand(resized_img, (left_pad, top_pad, right_pad, down_pad), fill='white')

    resize_info = {
        'scale':scale,
        'width':width,
        'height':height,
        'left_pad':left_pad,
        'top_pad':top_pad
    }
    return transform(padded_image), resize_info

def test_loader(path, resize, transform):
    img = Image.open(path).convert('RGB')
    width, height = img.size

    scale = resize / float(max(width, height))
    scaled_width = int(width * scale)
    scaled_height = int(height * scale)

    resized_img = img.resize((scaled_width, scaled_height))

    top_pad =  int((resize-scaled_height)/2)
    down_pad = resize-scaled_height-top_pad
    left_pad = int((resize-scaled_width)/2)
    right_pad = resize-scaled_width-left_pad

    padded_image = ImageOps.expand(resized_img, (left_pad, top_pad, right_pad, down_pad), fill='white')

    resize_info = {
        'scale':scale,
        'width':width,
        'height':height,
        'left_pad':left_pad,
        'top_pad':top_pad,
        'resize':resize
    }
    return transform(padded_image), resize_info


def read_csv(csv_path):
    with open(csv_path) as f:
        line = f.readline()
        item_names = [item.strip() for item in line.split(',')]
        csv_content = {}
        for line in f:
            items = line.split(',')
            image_id = items[0]

            points = []
            for item in line.split(','):
                if '_' not in item:
                    pass
                else:
                    location = [int(i.strip()) for i in item.split('_')]
                    points.append(location)
            csv_content[image_id] = {key: value for key, value in zip(item_names[2:], points)}
    return csv_content

class_points = {
    'blouse':['neckline_left', 'neckline_right','shoulder_left','shoulder_right','center_front',
             'armpit_left', 'armpit_right', 'top_hem_left', 'top_hem_right',
             'cuff_left_in', 'cuff_left_out', 'cuff_right_in', 'cuff_right_out'],

    'dress': ['neckline_left', 'neckline_right', 'shoulder_left','shoulder_right', 'center_front',
             'armpit_left', 'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in',
             'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'hemline_left', 'hemline_right'],

    'outwear': ['neckline_left', 'neckline_right', 'shoulder_left', 'shoulder_right', 'armpit_left',
               'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in', 'cuff_left_out',
               'cuff_right_in', 'cuff_right_out', 'top_hem_left', 'top_hem_right'],

    'skirt': ['waistband_left', 'waistband_right', 'hemline_left', 'hemline_right'],

    'trousers': ['waistband_left', 'waistband_right', 'crotch',
                'bottom_left_in', 'bottom_left_out', 'bottom_right_in', 'bottom_right_out'],
}
# points for distance normalization
# the element in the least record the point index in class_points
metric_points_location = {
    'blouse':[5,6],
    'blouse_backup':[2,3],
    'dress':[5,6],
    'dress_backup':[0,1],
    'outwear':[4,5],
    'outwear_backup':[2,3],
    'skirt':[0,1],
    'skirt_backup':[2,3],
    'trousers':[0,1],
    'trousers_backup':[3,4]
}

train_test_split = {
    'outwear':[(0, 7000), (7000, 7734)],
    'skirt':[(0, 9200), (9200, 9910)],
    'blouse':[(0, 9400), (9400, 10155)],
    'dress':[(0, 6700), (6700, 7224)],
    'trousers':[(0, 8700), (8700, 9142)]
}

mean_std = {
    'outwear':{'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]},
    'skirt':{'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]},
    'blouse':{'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]},
    'dress':{'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]},
    'trousers':{'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]},
}

fakes = []

class fashionSet(data.Dataset):
    def __init__(self,  selected, classname='blouse', root='/home/wuxiaodong/fa/train/Images/',train='train', transform=default_transform, resize=224):
        self.image_dir = os.path.join(root, classname)
        self.selected = selected # a tuple indicating start and end pic index (1000, 2000)
        self.transform = transform
        self.resize = resize
        self.paths = []
        self.labels = []
        self.train = train
        self.classname = classname
        self.numpoints = len(class_points[self.classname])

        img_names = os.listdir(self.image_dir)
        img_names.sort()
        if selected[1]>len(img_names):
            raise IndexError
        self.selected_imgs = list(img_names[selected[0]:selected[1]])

        self.csv_path = os.path.join(root, '../Annotations/train.csv')
        self.csv_content = read_csv(self.csv_path)
        for img_name in self.selected_imgs:
            image_id = 'Images/'+classname+'/'+img_name
            label = []
            points = [self.csv_content[image_id][point_name] for point_name in class_points[self.classname]]
            for point in points:
                label += point[:2]
            for point in points:
                label += [point[2]+1]  # change label form {-1, 0, 1} to {0, 1, 2}
            self.labels.append(label)

        self.length = len(self.selected_imgs)

    def __getitem__(self, index):
        if self.train is 'train':
            img, resize_info = train_eval_loader(os.path.join(self.image_dir, self.selected_imgs[index]), self.resize, self.transform)
            label = list(self.labels[index])
            for i in range(self.numpoints):
                x = label[2*i]
                y = label[2*i+1]
                new_x = (x*resize_info['scale']+resize_info['left_pad'])/self.resize - 0.5
                new_y = (y*resize_info['scale']+resize_info['top_pad'])/self.resize - 0.5

                label[2*i] = new_x
                label[2*i+1] = new_y

            return img, torch.Tensor(label)
        elif self.train is 'val':
            img, resize_info = test_loader(os.path.join(self.image_dir, self.selected_imgs[index]), self.resize, self.transform)
            label = list(self.labels[index])
            for i in range(self.numpoints):
                x = label[2*i]
                y = label[2*i+1]
                new_x = (x*resize_info['scale']+resize_info['left_pad'])/self.resize - 0.5
                new_y = (y*resize_info['scale']+resize_info['top_pad'])/self.resize - 0.5

                label[2*i] = new_x
                label[2*i+1] = new_y
            # if label[2*5] == label[2*6] and label[2*5+1] == label[2*6+1]:
            #     print(os.path.join(self.image_dir, self.selected_imgs[index]))
            #     fakes.append(os.path.join(self.image_dir, self.selected_imgs[index]))
            return img, torch.Tensor(label)

        elif self.train is 'test':
            img, resize_info = test_loader(os.path.join(self.image_dir, self.selected_imgs[index]), self.resize, self.transform)
            return img, torch.Tensor([-2]*(self.numpoints*3))
        else:
            raise KeyError

    def __len__(self):
        return self.length

    def getName(self):
        pass


if __name__ == '__main__':
    from torch.autograd import Variable
    import torch.nn as nn
    torch.cuda.set_device(2)
    train_transform = transforms.Compose(
        [  transforms.ToTensor(),
         ])
         # transforms.Normalize(mean, std)])


    fa = fashionSet(selected=(0,7158), train='val',transform=train_transform)

    train_loader = torch.utils.data.DataLoader(fa, batch_size=40, shuffle=False)
    for batch_idx, (data, label) in enumerate(train_loader):
        pass
    # data, label = next(iter(train_loader))
    # data = Variable(data)
    # label = Variable(label)
    # criterion = nn.CrossEntropyLoss()
    # loss = criterion(Variable(torch.FloatTensor([[0.8, 0.1, 0.1]])), Variable(torch.LongTensor([2])))
    import os
    import shutil
    for file in fakes:
        shutil.copy(file.replace('/Images/', '/doted_Images/'), file.replace('/train/Images/blouse/', '/fake_images/'))
