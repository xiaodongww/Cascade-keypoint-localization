import os
import time
import torch
torch.cuda.set_device(3)
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable

import kres152
import FashionData

predicts = {}
def test2(clothes_type,net1_path, net2_path):
    global predicts
      
    test_sel = 'all'
    num_keypoints = len(FashionData.class_points[clothes_type])
    normalize = transforms.Normalize(mean=FashionData.mean_std[clothes_type]['mean'],
                                     std=FashionData.mean_std[clothes_type]['std'])
    transform = transforms.Compose(
        [transforms.ToTensor(), normalize])
    testSet = FashionData.fashionSet(selected=test_sel, classname=clothes_type,
                                     root='/home/wuxiaodong/fa/test/Images/',
                                     train='test', transform=transform)
    testloader = torch.utils.data.DataLoader(testSet, batch_size=40, shuffle=False,num_workers=2)
    
    net1 = kres152.get_kres152(state_dict=torch.load(net1_path, map_location=lambda storage, loc: storage),num_keypoints=num_keypoints)
    net2 = kres152.get_kres152(state_dict=torch.load(net2_path, map_location=lambda storage, loc: storage),num_keypoints=num_keypoints)
    net1.cuda()
    net2.cuda()
    net1.eval()
    net2.eval()
    for batch_idx, (data, label) in enumerate(testloader):
        batch_num = label.size()[0]
        batch_size = testloader.batch_size
        data = Variable(data.cuda(), volatile=True)
        output1 = net1(data)[:, :2*num_keypoints]
        output2 = net2(data)[:, :2*num_keypoints]
        output = output1 + output2
        output = output.cpu().data
        for i in range(batch_num):
            image_id = 'Images/{}/{}'.format(clothes_type, testSet.selected_imgs[batch_idx*batch_size+i])
   

            predict = {}
            for j in range(num_keypoints):
                point_name = FashionData.class_points[clothes_type][j]
                pre_new_x = output[i, 2*j]
                pre_new_y = output[i, 2*j+1]
                ori_resize = label[i, 0]
                ori_scale = label[i, 1]
                ori_left_pad = label[i, 2]
                ori_top_pad = label[i, 3]
                ori_x = int(((pre_new_x+0.5)*ori_resize - ori_left_pad)/ori_scale)
                ori_y = int(((pre_new_y+0.5)*ori_resize - ori_top_pad)/ori_scale)
                width = int((ori_resize-ori_left_pad)/ori_scale)
                height = int((ori_resize-ori_top_pad)/ori_scale)
                if ori_x<0 or ori_x>width:
                    ori_x = int(width/2)
                if ori_y<0 or ori_y>height:
                    ori_y = int(height/2)
                predict[point_name] = '{}_{}_{}'.format(ori_x, ori_y, 1)
            predicts[image_id]=predict

# for point_name in FashionData.class_points[clothes_type]
test2(clothes_type = 'blouse',
    net1_path = "/home/wuxiaodong/fa/models/blouse/snapshots/blouse_e2estage2_net1_epoch60.pth",
    net2_path = "/home/wuxiaodong/fa/models/blouse/snapshots/blouse_e2estage2_net2_epoch60.pth")

test2(clothes_type = 'dress',
    net1_path = "/home/wuxiaodong/fa/models/dress/snapshots/dress_e2estage2_net1_epoch60.pth",
    net2_path = "/home/wuxiaodong/fa/models/dress/snapshots/dress_e2estage2_net2_epoch60.pth")

test2(clothes_type = 'outwear',
    net1_path = "/home/wuxiaodong/fa/models/outwear/snapshots/outwear_e2estage2_try2_net1_epoch30.pth",
    net2_path = "/home/wuxiaodong/fa/models/outwear/snapshots/outwear_e2estage2_try2_net2_epoch30.pth")

test2(clothes_type = 'skirt',
    net1_path = "/home/wuxiaodong/fa/models/skirt/snapshots/skirt_e2estage2_net1_epoch60.pth",
    net2_path = "/home/wuxiaodong/fa/models/skirt/snapshots/skirt_e2estage2_net2_epoch60.pth")

test2(clothes_type = 'trousers',
    net1_path = "/home/wuxiaodong/fa/models/trousers/snapshots/trousers_e2estage2_net1_epoch60.pth",
    net2_path = "/home/wuxiaodong/fa/models/trousers/snapshots/trousers_e2estage2_net2_epoch60.pth")





with open("/home/wuxiaodong/fa/train/Annotations/train.csv") as f:
        line = f.readline()
        columns = [item.strip() for item in line.split(',')]
        point_names = columns[2:]

with open('/home/wuxiaodong/fa/test/test.csv', 'r') as f:
    with open('result_rotate.csv', 'w') as out:
        out.write(line)
        line = f.readline()
        for line in f:
            image_id = line.split(',')[0].strip()
            image_category = line.split(',')[1].strip()
            out.write(image_id+','+image_category)
            for point_name in point_names:
                if point_name in predicts[image_id].keys():
                    out.write(','+predicts[image_id][point_name])
                else:
                    out.write(',-1_-1_-1')
            out.write('\n')



