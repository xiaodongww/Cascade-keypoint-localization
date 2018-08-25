import os
import time
import torch
torch.cuda.set_device(3)
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable

import kres152
import FashionData

LR = 1e-2
loc_vis_weight = [1,0]
vis_balance_weight = [0,0,0]
clothes_type = 'trousers'  
postfix = '{}_e2estage2_all'.format(clothes_type)
# postfix = '{}_onestage_2_'.format(clothes_type)
num_keypoints = len(FashionData.class_points[clothes_type])

train_sel = FashionData.train_test_split[clothes_type][0]
val_sel = FashionData.train_test_split[clothes_type][1]

normalize = transforms.Normalize(mean=FashionData.mean_std[clothes_type]['mean'],
                                 std=FashionData.mean_std[clothes_type]['std'])
transform = transforms.Compose(
    [transforms.ToTensor(), normalize])

trainSet = FashionData.fashionSet(selected=train_sel, classname=clothes_type,
                                  root='/home/wuxiaodong/fa/train/Images/',
                                  train='train', transform=transform)
valSet = FashionData.fashionSet(selected=val_sel, classname=clothes_type,
                                 root='/home/wuxiaodong/fa/train/Images/',
                                 train='val', transform=transform)

trainloader = torch.utils.data.DataLoader(trainSet, batch_size=20, shuffle=True,num_workers=2)
valloader = torch.utils.data.DataLoader(valSet, batch_size=20, shuffle=False,num_workers=2)


state = {}
state['loss_window'] = [0] * 200
state['total_batch_num'] = 0
state['epoch'] = 0
state['loss_avg'] = 0.0
state['ne'] = 0.0
state['test_loss'] = 0.0

net1_path = "/home/wuxiaodong/fa/models/trousers/snapshots/trousers_e2estage2_net1_epoch60.pth"
net2_path = "/home/wuxiaodong/fa/models/trousers/snapshots/trousers_e2estage2_net2_epoch60.pth"

net1 = kres152.get_kres152(state_dict=torch.load(net1_path, map_location=lambda storage, loc: storage),num_keypoints=num_keypoints)
net2 = kres152.get_kres152(state_dict=torch.load(net2_path, map_location=lambda storage, loc: storage),num_keypoints=num_keypoints)
net1.cuda()
net2.cuda()

def train2():
    net1.train()
    net2.train()
    optimizer = torch.optim.SGD([{'params': net2.parameters(), 'lr': LR}, 
                                {'params': net1.parameters(), 'lr': LR},
                                ], 
                                 momentum=0.9, weight_decay=0.0005,nesterov=True)
    for batch_idx, (data, label)in enumerate(trainloader):
        state['total_batch_num'] += 1
        data = Variable(data.cuda())
        label = Variable(label.cuda())
        output1 = net1(data)[:, :2*num_keypoints]
        output2 = net2(data)[:, :2*num_keypoints]
        output = output1 + output2

        optimizer.zero_grad()
        loc_pre = output[:, :2*num_keypoints]   # shape (batch_size, 2*num_keypoints)
        loc_target = label[:, :2*num_keypoints] # shape (batch_size, 2*num_keypoints)
        vis_target  = label[:, 2*num_keypoints:]    # shape (batch_size, num_keypoints)
        loss = criterion(loc_pre=loc_pre, loc_target=loc_target, vis_target=vis_target)
        loss.backward()
        optimizer.step()


        state['loss_window'][state['total_batch_num'] % 20] = float(loss.data[0])
        if state['total_batch_num']>=20:
            loss_avg = sum(state['loss_window']) / float(20)
        else:
            loss_avg = sum(state['loss_window']) / float(state['total_batch_num'])
        state['loss_avg'] = loss_avg
        display = 20
        if (batch_idx + 1) % display == 0:
            toprint = '{}, LR: {}, epoch: {}, batch id: {}, avg_loss: {}, test_loss: {}, ne: {}'.format(
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), \
                LR, state['epoch'], batch_idx, round(loss_avg, 5), round(state['test_loss'], 5),round(state['ne'], 5))
            print(toprint)
            with open(os.path.join(os.path.dirname(__file__), postfix+'_log.txt'), 'a') as f:
                f.write(toprint + '\n')


def test2():
    num = 0
    norm_deviation_sum = 0
    distm = nn.PairwiseDistance(p=2)
    net1.eval()
    net2.eval()
    for batch_idx, (data, label) in enumerate(valloader):
        data = Variable(data.cuda(), volatile=True)
        output1 = net1(data)[:, :2*num_keypoints]
        output2 = net2(data)[:, :2*num_keypoints]
        output = output1 + output2
        output = output.cpu().data
        batch_size = label.size()[0]
        point1, point2 = FashionData.metric_points_location[clothes_type]

        norm_dis_vec = distm(label[:, (2*point1, 2*point1+1)], label[:, (2*point2, 2*point2+1)])

        for m in range(batch_size):
            if norm_dis_vec[m][0] < 1e-4:
                print('using back up nomalize distance')
                back_point1, back_point2 = FashionData.metric_points_location[clothes_type+'_backup']
                norm_dis_vec[m] = distm(label[:, (2*back_point1, 2*back_point1+1)], label[:, (2*back_point2, 2*back_point2+1)])[m]

        for i in range(num_keypoints):
            deviation = distm(label[:, (i*2, i*2+1)], output[:, (i*2, i*2+1)])
            norm_deviation = deviation / norm_dis_vec
            for j in range(batch_size):
                # label is {0, 1, 2} and 0 means point not exist
                if label[j][2*num_keypoints+i] != 2:
                    pass
                else:
                    num += 1
                    norm_deviation_sum += norm_deviation[j][0]
                    if norm_deviation[j][0]>100:
                        print('{}th pic\'s {}th point\'s norm_deviation is {}'.format(batch_idx*batch_idx+j), i, norm_deviation[j][0])
        del data, label, output, norm_dis_vec, deviation
        
    ne = norm_deviation_sum/num
    state['ne'] = ne
    print('ne={}'.format(ne))


def test_loss():
    loss_sum = 0

    net1.eval()
    net2.eval()
    for batch_idx, (data, label) in enumerate(valloader):
        data = Variable(data.cuda(), volatile=True)
        label = Variable(label.cuda())
        output1 = net1(data)[:, :2*num_keypoints]
        output2 = net2(data)[:, :2*num_keypoints]
        output = output1 + output2

        loc_pre = output[:, :2*num_keypoints]   # shape (batch_size, 2*num_keypoints)
        loc_target = label[:, :2*num_keypoints] # shape (batch_size, 2*num_keypoints)
        vis_target  = label[:, 2*num_keypoints:]    # shape (batch_size, num_keypoints)
        loss = criterion(loc_pre=loc_pre, loc_target=loc_target, vis_target=vis_target)
        loss_sum += loss.data[0]
        del loss, output, label
    state['test_loss'] = loss_sum/batch_idx
    print('test loss = {}'.format(state['test_loss']))





if __name__ == '__main__':
    for epoch in range(81):
        if epoch == 20:
            LR = LR * 0.1
        if epoch == 50:
            LR = LR * 0.1
        if epoch == 150:
            LR = LR * 0.1
    
        criterion = kres152.NormKEYPointLoss(num_keypoints=num_keypoints, 
            stan_point=FashionData.metric_points_location[clothes_type], 
            loc_vis_weight=loc_vis_weight, 
            vis_balance_weight=vis_balance_weight).cuda()
        state['epoch'] = epoch
        train2()
        test2()
        test_loss()
        if epoch %10 == 0 :
            torch.save(net1.state_dict(), os.path.join(os.path.dirname(__file__),'snapshots' ,postfix+'_net1_epoch{}.pth'.format(epoch)))
            torch.save(net2.state_dict(), os.path.join(os.path.dirname(__file__),'snapshots' ,postfix+'_net2_epoch{}.pth'.format(epoch)))

