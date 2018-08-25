import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict

class kres152(nn.Module):
    def __init__(self,  num_keypoints):
        super(kres152, self).__init__()
        self.res152 = models.densenet161(pretrained=True)
        self.outputs = nn.Linear(2208, 5*num_keypoints)  # each keypoint, there is 2 for the location and 3 for the visibility

    def forward(self, x):
        features = self.res152.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.outputs(out)
        return out

def get_kres152(state_dict=None, **kwargs):
    net = kres152(**kwargs)
    own_state = net.state_dict()
    if state_dict is not None:
        for name, param in state_dict.items():
            try:
                if name not in own_state:
                    continue
                else:
                    own_state[name].copy_(param)
            except:
                if 'weight' in name:
                    nn.init.xavier_normal(own_state[name])
                else:
                    own_state[name].zero_()
    else:
        for name, param in own_state.items():
            if 'outputs' in name:
                print(name)
                if 'weight' in name:
                    nn.init.xavier_normal(own_state[name])
                else:
                    own_state[name].zero_()

    return net

def mse_loss(pre, target):
    return (pre-target)**2
class NormKEYPointLoss(nn.Module):
    def __init__(self, num_keypoints, stan_point, normalized=False,loc_vis_weight=[1, 1], vis_balance_weight=[0.5, 0.45, 0.05]):
        """
        point1 and point2 is used as the standard points
        """
        super(NormKEYPointLoss, self).__init__()
        self.num_keypoints = num_keypoints
        self.loc_vis_weight = loc_vis_weight
        self.vis_criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(vis_balance_weight))
        # self.msem = nn.MSELoss(reduce=False)
        self.msem = mse_loss
        self.point1, self.point2 = stan_point
        self.distm = nn.PairwiseDistance(p=2)
        self.normalized = normalized

    def forward(self, loc_pre, loc_target, vis_target, vis_pre=None, ):
        
        batch_size = loc_target.size()[0]

        # get the normalized mask to sovle the scale variance
        norm_square_vec = self.distm(
            loc_target.data[:, (2 * self.point1, 2 * self.point1 + 1)], 
            loc_target.data[:, (2 * self.point2, 2 * self.point2 + 1)]
            )**2    # shape (batch_size,1)

        for i in range(batch_size):
            if norm_square_vec[i][0] < 1e-3:
                norm_square_vec[i][0] = 1

        norm_square_vec = 1/norm_square_vec
        norm_square_vec = batch_size/norm_square_vec.sum()*norm_square_vec
        norm_mask = torch.cat([norm_square_vec]*(2*self.num_keypoints), 1)
        norm_mask = Variable(norm_mask)  # shape (batch_size, num_keypoints)

        # get the location mask. when calcualting mse loss, only choose the existing keypoints
        loc_mask = torch.ones(batch_size, 2 * self.num_keypoints).type_as(loc_target.data)
        for i in range(batch_size):
            for j in range(self.num_keypoints):
                # label is {0, 1, 2} and 0 means point not exist
                if vis_target.data[i][j] == 0:
                    loc_mask[i][2 * j] = 0
                    loc_mask[i][2 * j + 1] = 0
        loc_mask = Variable(loc_mask)
        if self.normalized:
            loc_loss = (self.msem(loc_pre, loc_target) * loc_mask * norm_mask).mean()
        else:
            loc_loss = (self.msem(loc_pre, loc_target) * loc_mask).mean()


        if vis_pre is None:
            return self.loc_vis_weight[0] * loc_loss
        else:
            vis_loss = Variable(torch.Tensor([0]).type_as(loc_target.data))
            for i in range(self.num_keypoints):
                visibility = vis_target[:, i].type(torch.cuda.LongTensor)
                vis_loss += self.vis_criterion(vis_pre[:, 3*i:3*i+3], visibility)
        
            with open('loss_tmp4.txt', 'a') as f:
                f.write('{}\t{}\n'.format(self.loc_vis_weight[0] * loc_loss.data[0], self.loc_vis_weight[1] * vis_loss.data[0]))
            
            return self.loc_vis_weight[0] * loc_loss + self.loc_vis_weight[1] * vis_loss


class localCNN(nn.Module):
    def __init__(self):
        super(localCNN, self).__init__()
        self.conv = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 16, kernel_size=6, stride=2, padding=0, bias=True)),
            ('batchnorm1', nn.BatchNorm2d(16)),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)),

            ('conv2', nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0, bias=True)),
            ('batchnorm2', nn.BatchNorm2d(16)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool2', nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)),
        ]))

        self.linear = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(16*3*3, 128)),
            ('batchnorm3', nn.BatchNorm1d(128)),
            ('relu3', nn.ReLU(inplace=True)),

            ('linear2', nn.Linear(128, 136)),
        ]))
    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x.view(x.size(0), -1))
        return x


class localfeaFromImage(nn.Module):
    def __init__(self,  num_keypoints):
        super(localfeaFromImage, self).__init__()
        self.num_keypoints = num_keypoints
        self.dense161 = models.densenet161(pretrained=True)
        self.outputs = nn.Linear(2208, 5*num_keypoints)  # each keypoint, there is 2 for the location and 3 for the visibility
        self.localcnns = nn.ModuleList([localCNN() for i in range(num_keypoints)])
        self.premodels = nn.ModuleList([nn.Linear(2208+136, 2) for i in range(num_keypoints)])

    def forward(self, x, points):
        """
        points: [x1, y1, x2, y2, ...,xn, yn]
        location in resized image = (x+0.5)*resize
        """
        features = self.dense161.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        batch_size = x.size()[0]

        # local feas

        predicts = []
        localfeas = []
        for i in range(self.num_keypoints):
            patches = []
            for j in range(batch_size):
                # print(i,j)
                # if not(((points[j, 2*i] + 0.5) * x.size()[2]) <100000 ):
                #     print((points[j, 2*i]))
                loc_x = int((points[j, 2*i] + 0.5) * x.size()[2])
                loc_y = int((points[j, 2*i+1] + 0.5) * x.size()[2])
                if loc_x-15<0:
                    loc_x_1 = 0
                    loc_x_2 = 31
                elif loc_x + 16 > x.size()[2]:
                    loc_x_1 = x.size()[2]-31
                    loc_x_2 = x.size()[2]
                else:
                    loc_x_1 = loc_x - 15
                    loc_x_2 = loc_x + 16

                if loc_y-15<0:
                    loc_y_1 = 0
                    loc_y_2 = 31
                elif loc_y + 16 > x.size()[2]:
                    loc_y_1 = x.size()[2]-31
                    loc_y_2 = x.size()[2]
                else:
                    loc_y_1 = loc_y - 15
                    loc_y_2 = loc_y + 16

                # print(loc_x_1, loc_x_2,  loc_y_1, loc_y_2)
                patch = x[j, :, loc_x_1:loc_x_2, loc_y_1:loc_y_2]
                # patch = x[j, :, loc_x-15:loc_x+16, loc_y-15:loc_y+16]
                patches.append(patch)
            localfea = self.localcnns[i](torch.stack(patches))
            fea = torch.cat((out, localfea), 1)
            pre = self.premodels[i](fea)
            predicts.append(pre)
        predicts = torch.cat(predicts, 1)

        return predicts




def get_localmodel(state_dict=None, **kwargs):
    net = localfeaFromImage(**kwargs)
    own_state = net.state_dict()
    if state_dict is not None:
        for name, param in state_dict.items():
            try:
                if name not in own_state:
                    continue
                else:
                    own_state[name].copy_(param)
            except:
                if 'weight' in name:
                    nn.init.xavier_normal(own_state[name])
                else:
                    own_state[name].zero_()
    else:
        for name, param in own_state.items():
            if 'outputs' in name:
                print(name)
                if 'weight' in name:
                    nn.init.xavier_normal(own_state[name])
                else:
                    own_state[name].zero_()

    return net


if __name__ == '__main__':
    import FashionData
    clothes_type = 'blouse'
    num_keypoints = len(FashionData.class_points['blouse'])
    net = get_kres152(num_keypoints=num_keypoints)
    lossm = KEYPointLoss(num_keypoints=num_keypoints, weight=[1, 0], vis_weight=[0,0,0])
    lossm = NormKEYPointLoss(num_keypoints=num_keypoints, weight=[1, 0], vis_weight=[0,0,0])

    label = Variabel(torch.rand(10, ))
    loss = lossm(net(FashionData.data), FashionData.label)
