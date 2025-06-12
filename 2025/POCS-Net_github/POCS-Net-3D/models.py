import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class TransformModule(nn.Module):
    def __init__(self):
        super(TransformModule, self).__init__()
        self.a = nn.Parameter(torch.Tensor([0.1]))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3, 3)))

    def forward(self, x_input):
        x = F.conv3d(x_input, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv3d(x, self.conv2_forward, padding=1)

        soft_thr = torch.pow(self.a,2)
        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - soft_thr))

        x = F.conv3d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv3d(x, self.conv2_backward, padding=1)

        x_pred = x_backward.view(-1,32768)

        x = F.conv3d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_est = F.conv3d(x, self.conv2_backward, padding=1)
        symloss = x_est - x_input

        return [x_pred, symloss]

class POCSNet(nn.Module):
    def __init__(self, LayerNo):
        super(POCSNet, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo ## the number of phases

        for i in range(LayerNo):
            onelayer.append(TransformModule())

        self.fcs = nn.ModuleList(onelayer)

    def forward(self,Phi,b,x0):
        x = x0
        layers_sym = []   # for computing symmetric loss
        for i in range(self.LayerNo):
            #Transform Module
            [x, layer_sym] = self.fcs[i](x)
            x = x.view(-1,1,32,32,32)
            layers_sym.append(layer_sym)
            #Correction Module
            x = torch.mul(1-Phi, x) + b
        x_final = x.view(-1,32768)

        return [x_final, layers_sym]

class TransformModule_prediction(nn.Module):
    def __init__(self,len_data):
        super(TransformModule_prediction, self).__init__()
        self.a = nn.Parameter(torch.Tensor([0.1]))
        self.len_data = len_data
        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3, 3)))

    def forward(self, x_input):
        x = F.conv3d(x_input, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv3d(x, self.conv2_forward, padding=1)

        soft_thr = torch.pow(self.a,2)
        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - soft_thr))

        x = F.conv3d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv3d(x, self.conv2_backward, padding=1)

        x_pred = x_backward.view(-1,self.len_data)

        return x_pred

class POCSNet_prediction(nn.Module):
    def __init__(self, LayerNo,m,n,p,len_data):
        super(POCSNet_prediction, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo ## the number of phases
        self.m = m
        self.n = n
        self.p = p
        self.len_data = len_data
        for i in range(LayerNo):
            onelayer.append(TransformModule_prediction(self.len_data))

        self.fcs = nn.ModuleList(onelayer)

    def forward(self,Phi,b,x0):
        x = x0
        for i in range(self.LayerNo):
            print("Iteration%d..."%i)
            #Transform Module
            x = self.fcs[i](x)
            x = x.view(-1,1,self.m,self.n,self.p)
            #Correction Module
            x = torch.mul(1-Phi, x) + b
        x_final = x.view(-1,self.len_data)

        return x_final