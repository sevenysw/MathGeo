# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 10:25:26 2023

@author: 95424
"""

import numpy as np
import torch

#定义偏导函数
def pardiv(A,method):
    if method == 'Dx':
        Dx=torch.diff(A,1,dim=1)
        return Dx
    elif method == 'Dy':
        Dy=torch.diff(A,1,dim=0)
        return Dy
    elif method == 'Dxx':
        Dxx =torch.diff(A,2,dim=1)
        return Dxx
    elif method == 'Dyy':
        Dyy=torch.diff(A,2,dim=0)
        return Dyy
    
 #这里使用的p=[u,u]   
def TGV(u,sigma):
    m,n=u.shape
    u = sigma*u
    Dx = pardiv(u,'Dx')
    Dy = pardiv(u,'Dy')
    p1 = 0.5*Dx
    p2 = 0.5*Dy
    p_dx = pardiv(p1,'Dx')
    p_dy = pardiv(p2,'Dy')
    p_dyx = pardiv(p1,'Dy')
    p_dxy = pardiv(p2,'Dx')
    TGV_1 = abs(Dx-p1).sum()+abs(Dy-p2).sum()
    TGV_2 = abs(p_dx).sum()+abs(1/2*(p_dyx+p_dxy)).sum()+abs(1/2*(p_dyx+p_dxy)).sum()+abs(p_dy).sum()
    return 2*TGV_1+TGV_2


#x=torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]],dtype=torch.float32,requires_grad=True)
#x=torch.randn(5,3,requires_grad=True)
#loss = TGV(x,0.1,0.2)
#loss.backward()