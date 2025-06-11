import torch

def interpolator_BL_2D(U,xn,xf,N,type,device):
    U = U.transpose(0,1)
    nt = U.shape[0]
    xn = xn.transpose(0,1)
    N1 = xn.shape[0]
    nk = xf.shape[1]
    M = N
    kt = N
    Mf = 1
    if type == 0:
        d_rec = torch.zeros(nt,nk).to(device)
        for k in range(nk):
            nx = N1
            l = xn
            pd = torch.ones(nx-kt,1).to(device)
            for s in range(nx-kt):
                for i in range(kt+1):
                    pd[s,0] = pd[s,0] * torch.abs(xf[0,k] - l[s+i,0])
            sj = torch.argmin(pd)
            p1 = sj
            p2 = sj +kt
            xt = xn[p1:p2+1,0]
            min_ia = p1
            max_ia = p2
            X = xt.repeat(M+1,1)
            Wx = 1 / torch.prod(X - X.transpose(0,1) + torch.eye(M+1).to(device),1)
            xdist = xf[0,k].repeat(1,M+1) - xt.repeat(Mf,1)
            if torch.prod(xdist) == 0:
                d_rec[:,k] = d_rec[:,k] + U[:,min_ia]
                continue
            Wt = Wx / xdist
            w = 0
            for ia in range(min_ia,max_ia+1):
                w = w + Wt[0,ia-min_ia]
            
            for ia in range(min_ia,max_ia+1):
                d_rec[:,k] = d_rec[:,k] + Wt[0,ia-min_ia]/w*U[:,ia]
        d_rec = d_rec.transpose(0,1)
    else: #from iregular to regular
        d_rec = torch.zeros(nt,N1).to(device)
        for k in range(nk):
            nx = N1
            l = xn
            pd = torch.ones(nx-kt,1).to(device)
            for s in range(nx-kt):
                for i in range(kt+1):
                    pd[s,0] = pd[s,0] * torch.abs(xf[0,k] - l[s+i,0])
            sj = torch.argmin(pd)
            p1 = sj
            p2 = sj +kt
            xt = xn[p1:p2+1,0]
            min_ia = p1
            max_ia = p2
            X = xt.repeat(M+1,1)
            Wx = 1 / torch.prod(X - X.transpose(0,1) + torch.eye(M+1).to(device),1)
            xdist = xf[0,k].repeat(1,M+1) - xt.repeat(Mf,1)
            for i in range(M+1):
                if xdist[0,i] == 0:
                    xdist[0,i] = 2.2204e-16
            Wt = Wx / xdist
            w = 0
            for ia in range(min_ia,max_ia+1):
                w = w + Wt[0,ia-min_ia]
            
            for ia in range(min_ia,max_ia+1):
                d_rec[:,ia] = d_rec[:,ia] + Wt[0,ia-min_ia]/w*U[:,k]
        d_rec = d_rec.transpose(0,1)
    return d_rec

def interpolator_BL_2D_batch(U_batch,xn,xf_batch,N,type):
    batch_size = U_batch.shape[0]
    nt = U_batch.shape[3]
    N1 = xn.shape[1]
    nk = xf_batch.shape[2]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    U_batch = U_batch.to(device)
    xn = xn.to(device)
    xf_batch = xf_batch.to(device)
    if type == 0:
        D_rec = torch.zeros(batch_size,1,nk,nt).to(device)
    else:
        D_rec = torch.zeros(batch_size,1,N1,nt).to(device)
    for i in range(batch_size):
        U = U_batch[i,:,:,:].squeeze()
        xf = xf_batch[i,:,:].view(1,32)
        D_rec[i,:,:,:] = interpolator_BL_2D(U,xn,xf,N,type,device)
    return D_rec

