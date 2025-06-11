% DTW_dealiased_interpolation_demo
clc;clear
load hyper.mat
d0 = d(:,2:end); %original data
d2 = d(:,2:2:end); 
[nx,ny]=size(d0);
m2=zeros(nx,ny); m2(:,1:2:end)=1;
d3=d0.*m2;  %sample data 
%% Interpolation process
t1=tic;
for j=1:1
    [nt,nx] = size(d2);
    sx = zeros(nt,nx); sy = sx;
    di = zeros(nt,nx*2-1);
    di(:,1:2:end) = d2;
    for i=1:nx-1
       [Distance, w] = LARDTW(d2(:,i),d2(:,i+1), 1,0.1, 0.2, 5); 
        ix = w(:,1)'; iy = w(:,2)';
        ix2 = [find(diff(ix)==1),length(ix)];
        iy2 = [find(diff(iy)==1),length(ix)];
        ixy = ix(iy2);
        iyx = iy(ix2);
        ixy2 =  linspace(1,nt,nt) + round((ixy-linspace(1,nt,nt))/2); 
        k1=round((ixy-linspace(1,nt,nt))/2);
        iyx2 =  linspace(1,nt,nt) + round((iyx-linspace(1,nt,nt))/2);
        k2= round((iyx-linspace(1,nt,nt))/2);
        d2_ = ( d2(ixy2,i) + d2(iyx2,i+1))/2;
        di(:,2*i) = d2_;
    end
    d2 = di;
end
toc(t1)
snr_xuf(d0,d2)
%Plot 
figure,imagesc(d0,[-1 1]/2);colormap gray;colorbar;
xlabel('Trace number');ylabel('Time sample number');
figure,imagesc(d3,[-1 1]/2);colormap gray;colorbar;
xlabel('Trace number');ylabel('Time sample number');
figure,imagesc(d2,[-1 1]/2);colormap gray;colorbar;
xlabel('Trace number');ylabel('Time sample number');
figure,imagesc(d0-d2,[-1 1]/2);colormap gray;colorbar;
xlabel('Trace number');ylabel('Time sample number');