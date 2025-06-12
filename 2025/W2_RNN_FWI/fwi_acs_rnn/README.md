Using RNN for 2D Acoustic FWI.

Author: Alan Richardson
Modified by: Wenlong Wang

########环境依赖
tensorflow 2.x.x



########使用说明
（1）含code和data两个文件夹
     code：各种运行代码放置处
	 data：速度模型数据以及地震数据放置处


（2）code文件夹
	 setup_mar.py：设置Marmousi模型参数（最大迭代epoch数：num_epochs、early stopping数：n_epochs_stop、batch_size、模型参数路径、模型参数、炮数、震源子波主频等）
	 setup_par.py：设置Camembert模型参数
	 forward2d.py：正演程序
	 fwi.py：FWI程序，在__init__()中可设置不同的loss（L2/NIM/W2），
	         在_train_loop()中保存loss及反演model
	 gen_data.py + make_seis_data.py：生成地震数据
	 gen_resi.py + make_adj_data.py：生成伴随震源
	 make_fwi_adam.py：运行FWI，得到反演loss、model（可设置不同的优化算法）
	 NIM_loss.py：积分归一化函数（（NIM loss）
	 W2_loss.py：线性正变换二次Wasserstein度量函数（W2 loss）
	 W2_square_loss.py：平方正变换二次Wasserstein度量函数（W2 loss）
	 wavelets.py：Ricker子波函数
	 
	 画图
	 plot_loss.py：训练、验证损失
	 plot_model.py：反演得到的速度模型
	 plot_shot.py：某一炮的地震数据
	 plot_source.py：Ricker子波及其频谱图
	 plot_wave.py：波场
	 plot_vel_curve.py：速度曲线
	 calc_r2.py + plot_IC.py：交汇图及其相关系数
	 
	 
（3）./data/model文件夹
	 mar_60_100.bin：Marmousi纵波速度模型，大小为nx*nz = 100*60
	 mar_60_100_init.bin：Marmousi初始速度模型


	 mar_130_340.vp：Marmousi纵波速度模型，大小为nx*nz = 340*130
     mar_130_340_init1.vp：方差20的高斯滤波，初始模型1（较好的初始模型，L2/NIM/W2均可）
	 mar_130_340_init2.vp：方差30的高斯滤波，初始模型2（较差的初始模型，L2/NIM出现cycle skipping）
	
	
（4）参数
	 receivers：观测地震记录（由真实速度模型正演得到），大小为nt*ns*nr（时间*炮数*检波器）
	 out_receivers：模拟地震记录，大小为nt*ns*nr维（时间*炮数*检波器）