# MathGeo

地震数据处理工具箱，地球物理中心，哈尔滨工业大学

## 2025 更新

* DLR

一种通过深度神经网络约束和求解偏移的配准方法。

参考文献：

H. Yue, S. Yu*, Seismic registration with a deep neural network constraint. IEEE Geoscience and Remote Sensing Letters, 2024, 21:7505005.

* DTW_dealiased_interpolation_xyj

用于去假频地震数据插值的开源Matlab代码。

参考文献：

Xu Y, Yu S, Dong L, Ma J. Dealiased seismic data interpolation by dynamic matching[J]. Geophysics, 2024, 89(5): V361-V376.

* FWI-GAN

一种无监督学习范式，在分布意义上估计物理一致的速度模型。

参考文献：

P. Sun, F. Yang*, H. Liang, J. Ma, Full-waveform inversion using a learned regularization, IEEE Transactions on Geoscience and Remote Sensing, 2023, 61, 5920715.

* POCS-Net

一种插值的网络方法。

参考文献：

Y. Chen, S. Yu*, J. Ma, A POCS-Net for 3D seismic data interpolation, Geophysics, 2023, 88(3): 1MJ-V289.

* Regularized_Full-Waveform_Inversion_With_Shearlet_Transform_and_Total_Generalized_Variation

一种提高地下介质中尖锐界面和平滑背景的反演精度的方法。

参考文献：

H. Wang, S. Yu*, Regularized full-waveform inversion with shearlet transform and total generalized variation. IEEE Transactions on Geoscience and Remote Sensing, 2024, 62: 5925115.

* SSL_POCS_Net

用于插值的自监督学习POCS-Net。

参考文献：

Y. Chen, S. Yu, R. Lin, Self-supervised transfer learning POCS-Net for Seismic Data Interpolation, IEEE Transactions on Geoscience and Remote Sensing, 2024, 62: 1-18

* W2_RNN_FWI

基于RNN的全波形反演包，具有Wasserstein损失函数，可减轻周期跳跃。

参考文献：

李燕梅，顾焕申，许凯，孙振涛，王文龙*，2022，基于最优输运与循环神经网络的全波形反演，地球物理学进展, 37(6)，2408-2416.

* DDPM_ir

非均匀插值的生成模型（去噪扩散概率模型，DDPM）。

参考文献：

陈尧，于四伟*，林荣智，基于扩散概率模型的非均一地震数据插值，煤田地质与勘探, 2024, 52(8), 177-186.


## 2020更新

#Deep learning related works

* python_segy

将去噪卷积神经网络用于随机噪声去除。

参考文献:

S. Yu, J. Ma*, W. Wang, Deep learning for denoising, Geophysics, 2019, 84 (6), V333-V350

* FCNVMB

将U-net用于从地震记录中预测速度模型

参考文献:

F. Yang, J. Ma*, Deep-learning inversion: a next generation seismic velocity model building method, Geophysics, 2019, 84 (4), R583-R599.

* CNN-POCS

将CNN和POCS组合起来用于地震数据插值

参考文献:

H. Zhang, X. Yang, J. Ma*, Can learning from natural image denoising be used for seismic data interpolation, Geophysics, 2020, 85 (4)

## 2018更新

* GMD

几何模态分解用于分解含有线性或双曲同相轴的地震数据，并可以用于去噪和插值。

参考文献：

S. Yu, J. Ma, S. Osher, Geometric mode decomposition, Inverse Problem and Imaging, 2018, 12 (4), 831-852.

* GVRO

以局部梯度向量为列构成梯度向量矩阵。对于单倾角信号，梯度向量落在同一直线上，从而对应的梯度向量矩阵可近似为秩一矩阵。对于多倾角信号，局部地震数据分解为多个单倾角信号，约束每一个子信号的梯度向量矩阵近似为秩一矩阵。基于块坐标下降算法可求解此梯度向量矩阵秩一约束模型（GRVO），可应用于地震信号随机噪声压制，以及基于倾角差异实现相干信号分离。

参考文献：

K. Cai, J. Ma, GVRO: gradient vector rank-one regularization with applications on seismic data processing, Geophysical Prospecting, 2018

* SR1

平移秩1方法可以用于表示数据中移动的物体，可以应用于地震数据或超声波图像处理以及视频处理。

参考文献：

F. Bossmann, J. Ma, Enhanced image approximation using shifted rank-1 reconstruction, http://arxiv.org/abs/1810.01681

* TSDL

树字典学习方法基于两个方面：基于字典学习的稀疏表示以及基于图的图像块的相似性。

参考文献：

L. Liu, J. Ma, G. Plonka, Sparse graph-regularized dictionary learning for random seismic noise, Geophysics, 2018, 83 (3), V213-V231.

## 2017

* AGCM:

程序使用非对称高斯线性调频小波模型建立一个不需要字典的正交匹配追踪算法，使用贪婪算法来近似地震数据道

参考文献：

F. Bossmann, J. Ma, Asymmetric chirplet transform for sparse representation of seismic data, Geophysics, 2015, 80 (6), WD89-WD100.

F. Bossmann, J. Ma, Asymmetric chirplet transform ¡ª Part 2: Phase, frequency, and chirp rate, Geophysiscs, 2016, 81(6):V425-V439.

* Decurtain:

程序用一个下卷积模型将受干扰的3D图像到分解成干净的图像以及两种类型的干扰，即条纹部分和层。

参考文献：

Fitschen J H, Ma J, Schuff S. Removal of curtaining effects by a variational model with directional forward differences[J]. Computer Vision & Image Understanding, 2017, 155(13):24-32.

* EMPCR:

我们提出了一种简单有效的随机插值算法，使用了Hankel矩阵。

参考文献：

Jia Y, Yu S, Liu L, et al. A fast rank-reduction algorithm for three-dimensional seismic data interpolation[J]. Journal of Applied Geophysics, 2016, 132:137-145.

* RegistrationMultiComponent:

程序使用基于曲波变换的关联方法来提高数据关联的精度，尤其是对含有噪声的数据

参考文献：

Wang H, Cheng Y, Ma J. Curvelet-based registration of multi-component seismic waves[J]. Journal of Applied Geophysics, 2014, 104(5):90-96.

* DL_toolbox

程序能够同时进行字典学习和去噪

参考文献：

Beckouche S, Ma J. Simultaneous dictionary learning and denoising for seismic data[J]. Geophysics, 2014, 79(3):A27-A31.

* DDTF3D:

程序使用数据驱动紧框架方法对三维地震数据进行了噪声压制以及插值

参考文献：

Yu S, Ma J, Zhang X, et al. Interpolation and denoising of high-dimensional seismic data by learning a tight frame[J]. Geophysics, 2015, 80(5):V119-V132.

* MCDDTF3D:

我们为DDTF方法设计了一种新的样本选取方法。假设方差大的块包含的信息多，对应于复杂结构，因此应以更大概率选入训练集合

参考文献：

Yu S, Ma J, Osher S. Monte Carlo data-driven tight frame for seismic data recovery[J]. Geophysics, 2016, 81(4):V327-V340.

* CVMD:

我们将VMD拓展到复数情况，然后应用于地震数据的f-x谱上，用于去噪

参考文献: 

Yu S, Ma J. Complex Variational Mode Decomposition for Slop-preserving Denoising, summited to IEEE Transactions on Geoscience and Remote Sensing

* LDMM

我们将低维流形模型用于地震数据强噪声压制。LDMM用一个低维流形近似地震数据的所有数据块。

参考文献:

Yu S, et. al. Noise attenuation in a low dimensional manifold, Geophysics, 2017.

* 测试数据下载链接

http://pan.baidu.com/s/1qYwI1IG
