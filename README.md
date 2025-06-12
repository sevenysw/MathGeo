# MathGeo
A toolbox for seismic data processing from Center of Geopyhsics, Harbin Institute of Technology, China

## 2025 updates

## 2020 updates



# Deep learning related works

* python_segy

A Denoised CNN is use for attenuation of random noise.

Reference:

S. Yu, J. Ma, W. Wang, Deep learning for denoising, Geophysics, 2019, 84 (6), V333-V350

* FCNVMB

A U-Net is used for predicting velocity model from raw prestack seismic records.

Reference:

F. Yang, J. Ma, Deep-learning inversion: a next generation seismic velocity model building method, Geophysics, 2019, 84 (4), R583-R599.

* CNN-POCS

A combination of CNN denoiser and POCS is used for seismic interpolation. 

Reference:

H. Zhang, X. Yang, J. Ma, Can learning from natural image denoising be used for seismic data interpolation, Geophysics, 2020, 85 (4)

## 2018 updates

* GMD

Geometric mode decomposition (GMD) is designed to decompose seismic signal with linear or hpyerbolic events, with applications to denoising and interpolation.

Reference:

S. Yu, J. Ma, S. Osher, Geometric mode decomposition, Inverse Problem and Imaging, 2018, 12 (4), 831-852.

* GVRO

Gradient-vector matrices are formed by collecting gradient vectors in a local seismic patch as columns. For single-dip signals, the gradient vectors will group along same lines. So, gradient-vector matrices should be approximately rank-one matrices. For multi-dip signals, the local seismic data are decomposed into single-dip components, with each components’ gradient-vector matrices regularized to be rank-one matrices. The proposed gradient-vector rank-one regularization (GVRO) model is solved in the frame work of block coordinate descending algorithm, and can be used for random noise attenuation and coherent signals separation according to the dip differences.


Reference:

K. Cai, J. Ma, MDCA: multidirectional component analysis for robust estimation of multiple local dips, IEEE Transactions on Geoscience and Remote Sensing, 2019, 57 (5), 2798-2810.

* SR1

We presented a generalization of the low-rank approximation, which allows to individually shift the column of rank-1 matrices (SR1). This model was designed to represent objects that move through the data. This holds in applications such as seismic or ultrasonic image processing as well as video processing. 

Reference:

F. Bossmann, J. Ma, Enhanced image approximation using shifted rank-1 reconstruction, Inverse Problems and Imaging, 2020, 14 (2), 267-290.

* TSDL

We proposed a  tree structure dictionary learning (TSDL) method.  Our approach is based on two components: a sparse data representation in a learned dictionary and a similarity measure for image patches that is evaluated using the Laplacian matrix of a graph.

Reference:

L. Liu, J. Ma, G. Plonka, Sparse graph-regularized dictionary learning for random seismic noise, Geophysics, 2018, 83 (3), V213-V231.

## 2017

* AGCM:

We have used the asymmetric Gaussian chirplet model (AGCM) and established a dictionary-free variant of the orthogonal matching pursuit, a greedy algorithm for sparse approximation of seismic traces.

Reference:

F. Bossmann, J. Ma, Asymmetric chirplet transform for sparse representation of seismic data, Geophysics, 2015, 80(6):WD89-WD100.

F. Bossmann, J. Ma, Asymmetric chirplet transform Part 2: Phase, frequency, and chirp rate, Geophysiscs, 2016, 81(6):V425-V439.

* Decurtain:

An infimal convolution model is applied to split the corrupted 3D image into the clean image and two types of corruptions, namely a striped part and a laminar one.

Reference: 

J. Fitschen, J. Ma, S. Schuff, Removel of curtaining effects by a variational model with directional first and second order differences, Computer Vision and Image Understanding, 2017, 155, 24-32.

* EMPCR:

We propose a simple yet efficient interpolation algorithm, which is based on the Hankel matrix, for randomly missing traces.

Reference:

Y. Jia, S. Yu, L. Liu, J. Ma, A fast rank-reduction algorithm for three-dimensional seismic data interpolation. Journal of Applied Geophysics, 2016, 132:137-145.

* RegistrationMultiComponent:

We propose a new curvelet-based registration method to improve the precision of registration, especially for the data with heavy random noises.

Reference:

H. Wang, Y. Cheng, J. Ma, Curvelet-based registration of multi-component seismic waves. Journal of Applied Geophysics, 2014, 104(5):90-96.

* DL_toolbox

We propose a simultaneous dictionary learning and denoising method for seismic data.

Reference:

S. Beckouche, J. Ma, Simultaneously dictionary learning and denoising for seismic data, Geophysics, 2014, 79 (3), A27-A31.

* DDTF3D:

We study an application of the data-driven tight frame (DDTF) method to noise suppression and interpolation of high-dimensional seis- mic data.

Reference: 

S. Yu, J. Ma, X. Zhang, M. Sacchi, Interpolation and denoising of high-dimensional seismic data by learning a tight frame, Geophysics, 2015, 80 (5), V119-V132.

* MCDDTF3D:

We have designed a new patch selection method for DDTF seismic data recovery. We suppose that patches with higher variance contain more information related to complex structures, and should be selected into the training set with higher probability.

Reference: 

S. Yu, J. Ma, S. Osher, Monte Carlo data-driven tight frame for seismic data recovery, Geophysics, 2016, 81 (4), V327-V340.

* CVMD:

We have extended varitional mode decomposition to complex-valued situation and apply CVMD to f-x spectrum of seismic for denoising.

Reference: 

S. Yu, J. Ma, Complex variational model decomposition for slop-preserving denoising, IEEE Transactions on Geoscience and Remote Sensing, 2018, 56 (1), 586 - 597.

* LDMM

We have applied low dimensional manifold method for seismic strong noise attenuation. LDMM uses a low dimensional method to approximate all the patches of seismic data. 

Reference:

S. Yu, S. Osher, J. Ma, Z. Shi, Noise attenuation in a low dimensional manifold, Geophysics, 2017, 82 (5), V321-V334.

* test data download link

http://pan.baidu.com/s/1qYwI1IG
