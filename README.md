# ACNet
## Asymmetric CNN for image super-resolution
## This paper is conducted by Chunwei Tian, Yong Xu, Wangmeng Zuo, Chia-Wen Lin and David Zhang. It is implemented by Pytorch. 
## Besides, it is accepted by the IEEE Transactions on Systmes, Man, and  Cybernetics: Systems (IEEE TSMC, IF:9.309) in 2021. (Acceptance rate of 10%)
## This paper can be obtained at https://arxiv.org/abs/2103.13634.

## Abstract
### Deep convolutional neural networks (CNNs) have been widely applied for low-level vision over the past five years. According to nature of different applications, designing appropriate CNN architectures is developed. However, customized architectures gather different features via treating all pixel points as equal to improve the performance of given application, which ignores the effects of local power pixel points and results in low training efficiency. In this paper, we propose an asymmetric CNN (ACNet) comprising an asymmetric block (AB), a memory enhancement block (MEB) and a high-frequency feature enhancement block (HFFEB) for image super-resolution. The AB utilizes one-dimensional asymmetric convolutions to intensify the square convolution kernels in horizontal and vertical directions for promoting the influences of local salient features for SISR. The MEB fuses all hierarchical low-frequency features from the AB via residual learning (RL) technique to resolve the long-term dependency problem and transforms obtained low-frequency features into high-frequency features. The HFFEB exploits low- and high-frequency features to obtain more robust super-resolution features and address excessive feature enhancement problem. Additionally, it also takes charge of reconstructing a high-resolution (HR) image. Extensive experiments show that our ACNet can effectively address single image super-resolution (SISR), blind SISR and blind SISR of blind noise problems. The code of the ACNet is shown at https://github.com/hellloxiaotian/ACNet.


## Requirements (Pytorch)  
#### Pytorch 0.41
#### Python 2.7
#### torchvision 
#### openCv for Python
#### HDF5 for Python

### 1. Network architecture of ACNet
![RUNOOB 图标](./results/fig1.png)

### 2. Implementations of the sub-pixel convolution.
![RUNOOB 图标](./results/fig2.png)


### Test Results
### 3. Average PSNR/SSIM values of different methods for three scale factors of x2, x3 and 4 on the Set5.
![RUNOOB 图标](./results/table4.png)

### 4. Average PSNR/SSIM values of different methods for three scale factors of x2, x3 and 4 on the Set14.
![RUNOOB 图标](./results/table5.png)


### 5. Average PSNR/SSIM values of different methods for three scale factors of x2, x3 and 4 on the B100.
![RUNOOB 图标](./results/table6.png)

### 6. Average PSNR/SSIM values of different methods for three scale factors of x2, x3 and 4 on the U100.
![RUNOOB 图标](./results/table7.png)

### 7. Complexity of five methods in SISR.
![RUNOOB 图标](./results/table8.png)

### 8. Running time (Seconds) of five methods on the given LR images of sizes 128x128, 256x256 and 512x512 for scale factor of x2.
![RUNOOB 图标](./results/table9.png)

### 9. Average FSIM values of different methods with three scale factors of x2, x3 and x4 on the B100.
![RUNOOB 图标](./results/table10.png)

### 10. Average PSNR/SSIM values of different methods for noise level of 15 with three scale factors of x2, x3 and x4 on the Set5, Set14, B100 and U100. 
![RUNOOB 图标](./results/table11.png)


#### 11.  Average PSNR/SSIM values of different methods for noise level of 25 with three scale factors of x2, x3 and x4 on the Set5, Set14, B100 and U100. 
![RUNOOB 图标](./results/table12.png)

#### 12. Average PSNR/SSIM values of different methods for noise level of 35 with three scale factors of x2, x3 and x4 on the Set5, Set14, B100 and U100. 
![RUNOOB 图标](./results/table13.png)

#### 13. Average PSNR/SSIM values of different methods for noise level of 50 with three scale factors of x2, x3 and x4 on the Set5, Set14, B100 and U100. 
![RUNOOB 图标](./results/table14.png)


#### 14. Visual results of Kodak24 
![RUNOOB 图标](./results/fig6.jpg)
