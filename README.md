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

### 1. Network architecture
![RUNOOB 图标](./results/fig1.jpg)

### 2. Real noisy images
![RUNOOB 图标](./results/fig2.jpg)

### 3. Effectiveness of key techniques in the DudeNet for image denoising
![RUNOOB 图标](./results/Table1.jpg)

### 4. Run-time of key techniques in the DudeNet for different sizes noisy images
![RUNOOB 图标](./results/Table2.jpg)


### Test Results
#### 5. DudeNet for BSD68
![RUNOOB 图标](./results/Table4.jpg)

#### 6. DudeNet for Set12
![RUNOOB 图标](./results/Table5.jpg)

#### 7. DudeNet for CBSD68 and Kodak24
![RUNOOB 图标](./results/Table6.jpg)

#### 8. DudeNet for real noisy images 
![RUNOOB 图标](./results/Table7.jpg)

#### 9. Run-time of DudeNet for a noisy image of different sizes.
![RUNOOB 图标](./results/Table9.jpg)

### 10.Complexity analysis of different networks.
![RUNOOB 图标](./results/Table3.jpg)

#### 11. Visual results of Set12
![RUNOOB 图标](./results/fig3.jpg)

#### 12. Visual results of BSD68
![RUNOOB 图标](./results/fig4.jpg)

#### 13. Visual results of CBSD68
![RUNOOB 图标](./results/fig5.jpg)

#### 14. Visual results of Kodak24 
![RUNOOB 图标](./results/fig6.jpg)
