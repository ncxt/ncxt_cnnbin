# Noise2Noise Binning

*Axel Ekman, Jian-Hua Chen, Venera Weinhardt, Gerry McDermott, Mark A. Le Gros, Carolyn Larabell*

As Presented by: [Axel Ekman](mailto:axel.ekman@iki.fi)
at
[CAMERA Workshop](http://microct.lbl.gov/cameratomo2018)
October 31 - November 2, 2018

## Intro
In terms of signal processing, the optimal digital filter to remove the high-frequency portion of the image is the sinc filter. When decimation is done by an integer factor, area-averaging is usually very close to optimal and produces usually not much aliasing. In this case, downsampling by a factor of 2 can be expressed in the from

<!-- 253 -->
<img src="images/latex_downsample.png" alt="downsampling" height="84" widht="auto"/>

Ideal filters like this are unbiased and do not take into account any priors that may be suitable for the image. The basic idea of this method is that we can construct separate signals from the data and train a CNN to do the downsampling.

Recent work of Lehtinen et al. show that instead of needing true signal, CNN filters can be trained using noisy images as both input and training target by minimizing some distance (loss function) L.

<!-- 150 -->
<img src="images/latex_n2nfilter.png" alt="n2nfilter" height="50" widht="auto"/>

Between the noisy obswervations.

[Noise2Noise: Learning Image Restoration without Clean Data](https://arxiv.org/abs/1803.04189)
```
Lehtinen, Jaakko, et al. “Noise2Noise: Learning Image Restoration without Clean Data.”

Proceedings of the 35th International Conference on Machine Learning, PMLR 80:2965-2974, 2018.
```


Now within the sampling rate of the output image, we can view all pixels corresponding to the same binned pixel as separate observations of the downsampled image. This provides information to optimize some parametrized filter such that we can use the result of Lehtinen et al. to train a CNN downsampler.

<!-- 102 -->
<img src="images/latex_n2nbin.png" alt="n2nbin" height="34" widht="auto"/>


where X1 and X2 are two uncorrelated data samples from the high-resolution image. This can be e.g. done by dividing each downsampled pixel into two diagonal regions (the fact that the center-of-mass is the same should take care of some sub-pixel artifacts). One could also choose random samples of the square to construct several permutations of the same image. In practice this made little difference in the results.

![Schematic](images/schematic.png)


## Examples

### Color images

Below we show the result for 'monarch' in SET14 with artififcial Gaussian  noise (sigma = 50) compared to the mean-binned image.
<!-- The reference methods (TV, NLM, BM3D) show the corresponding result by minimzing the true loss function using oracle information of the the reference image. -->

<!-- ![Schematic](images/monarch_example.png) -->
![Schematic](images/monarch_compare.png)

#### Comparison of different filters for BSD300 and SET14 (PSNR/SSIM)

|Dataset| Mean|TV|NLM|BM3D|CNN|
|:---| :---:|:---:|:---:|:---:|:---:|
|Gaussian , sigma = 30||||||
|BSD300|24.80 / 0.75|29.19 / 0.90|28.76 / 0.88|29.50 / 0.91|30.70 / 0.93  |
|Set14|24.87 / 0.77|29.16 / 0.91|29.15 / 0.90|29.33 / 0.92|30.70 / 0.94   |
|Poisson noise, lambda = 10||||||
|BSD300|24.33 / 0.74|28.60 / 0.89|27.83 / 0.85|29.11 / 0.90|30.65 / 0.93  |
|Set14|23.36 / 0.72|27.73 / 0.89|27.38 / 0.86|28.22 / 0.90|30.17 / 0.93   |
            
The [TV denoising](https://doi.org/10.1023/B:JMIV.0000011325.36760.1e) and [Non-Local Means](https://doi.org/10.5201/ipol.2014.120) were done using the implementations in [scikit-ikmage](https://scikit-image.org/). NLM was done with patch size 5 and patch distance of 6. The [BM3D](https://doi.org/10.1109/TIP.2007.901238) was done using the implementation in [pybm3d](\url{https://pypi.org/project/pybm3d/) with default parameters. For the reference methods, free parameters were optimized by minimizing the L2 norm using oracle information.

### Tomography

Example of a SXT reconstruction of a Human B-cell reconstructed with FBP (Ram-Lak). In this example, the net was trained simultaneously on all projection images.

|Binned projections | CNN binned projections |
| ---|---|
|![FBP noisy](images/fbp_stackm_noisy.png)|![FBP filtered](images/fbp_stackcnn_noisy.png)|


#### Credit where credit is due

Supported by:

[![CZI](images/czi-logo.jpg)](https://www.czbiohub.org/)


Encoder-decoder neural network implementation adapted from the UNet implementation of [jaxony](https://github.com/jaxony/unet-pytorch).

Summary function for PyTorch modules adapted from [sksq96](https://github.com/sksq96/pytorch-summary/tree/master/torchsummary).











