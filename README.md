
## Uncertainty Calibration for Latent-Variable Regression Models

Julia toolbox accompanying:

Duma, Z.-S., Lamminpää, O., Susiluoto, J., Haario, H., Zheng, T.,
Sihvonen, T., Braverman, A., Townsend, P. A., Reinikainen, S.-P.
(2025).\
*Uncertainty calibration for latent-variable regression models.*\
arXiv:2512.23444v1.

------------------------------------------------------------------------

## Overview

Latent-variable (LV) regression methods such as:

-   PCR (Principal Component Regression)\
-   PLS (Partial Least Squares)\
-   K-PCR (Kernel PCR)\
-   K-PLS (Kernel PLS)

are widely used in spectroscopy, chemometrics, and remote sensing.

However, these models traditionally produce point predictions only,
without calibrated predictive uncertainty.

This toolbox implements the LV-localized uncertainty calibration method
proposed in the accompanying paper.

The method:

-   Is inspired by split conformal prediction\
-   Produces input-dependent prediction intervals (PIs)\
-   Localizes uncertainty in latent-variable space\
-   Aggregates LV-specific residual quantiles using explained-variance
    weights\
-   Is compatible with linear and kernel LV models

------------------------------------------------------------------------

## Method Summary

### 1. Data Partitioning

The dataset is divided into:

-   **Training set** -- model fitting\
-   **Calibration set** -- residual quantile estimation\
-   **Test set** -- evaluation only

Typical split: 40% training, 40% calibration, 20% test.

### 2. Model Fitting

PCR, PLS, K-PCR, or K-PLS is fitted on the training set only.

For kernel models:

-   Construct kernel (Gram) matrix\
-   Add ridge term (δI) to training kernel only\
-   Center kernel matrices with respect to training data

### 3. Calibration in Latent Variable Space

For the calibration set:

-   Project samples onto retained LVs\

-   Discretize each LV into k intervals\

-   Compute absolute residuals:

        s_j = |y_j − f̂(x_j)|

-   Compute empirical (1 − α) quantiles within each LV interval\

-   Weight LVs by explained variance

### 4. Prediction for New Samples

For a new sample:

-   Project onto LVs\

-   Determine LV interval membership\

-   Retrieve interval-specific quantiles\

-   Weight using explained variance\

-   Construct prediction interval:

        [ŷ − q(x), ŷ + q(x)]

------------------------------------------------------------------------

## Implemented Models

### Linear

-   PCR\
-   PLS (SIMPLS / NIPALS)

### Kernel

-   K-PCR\
-   K-PLS

Supported kernels:

-   RBF\
-   Cauchy\
-   Additive\
-   Anisotropic RBF (individual σᵢ per variable)

------------------------------------------------------------------------

## Kernel Parameter Optimization (Optional)

Kernel parameters may be:

-   Fixed by user\
-   Optimized via kernel flows

Options include:

-   Global kernel width\
-   Individual per-variable parameters\
-   Combined kernels\
-   Gradient-based optimization\
-   Optional gradient clipping

------------------------------------------------------------------------

## Case Studies Included

-   Case 2 -- Soil Moisture Regression 

Felix M. Riese and Sina Keller, "Hyperspectral benchmark dataset on soil moisture", Dataset, Zenodo, 2018. 

@misc{riesekeller2018,
    author = {Riese, Felix~M. and Keller, Sina},
    title = {Hyperspectral benchmark dataset on soil moisture},
    year = {2018},
    DOI = {10.5281/zenodo.1227837},
    publisher = {Zenodo},
    howpublished = {\href{https://doi.org/10.5281/zenodo.1227837}{doi.org/10.5281/zenodo.1227837}}
}\
-   Case 3 -- Corn NIR Dataset https://www.eigenvector.com/data/Corn/ \
-   Case 4 -- Plant Trait Estimation Zheng, T., Queally, N., Chadwick, K. D., Cryer, J., Reim, P., Villanueva-Weeks, C., ... & Williams, A. (2024). SBG High Frequency Time Series (SHIFT). ORNL Distributed Active Archive Center (DAAC) dataset 10.3334/ORNLDAAC/2337 (2024, 2337.

@article{zheng2024sbg,
  title={SBG High Frequency Time Series (SHIFT)},
  author={Zheng, T and Queally, N and Chadwick, KD and Cryer, J and Reim, P and Villanueva-Weeks, C and Townsend, P and Berg, M and Breuer, Z and Burkard, N and others},
  journal={ORNL Distributed Active Archive Center (DAAC) dataset 10.3334/ORNLDAAC/2337 (2024},
  pages={2337},
  year={2024}
}

------------------------------------------------------------------------

## Outputs

Each run returns:

-   `yPredTest`\
-   `qTest` (PI half-widths)\
-   Empirical coverage\
-   Interval width statistics\
-   Diagnostic plots

------------------------------------------------------------------------

## Citation

If you use this toolbox, please cite:

    Duma, Z.-S. et al. (2025).
    Uncertainty calibration for latent-variable regression models.
    arXiv:2512.23444v1.

------------------------------------------------------------------------

## Contact

Zina-Sabrina Duma\
Zina-Sabrina.Duma@lut.fi
