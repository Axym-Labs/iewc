# Empirical Tables And Plots

## Main Results

Classification uses three seeds on the full ten-distribution FACIL CIFAR-100 task-aware protocol. Regression and diffusion use five seeds; segmentation uses three seeds on VOC2012 animal/vehicle class-set foreground masks. The main cross-domain rows use diagonal matrix surrogates.

| Task type      | Metric                                                  | Sequential                  | EF-EWC                  | EWC-DR              | IEWC                    | IEWC-SW                    |
| -------------- | ------------------------------------------------------- | --------------------------- | ----------------------- | ------------------- | ----------------------- | -------------------------- |
| Classification | Final avg. TAw accuracy ↑                               | 0.3974 $\pm$ 0.0294         | 0.5617 $\pm$ 0.0062     | 0.5932 $\pm$ 0.0296 | **0.5978 $\pm$ 0.0260** | --                         |
| Classification | Avg. TAw forgetting ↓                                   | 0.4783 $\pm$ 0.0325         | 0.1997 $\pm$ 0.0107     | 0.1039 $\pm$ 0.0085 | **0.0624 $\pm$ 0.0025** | --                         |
| Regression     | Old-distribution MSE after new distribution ↓           | 0.4484 $\pm$ 0.0041         | 0.4374 $\pm$ 0.0075     | --                  | **0.3951 $\pm$ 0.0107** | --                         |
| Regression     | Forgetting (MSE increase) ↓                             | 0.4479 $\pm$ 0.0039         | 0.4369 $\pm$ 0.0076     | --                  | **0.3946 $\pm$ 0.0109** | --                         |
| Regression     | New-distribution MSE ↓                                  | **1.71e-04 $\pm$ 1.43e-04** | 0.0024 $\pm$ 0.0020     | --                  | 0.0075 $\pm$ 0.0016     | --                         |
| Diffusion      | Old-distribution denoising MSE after new distribution ↓ | 0.0705 $\pm$ 0.0086         | 0.0344 $\pm$ 0.0049     | --                  | 0.0332 $\pm$ 0.0045     | **0.0279 $\pm$ 4.60e-04**  |
| Diffusion      | Forgetting (MSE increase) ↓                             | 0.0420 $\pm$ 0.0095         | 0.0058 $\pm$ 0.0066     | --                  | 0.0048 $\pm$ 0.0061     | **-6.52e-04 $\pm$ 0.0020** |
| Diffusion      | New-distribution denoising MSE ↓                        | 0.0540 $\pm$ 0.0041         | **0.0284 $\pm$ 0.0019** | --                  | 0.0288 $\pm$ 0.0021     | 0.0348 $\pm$ 0.0023        |
| Segmentation   | Old class-set foreground IoU after new class set ↑      | 0.2623 $\pm$ 0.0360         | 0.2798 $\pm$ 0.0531     | --                  | **0.2882 $\pm$ 0.0508** | --                         |
| Segmentation   | Forgetting (IoU decrease) ↓                             | 0.1633 $\pm$ 0.0273         | 0.1517 $\pm$ 0.0480     | --                  | **0.1371 $\pm$ 0.0463** | --                         |
| Segmentation   | New class-set foreground IoU ↑                          | **0.4792 $\pm$ 0.0302**     | 0.4709 $\pm$ 0.0158     | --                  | 0.4587 $\pm$ 0.0157     | --                         |

## Contamination

| Method | $p=0$               | $p=0.1$             | $p=0.25$            |
| ------ | ------------------- | ------------------- | ------------------- |
| EF-EWC | 0.5617 $\pm$ 0.0062 | 0.5132 $\pm$ 0.0112 | 0.2773 $\pm$ 0.0320 |
| EWC-DR | 0.5932 $\pm$ 0.0296 | 0.5310 $\pm$ 0.0226 | 0.4598 $\pm$ 0.0095 |
| IEWC   | 0.5978 $\pm$ 0.0260 | 0.5642 $\pm$ 0.0215 | 0.5322 $\pm$ 0.0239 |

## Label-Noise Test of the Contamination Model

| Method | Noise | Entries | Stored-tail mass ratio | Stored-tail profile L1 | Stored-tail log-scale RMSE |
| ------ | ----- | ------- | ---------------------- | ---------------------- | -------------------------- |
| EF-EWC | 0.1   | 512     | 0.0288                 | 0.2610                 | 1.4660                     |
| EF-EWC | 0.25  | 512     | 0.0244                 | 0.2737                 | 1.5337                     |
| IEWC   | 0.1   | 512     | 1.0833                 | 0.0350                 | 0.0350                     |
| IEWC   | 0.25  | 512     | 1.1003                 | 0.0154                 | 0.0415                     |

## Diffusion Output Geometry

| Metric used in IEWC    | Seeds | Old-output SW drift   | Old MSE increase | New-distribution MSE |
| ---------------------- | ----- | --------------------- | ---------------- | -------------------- |
| Euclidean $G=I$        | 5     | 31.7878 $\pm$ 12.4909 | 0.0048           | 0.0288               |
| Sliced-Wasserstein $G$ | 5     | 0.9981 $\pm$ 0.5442   | -6.52e-04        | 0.0348               |

## Plot Files

- `docs/empirical-evidence/artifacts/paper-plots/classification_distribution_retention.png`
- `docs/empirical-evidence/artifacts/paper-plots/contamination_diagonal_spectra.png`
- `docs/empirical-evidence/artifacts/paper-plots/diffusion_generated_samples.png`
- `docs/empirical-evidence/artifacts/paper-plots/diffusion_wasserstein_geometry.png`
- `docs/empirical-evidence/artifacts/paper-plots/summand_norm_ecdf.png`
- `docs/empirical-evidence/artifacts/paper-plots/tau_sensitivity_cifar100.png`
