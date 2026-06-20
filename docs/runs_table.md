# Empirical Run Table

This tracked table summarizes the empirical-2 run families currently present in
`docs/empirical-2/artifacts/`. The raw JSON artifacts and plots remain ignored,
but the selected/tuned results below are kept in the repository for review. The
`grid pts` column counts the lambda values evaluated for that method, including
edge extensions; `edge ext.` counts how many edge-extension lambda values were
added by the tuning runner. Entries with multiple seeds report mean `+-` sample
standard deviation.

## Vision Classification

| Run family | Method | lambda | grid pts | edge ext. | seeds | Final avg accuracy | Forgetting |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| smoke: ImageNet-R ViT-tiny LoRA, 2x3, task-aware | Sequential | - | - | - | 0 | 0.3750 | 0.0000 |
| smoke: ImageNet-R ViT-tiny LoRA, 2x3, task-aware | EF-EWC | 1 | 2 | 0 | 0 | 0.3750 | 0.0000 |
| smoke: ImageNet-R ViT-tiny LoRA, 2x3, task-aware | IEWC-GSS | 1 | 2 | 0 | 0 | 0.3750 | 0.0000 |
| ImageNet-R ViT-B/16 LoRA, 10x20, class-incremental | Sequential | - | - | - | 0,1,2 | 0.2896 +- 0.0405 | 0.7066 +- 0.0466 |
| ImageNet-R ViT-B/16 LoRA, 10x20, class-incremental | EF-EWC | 333.333 | 5 | 2 | 0,1,2 | 0.5981 +- 0.0337 | 0.3595 +- 0.0378 |
| ImageNet-R ViT-B/16 LoRA, 10x20, class-incremental | EWC-DR | 3 | 4 | 1 | 0,1,2 | 0.5687 +- 0.0263 | 0.3933 +- 0.0286 |
| ImageNet-R ViT-B/16 LoRA, 10x20, class-incremental | IEWC | 33.3333 | 6 | 3 | 0,1,2 | 0.5960 +- 0.0295 | 0.3650 +- 0.0341 |
| ImageNet-R ViT-B/16 LoRA, 10x20, class-incremental | IEWC-GSS | 11.1111 | 6 | 3 | 0,1,2 | 0.6250 +- 0.0263 | 0.3285 +- 0.0278 |
| ImageNet-R ViT-B/16 LoRA, 10x20, class-incremental | IEWC-FROMP | 33.3333 | 5 | 2 | 0,1,2 | 0.6365 +- 0.0165 | 0.3068 +- 0.0170 |
| ImageNet-R ViT-B/16 LoRA, 5x20, class-incremental | Sequential | - | - | - | 0 | 0.2920 | 0.7870 |
| ImageNet-R ViT-B/16 LoRA, 5x20, class-incremental | EF-EWC | 3000 | 9 | 2 | 0 | 0.6941 | 0.2592 |
| ImageNet-R ViT-B/16 LoRA, 5x20, class-incremental | EWC-DR | 10 | 7 | 0 | 0 | 0.6465 | 0.3314 |
| ImageNet-R ViT-B/16 LoRA, 5x20, class-incremental | IEWC | 1000 | 8 | 1 | 0 | 0.7337 | 0.2029 |
| ImageNet-R ViT-B/16 LoRA, 5x20, class-incremental | IEWC-GSS | 300 | 7 | 0 | 0 | 0.7254 | 0.2088 |
| ImageNet-R ViT-B/16 LoRA, 5x20, class-incremental | IEWC-FROMP | 300 | 7 | 0 | 0 | 0.7262 | 0.2111 |

## Time-Series Forecasting / Regression

| Run family | Method | lambda | grid pts | edge ext. | seeds | Final avg MSE | Forgetting MSE |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| ETT transformer forecaster, 4 distributions | Sequential | - | - | - | 0,1,2 | 0.2093 +- 0.0127 | 0.0341 +- 0.0108 |
| ETT transformer forecaster, 4 distributions | EF-EWC | 0.00037037 | 9 | 3 | 0,1,2 | 0.1985 +- 0.0056 | 0.0268 +- 0.0039 |
| ETT transformer forecaster, 4 distributions | IEWC | 81 | 9 | 3 | 0,1,2 | 0.1923 +- 0.0058 | 0.0159 +- 0.0045 |
| ETT transformer forecaster, 4 distributions | IEWC-GSS | 81 | 9 | 3 | 0,1,2 | 0.1921 +- 0.0057 | 0.0153 +- 0.0047 |

## NLP Text Classification

| Run family | Method | lambda | grid pts | edge ext. | seeds | Final avg accuracy | Forgetting |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| T5-small LoRA GLUE guard, SST-2 -> MRPC -> QQP | Sequential | - | - | - | 0 | 0.5505 | 0.2491 |
| T5-small LoRA GLUE guard, SST-2 -> MRPC -> QQP | EF-EWC | 2700 | 8 | 2 | 0 | 0.5914 | 0.1716 |
| T5-small LoRA GLUE guard, SST-2 -> MRPC -> QQP | EWC-DR | 2700 | 8 | 2 | 0 | 0.5893 | 0.1777 |
| T5-small LoRA GLUE guard, SST-2 -> MRPC -> QQP | IEWC | 2700 | 8 | 2 | 0 | 0.5899 | 0.1777 |
| T5-small LoRA GLUE guard, SST-2 -> MRPC -> QQP | IEWC-GSS | 2700 | 8 | 2 | 0 | 0.5886 | 0.1777 |
| T5-small LoRA GLUE guard, SST-2 -> MRPC -> QQP | IEWC-FROMP | 900 | 8 | 2 | 0 | 0.5900 | 0.1728 |
| T5-base LoRA GLUE, SST-2 -> MRPC -> QQP | Sequential | - | - | - | 0,1,2 | 0.6937 +- 0.0214 | 0.1543 +- 0.0366 |
| T5-base LoRA GLUE, SST-2 -> MRPC -> QQP | EF-EWC | 2700 | 5 | 0 | 0,1,2 | 0.7447 +- 0.0123 | 0.0347 +- 0.0148 |
| T5-base LoRA GLUE, SST-2 -> MRPC -> QQP | EWC-DR | 33.3333 | 8 | 3 | 0,1,2 | 0.7211 +- 0.0117 | 0.0948 +- 0.0129 |
| T5-base LoRA GLUE, SST-2 -> MRPC -> QQP | IEWC | 900 | 5 | 0 | 0,1,2 | 0.7443 +- 0.0134 | 0.0372 +- 0.0141 |
| T5-base LoRA GLUE, SST-2 -> MRPC -> QQP | IEWC-GSS | 900 | 5 | 0 | 0,1,2 | 0.7058 +- 0.0205 | 0.0662 +- 0.0251 |
| T5-base LoRA GLUE, SST-2 -> MRPC -> QQP | IEWC-FROMP | 900 | 5 | 0 | 0,1,2 | 0.7117 +- 0.0217 | 0.0645 +- 0.0282 |
