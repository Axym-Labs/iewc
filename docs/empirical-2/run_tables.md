# Raw Empirical Training Runs

This tracked ledger records every individual empirical-2 training-run JSON currently present in `docs/empirical-2/artifacts/`, excluding only `*-tuning-summary.json` files because those summarize multiple runs rather than train a model. Raw artifacts and plots are ignored, but this per-run index is committed for auditability.

Total logged training runs: `330`.

## Vision Classification

Total runs: `140`.

| artifact | method | seed | lambda | dataset | model | adaptation | distributions | evaluation | epochs | samples/class | final avg accuracy | forgetting |
| --- | --- | ---: | ---: | --- | --- | --- | --- | --- | ---: | --- | ---: | ---: |
| `pilot-cifar100-vit-lora-ef-lam1.json` | EF-EWC | 0 | 1 | cifar100 | vit_tiny_patch16_224 | lora | 3x5 | - | 1 | train=20, test=10 | 0.3933 | 0.07 |
| `pilot-cifar100-vit-lora-ef-lam10.json` | EF-EWC | 0 | 10 | cifar100 | vit_tiny_patch16_224 | lora | 3x5 | - | 1 | train=20, test=10 | 0.3733 | 0.07 |
| `pilot-cifar100-vit-lora-ef-lam30.json` | EF-EWC | 0 | 30 | cifar100 | vit_tiny_patch16_224 | lora | 3x5 | - | 1 | train=20, test=10 | 0.38 | 0.07 |
| `pilot-cifar100-vit-lora-ef.json` | EF-EWC | 0 | 100 | cifar100 | vit_tiny_patch16_224 | lora | 3x5 | - | 1 | train=20, test=10 | 0.36 | 0.04 |
| `pilot-cifar100-vit-lora-iewc-fromp.json` | IEWC-FROMP | 0 | 100 | cifar100 | vit_tiny_patch16_224 | lora | 3x5 | - | 1 | train=20, test=10 | 0.3467 | 0.04 |
| `pilot-cifar100-vit-lora-iewc-gss.json` | IEWC-GSS | 0 | 100 | cifar100 | vit_tiny_patch16_224 | lora | 3x5 | - | 1 | train=20, test=10 | 0.3533 | 0.04 |
| `pilot-cifar100-vit-lora-iewc-lam1.json` | IEWC | 0 | 1 | cifar100 | vit_tiny_patch16_224 | lora | 3x5 | - | 1 | train=20, test=10 | 0.3933 | 0.07 |
| `pilot-cifar100-vit-lora-iewc-lam10.json` | IEWC | 0 | 10 | cifar100 | vit_tiny_patch16_224 | lora | 3x5 | - | 1 | train=20, test=10 | 0.3867 | 0.06 |
| `pilot-cifar100-vit-lora-iewc-lam30.json` | IEWC | 0 | 30 | cifar100 | vit_tiny_patch16_224 | lora | 3x5 | - | 1 | train=20, test=10 | 0.38 | 0.06 |
| `pilot-cifar100-vit-lora-iewc.json` | IEWC | 0 | 100 | cifar100 | vit_tiny_patch16_224 | lora | 3x5 | - | 1 | train=20, test=10 | 0.3533 | 0.04 |
| `pilot-cifar100-vit-lora-sequential.json` | Sequential | 0 | - | cifar100 | vit_tiny_patch16_224 | lora | 3x5 | - | 1 | train=20, test=10 | 0.3933 | 0.07 |
| `pilot-cifar100-vit-scratch-ef-stronger.json` | EF-EWC | 0 | 100 | cifar100 | vit_tiny_patch16_224 | full | 3x5 | - | 8 | train=100, test=20 | 0.13 | 0.415 |
| `pilot-cifar100-vit-scratch-iewc-stronger.json` | IEWC | 0 | 100 | cifar100 | vit_tiny_patch16_224 | full | 3x5 | - | 8 | train=100, test=20 | 0.1333 | 0.425 |
| `pilot-cifar100-vit-scratch-sequential-stronger.json` | Sequential | 0 | - | cifar100 | vit_tiny_patch16_224 | full | 3x5 | - | 8 | train=100, test=20 | 0.11 | 0.405 |
| `pilot-cifar100-vit-scratch-sequential.json` | Sequential | 0 | - | cifar100 | vit_tiny_patch16_224 | full | 3x5 | - | 2 | train=20, test=10 | 0.0933 | 0.22 |
| `pilot-imagenet-r-vit-lora-ef-l100-3x10-50shot-e5-taskaware.json` | EF-EWC | 0 | 100 | imagenet_r | vit_tiny_patch16_224 | lora | 3x10 | task_aware | 5 | train=50, test=20 | 0.719 | 0.1896 |
| `pilot-imagenet-r-vit-lora-ewcdr-l100-3x10-50shot-e5-taskaware.json` | EWC-DR | 0 | 100 | imagenet_r | vit_tiny_patch16_224 | lora | 3x10 | task_aware | 5 | train=50, test=20 | 0.7813 | 0.0842 |
| `pilot-imagenet-r-vit-lora-iewc-fromp-l100-3x10-50shot-e5-taskaware.json` | IEWC-FROMP | 0 | 100 | imagenet_r | vit_tiny_patch16_224 | lora | 3x10 | task_aware | 5 | train=50, test=20 | 0.8463 | 0.0079 |
| `pilot-imagenet-r-vit-lora-iewc-gss-l100-3x10-50shot-e5-taskaware.json` | IEWC-GSS | 0 | 100 | imagenet_r | vit_tiny_patch16_224 | lora | 3x10 | task_aware | 5 | train=50, test=20 | 0.8441 | 0.0079 |
| `pilot-imagenet-r-vit-lora-iewc-l100-3x10-50shot-e5-taskaware.json` | IEWC | 0 | 100 | imagenet_r | vit_tiny_patch16_224 | lora | 3x10 | task_aware | 5 | train=50, test=20 | 0.8368 | 0.027 |
| `pilot-imagenet-r-vit-lora-sequential-3x10-10shot-e1-taskaware.json` | Sequential | 0 | - | imagenet_r | vit_tiny_patch16_224 | lora | 3x10 | task_aware | 1 | train=10, test=5 | 0.38 | 0 |
| `pilot-imagenet-r-vit-lora-sequential-3x10-10shot-e1.json` | Sequential | 0 | - | imagenet_r | vit_tiny_patch16_224 | lora | 3x10 | - | 1 | train=10, test=5 | 0.2333 | 0.04 |
| `pilot-imagenet-r-vit-lora-sequential-3x10-50shot-e5-taskaware.json` | Sequential | 0 | - | imagenet_r | vit_tiny_patch16_224 | lora | 3x10 | task_aware | 5 | train=50, test=20 | 0.7599 | 0.1327 |
| `run-imagenet-r-vit-lora-ef-l100-seed0-5x10-50shot-e5-taskaware.json` | EF-EWC | 0 | 100 | imagenet_r | vit_tiny_patch16_224 | lora | 5x10 | task_aware | 5 | train=50, test=20 | 0.7548 | 0.0958 |
| `run-imagenet-r-vit-lora-ef-l1000-seed0-5x10-50shot-e5-taskaware.json` | EF-EWC | 0 | 1000 | imagenet_r | vit_tiny_patch16_224 | lora | 5x10 | task_aware | 5 | train=50, test=20 | 0.7961 | 0.0562 |
| `run-imagenet-r-vit-lora-ef-l1000-seed1-5x10-50shot-e5-taskaware.json` | EF-EWC | 1 | 1000 | imagenet_r | vit_tiny_patch16_224 | lora | 5x10 | task_aware | 5 | train=50, test=20 | 0.7535 | 0.1175 |
| `run-imagenet-r-vit-lora-ef-l1000-seed2-5x10-50shot-e5-taskaware.json` | EF-EWC | 2 | 1000 | imagenet_r | vit_tiny_patch16_224 | lora | 5x10 | task_aware | 5 | train=50, test=20 | 0.7828 | 0.0571 |
| `run-imagenet-r-vit-lora-ef-l30-seed0-5x10-50shot-e5-taskaware.json` | EF-EWC | 0 | 30 | imagenet_r | vit_tiny_patch16_224 | lora | 5x10 | task_aware | 5 | train=50, test=20 | 0.7451 | 0.1037 |
| `run-imagenet-r-vit-lora-ef-l300-seed0-5x10-50shot-e5-taskaware.json` | EF-EWC | 0 | 300 | imagenet_r | vit_tiny_patch16_224 | lora | 5x10 | task_aware | 5 | train=50, test=20 | 0.7663 | 0.0841 |
| `run-imagenet-r-vit-lora-ewcdr-l10-seed0-5x10-50shot-e5-taskaware.json` | EWC-DR | 0 | 10 | imagenet_r | vit_tiny_patch16_224 | lora | 5x10 | task_aware | 5 | train=50, test=20 | 0.7431 | 0.0998 |
| `run-imagenet-r-vit-lora-ewcdr-l100-seed0-5x10-50shot-e5-taskaware.json` | EWC-DR | 0 | 100 | imagenet_r | vit_tiny_patch16_224 | lora | 5x10 | task_aware | 5 | train=50, test=20 | 0.6217 | 0.233 |
| `run-imagenet-r-vit-lora-ewcdr-l3-seed0-5x10-50shot-e5-taskaware.json` | EWC-DR | 0 | 3 | imagenet_r | vit_tiny_patch16_224 | lora | 5x10 | task_aware | 5 | train=50, test=20 | 0.7688 | 0.0808 |
| `run-imagenet-r-vit-lora-ewcdr-l3-seed1-5x10-50shot-e5-taskaware.json` | EWC-DR | 1 | 3 | imagenet_r | vit_tiny_patch16_224 | lora | 5x10 | task_aware | 5 | train=50, test=20 | 0.7776 | 0.0836 |
| `run-imagenet-r-vit-lora-ewcdr-l3-seed2-5x10-50shot-e5-taskaware.json` | EWC-DR | 2 | 3 | imagenet_r | vit_tiny_patch16_224 | lora | 5x10 | task_aware | 5 | train=50, test=20 | 0.8119 | 0.0407 |
| `run-imagenet-r-vit-lora-ewcdr-l30-seed0-5x10-50shot-e5-taskaware.json` | EWC-DR | 0 | 30 | imagenet_r | vit_tiny_patch16_224 | lora | 5x10 | task_aware | 5 | train=50, test=20 | 0.6989 | 0.1449 |
| `run-imagenet-r-vit-lora-iewc-fromp-l100-seed0-5x10-50shot-e5-taskaware.json` | IEWC-FROMP | 0 | 100 | imagenet_r | vit_tiny_patch16_224 | lora | 5x10 | task_aware | 5 | train=50, test=20 | 0.8148 | 0.0106 |
| `run-imagenet-r-vit-lora-iewc-fromp-l100-seed1-5x10-50shot-e5-taskaware.json` | IEWC-FROMP | 1 | 100 | imagenet_r | vit_tiny_patch16_224 | lora | 5x10 | task_aware | 5 | train=50, test=20 | 0.83 | 0.011 |
| `run-imagenet-r-vit-lora-iewc-fromp-l100-seed2-5x10-50shot-e5-taskaware.json` | IEWC-FROMP | 2 | 100 | imagenet_r | vit_tiny_patch16_224 | lora | 5x10 | task_aware | 5 | train=50, test=20 | 0.8156 | 0.0244 |
| `run-imagenet-r-vit-lora-iewc-fromp-l30-seed0-5x10-50shot-e5-taskaware.json` | IEWC-FROMP | 0 | 30 | imagenet_r | vit_tiny_patch16_224 | lora | 5x10 | task_aware | 5 | train=50, test=20 | 0.8043 | 0.0249 |
| `run-imagenet-r-vit-lora-iewc-fromp-l300-seed0-5x10-50shot-e5-taskaware.json` | IEWC-FROMP | 0 | 300 | imagenet_r | vit_tiny_patch16_224 | lora | 5x10 | task_aware | 5 | train=50, test=20 | 0.8102 | 0.015 |
| `run-imagenet-r-vit-lora-iewc-gss-l10-seed0-5x10-50shot-e5-taskaware.json` | IEWC-GSS | 0 | 10 | imagenet_r | vit_tiny_patch16_224 | lora | 5x10 | task_aware | 5 | train=50, test=20 | 0.8224 | 0.0238 |
| `run-imagenet-r-vit-lora-iewc-gss-l10-seed1-5x10-50shot-e5-taskaware.json` | IEWC-GSS | 1 | 10 | imagenet_r | vit_tiny_patch16_224 | lora | 5x10 | task_aware | 5 | train=50, test=20 | 0.8154 | 0.0403 |
| `run-imagenet-r-vit-lora-iewc-gss-l10-seed2-5x10-50shot-e5-taskaware.json` | IEWC-GSS | 2 | 10 | imagenet_r | vit_tiny_patch16_224 | lora | 5x10 | task_aware | 5 | train=50, test=20 | 0.806 | 0.0485 |
| `run-imagenet-r-vit-lora-iewc-gss-l100-seed0-5x10-50shot-e5-taskaware.json` | IEWC-GSS | 0 | 100 | imagenet_r | vit_tiny_patch16_224 | lora | 5x10 | task_aware | 5 | train=50, test=20 | 0.8093 | 0.0136 |
| `run-imagenet-r-vit-lora-iewc-gss-l30-seed0-5x10-50shot-e5-taskaware.json` | IEWC-GSS | 0 | 30 | imagenet_r | vit_tiny_patch16_224 | lora | 5x10 | task_aware | 5 | train=50, test=20 | 0.812 | 0.0137 |
| `run-imagenet-r-vit-lora-iewc-gss-l300-seed0-5x10-50shot-e5-taskaware.json` | IEWC-GSS | 0 | 300 | imagenet_r | vit_tiny_patch16_224 | lora | 5x10 | task_aware | 5 | train=50, test=20 | 0.8057 | 0.0179 |
| `run-imagenet-r-vit-lora-iewc-l100-seed0-5x10-50shot-e5-taskaware.json` | IEWC | 0 | 100 | imagenet_r | vit_tiny_patch16_224 | lora | 5x10 | task_aware | 5 | train=50, test=20 | 0.8235 | 0.0193 |
| `run-imagenet-r-vit-lora-iewc-l100-seed1-5x10-50shot-e5-taskaware.json` | IEWC | 1 | 100 | imagenet_r | vit_tiny_patch16_224 | lora | 5x10 | task_aware | 5 | train=50, test=20 | 0.8176 | 0.0415 |
| `run-imagenet-r-vit-lora-iewc-l100-seed2-5x10-50shot-e5-taskaware.json` | IEWC | 2 | 100 | imagenet_r | vit_tiny_patch16_224 | lora | 5x10 | task_aware | 5 | train=50, test=20 | 0.8034 | 0.042 |
| `run-imagenet-r-vit-lora-iewc-l30-seed0-5x10-50shot-e5-taskaware.json` | IEWC | 0 | 30 | imagenet_r | vit_tiny_patch16_224 | lora | 5x10 | task_aware | 5 | train=50, test=20 | 0.8022 | 0.0513 |
| `run-imagenet-r-vit-lora-iewc-l300-seed0-5x10-50shot-e5-taskaware.json` | IEWC | 0 | 300 | imagenet_r | vit_tiny_patch16_224 | lora | 5x10 | task_aware | 5 | train=50, test=20 | 0.8224 | 0.0068 |
| `run-imagenet-r-vit-lora-sequential-seed0-5x10-50shot-e5-taskaware.json` | Sequential | 0 | - | imagenet_r | vit_tiny_patch16_224 | lora | 5x10 | task_aware | 5 | train=50, test=20 | 0.6961 | 0.1747 |
| `run-imagenet-r-vit-lora-sequential-seed1-5x10-50shot-e5-taskaware.json` | Sequential | 1 | - | imagenet_r | vit_tiny_patch16_224 | lora | 5x10 | task_aware | 5 | train=50, test=20 | 0.6738 | 0.2058 |
| `run-imagenet-r-vit-lora-sequential-seed2-5x10-50shot-e5-taskaware.json` | Sequential | 2 | - | imagenet_r | vit_tiny_patch16_224 | lora | 5x10 | task_aware | 5 | train=50, test=20 | 0.7022 | 0.1586 |
| `smoke-imagenet-r-vitb-lora-sequential-2x20-full-e1-classinc.json` | Sequential | 0 | - | imagenet_r | vit_base_patch16_224 | lora | 2x20 | class_incremental | 1 | train=0, test=0 | 0.4921 | 0.6501 |
| `smoke-vision-lora-gss.json` | IEWC-GSS | 0 | 100 | cifar100 | vit_tiny_patch16_224 | lora | 2x2 | - | 1 | train=2, test=2 | 0.125 | 0 |
| `smoke-vision-sequential.json` | Sequential | 0 | - | cifar100 | vit_tiny_patch16_224 | full | 2x2 | - | 1 | train=2, test=2 | 0.25 | 0 |
| `vision-smoke-vit-tune-ef-seed0-lam1.json` | EF-EWC | 0 | 1 | imagenet_r | vit_tiny_patch16_224 | lora | 2x3 | task_aware | 1 | train=4, test=4 | 0.375 | 0 |
| `vision-smoke-vit-tune-ef-seed0-lam3.json` | EF-EWC | 0 | 3 | imagenet_r | vit_tiny_patch16_224 | lora | 2x3 | task_aware | 1 | train=4, test=4 | 0.375 | 0 |
| `vision-smoke-vit-tune-iewc_gss-seed0-lam1.json` | IEWC-GSS | 0 | 1 | imagenet_r | vit_tiny_patch16_224 | lora | 2x3 | task_aware | 1 | train=4, test=4 | 0.375 | 0 |
| `vision-smoke-vit-tune-iewc_gss-seed0-lam3.json` | IEWC-GSS | 0 | 3 | imagenet_r | vit_tiny_patch16_224 | lora | 2x3 | task_aware | 1 | train=4, test=4 | 0.375 | 0 |
| `vision-smoke-vit-tune-sequential-seed0.json` | Sequential | 0 | - | imagenet_r | vit_tiny_patch16_224 | lora | 2x3 | task_aware | 1 | train=4, test=4 | 0.375 | 0 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-ef-seed0-lam1000.json` | EF-EWC | 0 | 1000 | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.5946 | 0.3526 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-ef-seed0-lam111p111.json` | EF-EWC | 0 | 111.1111 | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.5692 | 0.3961 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-ef-seed0-lam3000.json` | EF-EWC | 0 | 3000 | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.4582 | 0.4978 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-ef-seed0-lam333p333.json` | EF-EWC | 0 | 333.3333 | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.5987 | 0.3563 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-ef-seed0-lam9000.json` | EF-EWC | 0 | 9000 | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.4445 | 0.4951 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-ef-seed1-lam333p333.json` | EF-EWC | 1 | 333.3333 | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.6315 | 0.3234 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-ef-seed2-lam333p333.json` | EF-EWC | 2 | 333.3333 | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.5642 | 0.3987 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-ewc_dr-seed0-lam1.json` | EWC-DR | 0 | 1 | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.453 | 0.5272 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-ewc_dr-seed0-lam10.json` | EWC-DR | 0 | 10 | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.4858 | 0.4779 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-ewc_dr-seed0-lam3.json` | EWC-DR | 0 | 3 | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.5896 | 0.3689 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-ewc_dr-seed0-lam30.json` | EWC-DR | 0 | 30 | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.3317 | 0.6237 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-ewc_dr-seed1-lam3.json` | EWC-DR | 1 | 3 | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.5774 | 0.386 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-ewc_dr-seed2-lam3.json` | EWC-DR | 2 | 3 | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.5392 | 0.4248 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-iewc-seed0-lam100.json` | IEWC | 0 | 100 | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.5602 | 0.3958 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-iewc-seed0-lam1000.json` | IEWC | 0 | 1000 | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.504 | 0.434 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-iewc-seed0-lam11p1111.json` | IEWC | 0 | 11.1111 | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.5022 | 0.4726 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-iewc-seed0-lam300.json` | IEWC | 0 | 300 | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.5212 | 0.4234 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-iewc-seed0-lam3000.json` | IEWC | 0 | 3000 | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.4406 | 0.4928 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-iewc-seed0-lam33p3333.json` | IEWC | 0 | 33.3333 | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.6146 | 0.3435 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-iewc-seed1-lam33p3333.json` | IEWC | 1 | 33.3333 | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.6114 | 0.3472 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-iewc-seed2-lam33p3333.json` | IEWC | 2 | 33.3333 | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.5621 | 0.4044 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-iewc_fromp-seed0-lam100.json` | IEWC-FROMP | 0 | 100 | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.605 | 0.3251 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-iewc_fromp-seed0-lam1000.json` | IEWC-FROMP | 0 | 1000 | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.4709 | 0.4583 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-iewc_fromp-seed0-lam11p1111.json` | IEWC-FROMP | 0 | 11.1111 | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.6304 | 0.3238 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-iewc_fromp-seed0-lam300.json` | IEWC-FROMP | 0 | 300 | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.4299 | 0.5116 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-iewc_fromp-seed0-lam33p3333.json` | IEWC-FROMP | 0 | 33.3333 | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.6363 | 0.3066 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-iewc_fromp-seed1-lam33p3333.json` | IEWC-FROMP | 1 | 33.3333 | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.653 | 0.2899 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-iewc_fromp-seed2-lam33p3333.json` | IEWC-FROMP | 2 | 33.3333 | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.6201 | 0.3238 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-iewc_gss-seed0-lam100.json` | IEWC-GSS | 0 | 100 | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.5427 | 0.395 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-iewc_gss-seed0-lam1000.json` | IEWC-GSS | 0 | 1000 | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.3849 | 0.5529 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-iewc_gss-seed0-lam11p1111.json` | IEWC-GSS | 0 | 11.1111 | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.6305 | 0.322 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-iewc_gss-seed0-lam300.json` | IEWC-GSS | 0 | 300 | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.4695 | 0.4679 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-iewc_gss-seed0-lam33p3333.json` | IEWC-GSS | 0 | 33.3333 | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.5826 | 0.363 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-iewc_gss-seed0-lam3p7037.json` | IEWC-GSS | 0 | 3.7037 | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.5938 | 0.3642 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-iewc_gss-seed1-lam11p1111.json` | IEWC-GSS | 1 | 11.1111 | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.6481 | 0.3045 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-iewc_gss-seed2-lam11p1111.json` | IEWC-GSS | 2 | 11.1111 | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.5964 | 0.3589 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-sequential-seed0.json` | Sequential | 0 | - | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.2805 | 0.7172 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-sequential-seed1.json` | Sequential | 1 | - | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.2543 | 0.747 |
| `vision-vitb-imagenetr-10x20-full-e2-classinc-sequential-seed2.json` | Sequential | 2 | - | imagenet_r | vit_base_patch16_224 | lora | 10x20 | class_incremental | 2 | train=0, test=0 | 0.3339 | 0.6556 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-ef-seed0-lam1.json` | EF-EWC | 0 | 1 | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.2947 | 0.7829 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-ef-seed0-lam10.json` | EF-EWC | 0 | 10 | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.317 | 0.7551 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-ef-seed0-lam100.json` | EF-EWC | 0 | 100 | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.5006 | 0.521 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-ef-seed0-lam1000.json` | EF-EWC | 0 | 1000 | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.6727 | 0.2947 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-ef-seed0-lam3.json` | EF-EWC | 0 | 3 | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.2919 | 0.7859 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-ef-seed0-lam30.json` | EF-EWC | 0 | 30 | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.3559 | 0.7062 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-ef-seed0-lam300.json` | EF-EWC | 0 | 300 | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.5969 | 0.4009 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-ef-seed0-lam3000.json` | EF-EWC | 0 | 3000 | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.6941 | 0.2592 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-ef-seed0-lam9000.json` | EF-EWC | 0 | 9000 | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.6923 | 0.252 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-ewc_dr-seed0-lam1.json` | EWC-DR | 0 | 1 | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.3893 | 0.6645 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-ewc_dr-seed0-lam10.json` | EWC-DR | 0 | 10 | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.6465 | 0.3314 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-ewc_dr-seed0-lam100.json` | EWC-DR | 0 | 100 | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.5111 | 0.4694 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-ewc_dr-seed0-lam1000.json` | EWC-DR | 0 | 1000 | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.3135 | 0.5735 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-ewc_dr-seed0-lam3.json` | EWC-DR | 0 | 3 | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.5698 | 0.4356 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-ewc_dr-seed0-lam30.json` | EWC-DR | 0 | 30 | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.6275 | 0.3437 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-ewc_dr-seed0-lam300.json` | EWC-DR | 0 | 300 | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.4045 | 0.5577 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-iewc-seed0-lam1.json` | IEWC | 0 | 1 | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.3044 | 0.7695 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-iewc-seed0-lam10.json` | IEWC | 0 | 10 | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.3926 | 0.6592 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-iewc-seed0-lam100.json` | IEWC | 0 | 100 | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.6353 | 0.3493 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-iewc-seed0-lam1000.json` | IEWC | 0 | 1000 | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.7337 | 0.2029 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-iewc-seed0-lam3.json` | IEWC | 0 | 3 | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.328 | 0.7425 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-iewc-seed0-lam30.json` | IEWC | 0 | 30 | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.5729 | 0.4304 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-iewc-seed0-lam300.json` | IEWC | 0 | 300 | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.7013 | 0.2571 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-iewc-seed0-lam3000.json` | IEWC | 0 | 3000 | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.6986 | 0.2396 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-iewc_fromp-seed0-lam1.json` | IEWC-FROMP | 0 | 1 | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.3319 | 0.737 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-iewc_fromp-seed0-lam10.json` | IEWC-FROMP | 0 | 10 | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.5991 | 0.3958 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-iewc_fromp-seed0-lam100.json` | IEWC-FROMP | 0 | 100 | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.7047 | 0.2472 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-iewc_fromp-seed0-lam1000.json` | IEWC-FROMP | 0 | 1000 | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.6624 | 0.283 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-iewc_fromp-seed0-lam3.json` | IEWC-FROMP | 0 | 3 | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.4171 | 0.6306 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-iewc_fromp-seed0-lam30.json` | IEWC-FROMP | 0 | 30 | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.6517 | 0.3303 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-iewc_fromp-seed0-lam300.json` | IEWC-FROMP | 0 | 300 | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.7262 | 0.2111 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-iewc_gss-seed0-lam1.json` | IEWC-GSS | 0 | 1 | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.3406 | 0.7257 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-iewc_gss-seed0-lam10.json` | IEWC-GSS | 0 | 10 | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.5987 | 0.3958 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-iewc_gss-seed0-lam100.json` | IEWC-GSS | 0 | 100 | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.7083 | 0.2402 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-iewc_gss-seed0-lam1000.json` | IEWC-GSS | 0 | 1000 | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.6758 | 0.2658 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-iewc_gss-seed0-lam3.json` | IEWC-GSS | 0 | 3 | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.4383 | 0.6036 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-iewc_gss-seed0-lam30.json` | IEWC-GSS | 0 | 30 | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.6599 | 0.3156 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-iewc_gss-seed0-lam300.json` | IEWC-GSS | 0 | 300 | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.7254 | 0.2088 |
| `vision-vitb-imagenetr-5x20-full-e2-classinc-sequential-seed0.json` | Sequential | 0 | - | imagenet_r | vit_base_patch16_224 | lora | 5x20 | class_incremental | 2 | train=0, test=0 | 0.292 | 0.787 |

## Time-Series Forecasting / Regression

Total runs: `91`.

| artifact | method | seed | lambda | dataset | distributions | model | context->horizon | windows | epochs | normalization | final avg MSE | forgetting MSE |
| --- | --- | ---: | ---: | --- | --- | --- | --- | --- | ---: | --- | ---: | ---: |
| `forecasting-ett-4task-transformer-d128-e10-lr3e4-sequential-seed0.json` | Sequential | 0 | - | ett | ETTh1,ETTh2,ETTm1,ETTm2 | TransformerEncoder d=128, L=3, H=4, ff=256 | 96->24 | train=512, eval=128 | 10 | - | 2.3022 | 0.7694 |
| `forecasting-ett-4task-transformer-d128-e10-lr3e4-series-ef-seed0-lam0p00037037.json` | EF-EWC | 0 | 0.00037 | ett | ETTh1,ETTh2,ETTm1,ETTm2 | TransformerEncoder d=128, L=3, H=4, ff=256 | 96->24 | train=512, eval=128 | 10 | series | 0.1952 | 0.0239 |
| `forecasting-ett-4task-transformer-d128-e10-lr3e4-series-ef-seed0-lam0p00111111.json` | EF-EWC | 0 | 0.0011 | ett | ETTh1,ETTh2,ETTm1,ETTm2 | TransformerEncoder d=128, L=3, H=4, ff=256 | 96->24 | train=512, eval=128 | 10 | series | 0.1952 | 0.0239 |
| `forecasting-ett-4task-transformer-d128-e10-lr3e4-series-ef-seed0-lam0p00333333.json` | EF-EWC | 0 | 0.0033 | ett | ETTh1,ETTh2,ETTm1,ETTm2 | TransformerEncoder d=128, L=3, H=4, ff=256 | 96->24 | train=512, eval=128 | 10 | series | 0.1952 | 0.0239 |
| `forecasting-ett-4task-transformer-d128-e10-lr3e4-series-ef-seed0-lam0p01.json` | EF-EWC | 0 | 0.01 | ett | ETTh1,ETTh2,ETTm1,ETTm2 | TransformerEncoder d=128, L=3, H=4, ff=256 | 96->24 | train=512, eval=128 | 10 | series | 0.1952 | 0.0239 |
| `forecasting-ett-4task-transformer-d128-e10-lr3e4-series-ef-seed0-lam0p03.json` | EF-EWC | 0 | 0.03 | ett | ETTh1,ETTh2,ETTm1,ETTm2 | TransformerEncoder d=128, L=3, H=4, ff=256 | 96->24 | train=512, eval=128 | 10 | series | 0.1952 | 0.0239 |
| `forecasting-ett-4task-transformer-d128-e10-lr3e4-series-ef-seed0-lam0p1.json` | EF-EWC | 0 | 0.1 | ett | ETTh1,ETTh2,ETTm1,ETTm2 | TransformerEncoder d=128, L=3, H=4, ff=256 | 96->24 | train=512, eval=128 | 10 | series | 0.1952 | 0.0239 |
| `forecasting-ett-4task-transformer-d128-e10-lr3e4-series-ef-seed0-lam0p3.json` | EF-EWC | 0 | 0.3 | ett | ETTh1,ETTh2,ETTm1,ETTm2 | TransformerEncoder d=128, L=3, H=4, ff=256 | 96->24 | train=512, eval=128 | 10 | series | 0.1952 | 0.0239 |
| `forecasting-ett-4task-transformer-d128-e10-lr3e4-series-ef-seed0-lam1.json` | EF-EWC | 0 | 1 | ett | ETTh1,ETTh2,ETTm1,ETTm2 | TransformerEncoder d=128, L=3, H=4, ff=256 | 96->24 | train=512, eval=128 | 10 | series | 0.1952 | 0.0239 |
| `forecasting-ett-4task-transformer-d128-e10-lr3e4-series-ef-seed0-lam3.json` | EF-EWC | 0 | 3 | ett | ETTh1,ETTh2,ETTm1,ETTm2 | TransformerEncoder d=128, L=3, H=4, ff=256 | 96->24 | train=512, eval=128 | 10 | series | 0.1952 | 0.0239 |
| `forecasting-ett-4task-transformer-d128-e10-lr3e4-series-ef-seed1-lam0p00037037.json` | EF-EWC | 1 | 0.00037 | ett | ETTh1,ETTh2,ETTm1,ETTm2 | TransformerEncoder d=128, L=3, H=4, ff=256 | 96->24 | train=512, eval=128 | 10 | series | 0.1953 | 0.0253 |
| `forecasting-ett-4task-transformer-d128-e10-lr3e4-series-ef-seed2-lam0p00037037.json` | EF-EWC | 2 | 0.00037 | ett | ETTh1,ETTh2,ETTm1,ETTm2 | TransformerEncoder d=128, L=3, H=4, ff=256 | 96->24 | train=512, eval=128 | 10 | series | 0.205 | 0.0312 |
| `forecasting-ett-4task-transformer-d128-e10-lr3e4-series-iewc-seed0-lam0p01.json` | IEWC | 0 | 0.01 | ett | ETTh1,ETTh2,ETTm1,ETTm2 | TransformerEncoder d=128, L=3, H=4, ff=256 | 96->24 | train=512, eval=128 | 10 | series | 0.1952 | 0.0239 |
| `forecasting-ett-4task-transformer-d128-e10-lr3e4-series-iewc-seed0-lam0p03.json` | IEWC | 0 | 0.03 | ett | ETTh1,ETTh2,ETTm1,ETTm2 | TransformerEncoder d=128, L=3, H=4, ff=256 | 96->24 | train=512, eval=128 | 10 | series | 0.1952 | 0.0239 |
| `forecasting-ett-4task-transformer-d128-e10-lr3e4-series-iewc-seed0-lam0p1.json` | IEWC | 0 | 0.1 | ett | ETTh1,ETTh2,ETTm1,ETTm2 | TransformerEncoder d=128, L=3, H=4, ff=256 | 96->24 | train=512, eval=128 | 10 | series | 0.1952 | 0.0239 |
| `forecasting-ett-4task-transformer-d128-e10-lr3e4-series-iewc-seed0-lam0p3.json` | IEWC | 0 | 0.3 | ett | ETTh1,ETTh2,ETTm1,ETTm2 | TransformerEncoder d=128, L=3, H=4, ff=256 | 96->24 | train=512, eval=128 | 10 | series | 0.1953 | 0.024 |
| `forecasting-ett-4task-transformer-d128-e10-lr3e4-series-iewc-seed0-lam1.json` | IEWC | 0 | 1 | ett | ETTh1,ETTh2,ETTm1,ETTm2 | TransformerEncoder d=128, L=3, H=4, ff=256 | 96->24 | train=512, eval=128 | 10 | series | 0.1954 | 0.0242 |
| `forecasting-ett-4task-transformer-d128-e10-lr3e4-series-iewc-seed0-lam27.json` | IEWC | 0 | 27 | ett | ETTh1,ETTh2,ETTm1,ETTm2 | TransformerEncoder d=128, L=3, H=4, ff=256 | 96->24 | train=512, eval=128 | 10 | series | 0.1897 | 0.0174 |
| `forecasting-ett-4task-transformer-d128-e10-lr3e4-series-iewc-seed0-lam3.json` | IEWC | 0 | 3 | ett | ETTh1,ETTh2,ETTm1,ETTm2 | TransformerEncoder d=128, L=3, H=4, ff=256 | 96->24 | train=512, eval=128 | 10 | series | 0.1944 | 0.0233 |
| `forecasting-ett-4task-transformer-d128-e10-lr3e4-series-iewc-seed0-lam81.json` | IEWC | 0 | 81 | ett | ETTh1,ETTh2,ETTm1,ETTm2 | TransformerEncoder d=128, L=3, H=4, ff=256 | 96->24 | train=512, eval=128 | 10 | series | 0.1892 | 0.0144 |
| `forecasting-ett-4task-transformer-d128-e10-lr3e4-series-iewc-seed0-lam9.json` | IEWC | 0 | 9 | ett | ETTh1,ETTh2,ETTm1,ETTm2 | TransformerEncoder d=128, L=3, H=4, ff=256 | 96->24 | train=512, eval=128 | 10 | series | 0.1919 | 0.0206 |
| `forecasting-ett-4task-transformer-d128-e10-lr3e4-series-iewc-seed1-lam81.json` | IEWC | 1 | 81 | ett | ETTh1,ETTh2,ETTm1,ETTm2 | TransformerEncoder d=128, L=3, H=4, ff=256 | 96->24 | train=512, eval=128 | 10 | series | 0.1887 | 0.0123 |
| `forecasting-ett-4task-transformer-d128-e10-lr3e4-series-iewc-seed2-lam81.json` | IEWC | 2 | 81 | ett | ETTh1,ETTh2,ETTm1,ETTm2 | TransformerEncoder d=128, L=3, H=4, ff=256 | 96->24 | train=512, eval=128 | 10 | series | 0.199 | 0.0209 |
| `forecasting-ett-4task-transformer-d128-e10-lr3e4-series-iewc_gss-seed0-lam0p01.json` | IEWC-GSS | 0 | 0.01 | ett | ETTh1,ETTh2,ETTm1,ETTm2 | TransformerEncoder d=128, L=3, H=4, ff=256 | 96->24 | train=512, eval=128 | 10 | series | 0.1952 | 0.0239 |
| `forecasting-ett-4task-transformer-d128-e10-lr3e4-series-iewc_gss-seed0-lam0p03.json` | IEWC-GSS | 0 | 0.03 | ett | ETTh1,ETTh2,ETTm1,ETTm2 | TransformerEncoder d=128, L=3, H=4, ff=256 | 96->24 | train=512, eval=128 | 10 | series | 0.1952 | 0.0239 |
| `forecasting-ett-4task-transformer-d128-e10-lr3e4-series-iewc_gss-seed0-lam0p1.json` | IEWC-GSS | 0 | 0.1 | ett | ETTh1,ETTh2,ETTm1,ETTm2 | TransformerEncoder d=128, L=3, H=4, ff=256 | 96->24 | train=512, eval=128 | 10 | series | 0.1953 | 0.024 |
| `forecasting-ett-4task-transformer-d128-e10-lr3e4-series-iewc_gss-seed0-lam0p3.json` | IEWC-GSS | 0 | 0.3 | ett | ETTh1,ETTh2,ETTm1,ETTm2 | TransformerEncoder d=128, L=3, H=4, ff=256 | 96->24 | train=512, eval=128 | 10 | series | 0.1954 | 0.0241 |
| `forecasting-ett-4task-transformer-d128-e10-lr3e4-series-iewc_gss-seed0-lam1.json` | IEWC-GSS | 0 | 1 | ett | ETTh1,ETTh2,ETTm1,ETTm2 | TransformerEncoder d=128, L=3, H=4, ff=256 | 96->24 | train=512, eval=128 | 10 | series | 0.1953 | 0.0242 |
| `forecasting-ett-4task-transformer-d128-e10-lr3e4-series-iewc_gss-seed0-lam27.json` | IEWC-GSS | 0 | 27 | ett | ETTh1,ETTh2,ETTm1,ETTm2 | TransformerEncoder d=128, L=3, H=4, ff=256 | 96->24 | train=512, eval=128 | 10 | series | 0.1891 | 0.0163 |
| `forecasting-ett-4task-transformer-d128-e10-lr3e4-series-iewc_gss-seed0-lam3.json` | IEWC-GSS | 0 | 3 | ett | ETTh1,ETTh2,ETTm1,ETTm2 | TransformerEncoder d=128, L=3, H=4, ff=256 | 96->24 | train=512, eval=128 | 10 | series | 0.1936 | 0.0226 |
| `forecasting-ett-4task-transformer-d128-e10-lr3e4-series-iewc_gss-seed0-lam81.json` | IEWC-GSS | 0 | 81 | ett | ETTh1,ETTh2,ETTm1,ETTm2 | TransformerEncoder d=128, L=3, H=4, ff=256 | 96->24 | train=512, eval=128 | 10 | series | 0.1888 | 0.0132 |
| `forecasting-ett-4task-transformer-d128-e10-lr3e4-series-iewc_gss-seed0-lam9.json` | IEWC-GSS | 0 | 9 | ett | ETTh1,ETTh2,ETTm1,ETTm2 | TransformerEncoder d=128, L=3, H=4, ff=256 | 96->24 | train=512, eval=128 | 10 | series | 0.191 | 0.0196 |
| `forecasting-ett-4task-transformer-d128-e10-lr3e4-series-iewc_gss-seed1-lam81.json` | IEWC-GSS | 1 | 81 | ett | ETTh1,ETTh2,ETTm1,ETTm2 | TransformerEncoder d=128, L=3, H=4, ff=256 | 96->24 | train=512, eval=128 | 10 | series | 0.1888 | 0.0121 |
| `forecasting-ett-4task-transformer-d128-e10-lr3e4-series-iewc_gss-seed2-lam81.json` | IEWC-GSS | 2 | 81 | ett | ETTh1,ETTh2,ETTm1,ETTm2 | TransformerEncoder d=128, L=3, H=4, ff=256 | 96->24 | train=512, eval=128 | 10 | series | 0.1987 | 0.0207 |
| `forecasting-ett-4task-transformer-d128-e10-lr3e4-series-sequential-seed0.json` | Sequential | 0 | - | ett | ETTh1,ETTh2,ETTm1,ETTm2 | TransformerEncoder d=128, L=3, H=4, ff=256 | 96->24 | train=512, eval=128 | 10 | series | 0.2186 | 0.0416 |
| `forecasting-ett-4task-transformer-d128-e10-lr3e4-series-sequential-seed1.json` | Sequential | 1 | - | ett | ETTh1,ETTh2,ETTm1,ETTm2 | TransformerEncoder d=128, L=3, H=4, ff=256 | 96->24 | train=512, eval=128 | 10 | series | 0.1948 | 0.0217 |
| `forecasting-ett-4task-transformer-d128-e10-lr3e4-series-sequential-seed2.json` | Sequential | 2 | - | ett | ETTh1,ETTh2,ETTm1,ETTm2 | TransformerEncoder d=128, L=3, H=4, ff=256 | 96->24 | train=512, eval=128 | 10 | series | 0.2145 | 0.0389 |
| `forecasting-ett-4task-transformer-d128-e6-sequential-seed0.json` | Sequential | 0 | - | ett | ETTh1,ETTh2,ETTm1,ETTm2 | TransformerEncoder d=128, L=3, H=4, ff=256 | 96->24 | train=512, eval=128 | 6 | - | 2.4618 | 0.8302 |
| `forecasting-longhorizon-patchtst-3task-guard-ef-seed0-lam0p0001.json` | EF-EWC | 0 | 0.0001 | long_horizon | Weather,ECL,Traffic | PatchTST d=128, L=3, H=8, ff=256, p=16, s=8 | 336->96 | train=8, eval=4 | 2 | series | 0.536 | 0.028 |
| `forecasting-longhorizon-patchtst-3task-guard-ef-seed0-lam0p001.json` | EF-EWC | 0 | 0.001 | long_horizon | Weather,ECL,Traffic | PatchTST d=128, L=3, H=8, ff=256, p=16, s=8 | 336->96 | train=8, eval=4 | 2 | series | 0.536 | 0.028 |
| `forecasting-longhorizon-patchtst-3task-guard-iewc-seed0-lam10.json` | IEWC | 0 | 10 | long_horizon | Weather,ECL,Traffic | PatchTST d=128, L=3, H=8, ff=256, p=16, s=8 | 336->96 | train=8, eval=4 | 2 | series | 0.532 | 0.026 |
| `forecasting-longhorizon-patchtst-3task-guard-iewc-seed0-lam100.json` | IEWC | 0 | 100 | long_horizon | Weather,ECL,Traffic | PatchTST d=128, L=3, H=8, ff=256, p=16, s=8 | 336->96 | train=8, eval=4 | 2 | series | 0.505 | 0.0125 |
| `forecasting-longhorizon-patchtst-3task-guard-sequential-seed0.json` | Sequential | 0 | - | long_horizon | Weather,ECL,Traffic | PatchTST d=128, L=3, H=8, ff=256, p=16, s=8 | 336->96 | train=8, eval=4 | 2 | series | 0.5404 | 0.0296 |
| `forecasting-longhorizon-patchtst-3task-scaled-ef-seed0-lam0p0001.json` | EF-EWC | 0 | 0.0001 | long_horizon | Weather,ECL,Traffic | PatchTST d=256, L=4, H=8, ff=512, p=16, s=8 | 336->96 | train=16, eval=8 | 8 | series | 0.3244 | 0.0132 |
| `forecasting-longhorizon-patchtst-3task-scaled-ef-seed0-lam0p001.json` | EF-EWC | 0 | 0.001 | long_horizon | Weather,ECL,Traffic | PatchTST d=256, L=4, H=8, ff=512, p=16, s=8 | 336->96 | train=16, eval=8 | 8 | series | 0.3244 | 0.0132 |
| `forecasting-longhorizon-patchtst-3task-scaled-ef-seed0-lam0p01.json` | EF-EWC | 0 | 0.01 | long_horizon | Weather,ECL,Traffic | PatchTST d=256, L=4, H=8, ff=512, p=16, s=8 | 336->96 | train=16, eval=8 | 8 | series | 0.3244 | 0.0132 |
| `forecasting-longhorizon-patchtst-3task-scaled-ef-seed0-lam0p03.json` | EF-EWC | 0 | 0.03 | long_horizon | Weather,ECL,Traffic | PatchTST d=256, L=4, H=8, ff=512, p=16, s=8 | 336->96 | train=16, eval=8 | 8 | series | 0.3244 | 0.0132 |
| `forecasting-longhorizon-patchtst-3task-scaled-ef-seed0-lam0p09.json` | EF-EWC | 0 | 0.09 | long_horizon | Weather,ECL,Traffic | PatchTST d=256, L=4, H=8, ff=512, p=16, s=8 | 336->96 | train=16, eval=8 | 8 | series | 0.3244 | 0.0132 |
| `forecasting-longhorizon-patchtst-3task-scaled-ef-seed0-lam0p27.json` | EF-EWC | 0 | 0.27 | long_horizon | Weather,ECL,Traffic | PatchTST d=256, L=4, H=8, ff=512, p=16, s=8 | 336->96 | train=16, eval=8 | 8 | series | 0.3244 | 0.0132 |
| `forecasting-longhorizon-patchtst-3task-scaled-ef-seed0-lam1em05.json` | EF-EWC | 0 | 1e-05 | long_horizon | Weather,ECL,Traffic | PatchTST d=256, L=4, H=8, ff=512, p=16, s=8 | 336->96 | train=16, eval=8 | 8 | series | 0.3244 | 0.0132 |
| `forecasting-longhorizon-patchtst-3task-scaled-ef-seed1-lam0p27.json` | EF-EWC | 1 | 0.27 | long_horizon | Weather,ECL,Traffic | PatchTST d=256, L=4, H=8, ff=512, p=16, s=8 | 336->96 | train=16, eval=8 | 8 | series | 0.3345 | 0.0323 |
| `forecasting-longhorizon-patchtst-3task-scaled-ef-seed2-lam0p27.json` | EF-EWC | 2 | 0.27 | long_horizon | Weather,ECL,Traffic | PatchTST d=256, L=4, H=8, ff=512, p=16, s=8 | 336->96 | train=16, eval=8 | 8 | series | 0.3442 | 0.0383 |
| `forecasting-longhorizon-patchtst-3task-scaled-iewc-seed0-lam10.json` | IEWC | 0 | 10 | long_horizon | Weather,ECL,Traffic | PatchTST d=256, L=4, H=8, ff=512, p=16, s=8 | 336->96 | train=16, eval=8 | 8 | series | 0.3251 | 0.0109 |
| `forecasting-longhorizon-patchtst-3task-scaled-iewc-seed0-lam100.json` | IEWC | 0 | 100 | long_horizon | Weather,ECL,Traffic | PatchTST d=256, L=4, H=8, ff=512, p=16, s=8 | 336->96 | train=16, eval=8 | 8 | series | 0.3351 | 0.0137 |
| `forecasting-longhorizon-patchtst-3task-scaled-iewc-seed0-lam1p11111.json` | IEWC | 0 | 1.1111 | long_horizon | Weather,ECL,Traffic | PatchTST d=256, L=4, H=8, ff=512, p=16, s=8 | 336->96 | train=16, eval=8 | 8 | series | 0.3247 | 0.0126 |
| `forecasting-longhorizon-patchtst-3task-scaled-iewc-seed0-lam30.json` | IEWC | 0 | 30 | long_horizon | Weather,ECL,Traffic | PatchTST d=256, L=4, H=8, ff=512, p=16, s=8 | 336->96 | train=16, eval=8 | 8 | series | 0.3279 | 0.0102 |
| `forecasting-longhorizon-patchtst-3task-scaled-iewc-seed0-lam300.json` | IEWC | 0 | 300 | long_horizon | Weather,ECL,Traffic | PatchTST d=256, L=4, H=8, ff=512, p=16, s=8 | 336->96 | train=16, eval=8 | 8 | series | 0.3388 | 0.0105 |
| `forecasting-longhorizon-patchtst-3task-scaled-iewc-seed0-lam3p33333.json` | IEWC | 0 | 3.3333 | long_horizon | Weather,ECL,Traffic | PatchTST d=256, L=4, H=8, ff=512, p=16, s=8 | 336->96 | train=16, eval=8 | 8 | series | 0.3251 | 0.0121 |
| `forecasting-longhorizon-patchtst-3task-scaled-iewc-seed0-lam900.json` | IEWC | 0 | 900 | long_horizon | Weather,ECL,Traffic | PatchTST d=256, L=4, H=8, ff=512, p=16, s=8 | 336->96 | train=16, eval=8 | 8 | series | 0.3485 | 0.0066 |
| `forecasting-longhorizon-patchtst-3task-scaled-iewc-seed1-lam1p11111.json` | IEWC | 1 | 1.1111 | long_horizon | Weather,ECL,Traffic | PatchTST d=256, L=4, H=8, ff=512, p=16, s=8 | 336->96 | train=16, eval=8 | 8 | series | 0.3341 | 0.0317 |
| `forecasting-longhorizon-patchtst-3task-scaled-iewc-seed2-lam1p11111.json` | IEWC | 2 | 1.1111 | long_horizon | Weather,ECL,Traffic | PatchTST d=256, L=4, H=8, ff=512, p=16, s=8 | 336->96 | train=16, eval=8 | 8 | series | 0.3428 | 0.0358 |
| `forecasting-longhorizon-patchtst-3task-scaled-iewc_gss-seed0-lam10.json` | IEWC-GSS | 0 | 10 | long_horizon | Weather,ECL,Traffic | PatchTST d=256, L=4, H=8, ff=512, p=16, s=8 | 336->96 | train=16, eval=8 | 8 | series | 0.3251 | 0.0106 |
| `forecasting-longhorizon-patchtst-3task-scaled-iewc_gss-seed0-lam100.json` | IEWC-GSS | 0 | 100 | long_horizon | Weather,ECL,Traffic | PatchTST d=256, L=4, H=8, ff=512, p=16, s=8 | 336->96 | train=16, eval=8 | 8 | series | 0.3353 | 0.0129 |
| `forecasting-longhorizon-patchtst-3task-scaled-iewc_gss-seed0-lam1p11111.json` | IEWC-GSS | 0 | 1.1111 | long_horizon | Weather,ECL,Traffic | PatchTST d=256, L=4, H=8, ff=512, p=16, s=8 | 336->96 | train=16, eval=8 | 8 | series | 0.3246 | 0.0123 |
| `forecasting-longhorizon-patchtst-3task-scaled-iewc_gss-seed0-lam30.json` | IEWC-GSS | 0 | 30 | long_horizon | Weather,ECL,Traffic | PatchTST d=256, L=4, H=8, ff=512, p=16, s=8 | 336->96 | train=16, eval=8 | 8 | series | 0.3288 | 0.0108 |
| `forecasting-longhorizon-patchtst-3task-scaled-iewc_gss-seed0-lam300.json` | IEWC-GSS | 0 | 300 | long_horizon | Weather,ECL,Traffic | PatchTST d=256, L=4, H=8, ff=512, p=16, s=8 | 336->96 | train=16, eval=8 | 8 | series | 0.3387 | 0.0096 |
| `forecasting-longhorizon-patchtst-3task-scaled-iewc_gss-seed0-lam3p33333.json` | IEWC-GSS | 0 | 3.3333 | long_horizon | Weather,ECL,Traffic | PatchTST d=256, L=4, H=8, ff=512, p=16, s=8 | 336->96 | train=16, eval=8 | 8 | series | 0.3249 | 0.0119 |
| `forecasting-longhorizon-patchtst-3task-scaled-iewc_gss-seed0-lam900.json` | IEWC-GSS | 0 | 900 | long_horizon | Weather,ECL,Traffic | PatchTST d=256, L=4, H=8, ff=512, p=16, s=8 | 336->96 | train=16, eval=8 | 8 | series | 0.3496 | 0.0054 |
| `forecasting-longhorizon-patchtst-3task-scaled-iewc_gss-seed1-lam1p11111.json` | IEWC-GSS | 1 | 1.1111 | long_horizon | Weather,ECL,Traffic | PatchTST d=256, L=4, H=8, ff=512, p=16, s=8 | 336->96 | train=16, eval=8 | 8 | series | 0.3338 | 0.0313 |
| `forecasting-longhorizon-patchtst-3task-scaled-iewc_gss-seed2-lam1p11111.json` | IEWC-GSS | 2 | 1.1111 | long_horizon | Weather,ECL,Traffic | PatchTST d=256, L=4, H=8, ff=512, p=16, s=8 | 336->96 | train=16, eval=8 | 8 | series | 0.3421 | 0.0346 |
| `forecasting-longhorizon-patchtst-3task-scaled-sequential-seed0.json` | Sequential | 0 | - | long_horizon | Weather,ECL,Traffic | PatchTST d=256, L=4, H=8, ff=512, p=16, s=8 | 336->96 | train=16, eval=8 | 8 | series | 0.3211 | 0.0135 |
| `forecasting-longhorizon-patchtst-3task-scaled-sequential-seed1.json` | Sequential | 1 | - | long_horizon | Weather,ECL,Traffic | PatchTST d=256, L=4, H=8, ff=512, p=16, s=8 | 336->96 | train=16, eval=8 | 8 | series | 0.3402 | 0.0361 |
| `forecasting-longhorizon-patchtst-3task-scaled-sequential-seed2.json` | Sequential | 2 | - | long_horizon | Weather,ECL,Traffic | PatchTST d=256, L=4, H=8, ff=512, p=16, s=8 | 336->96 | train=16, eval=8 | 8 | series | 0.3421 | 0.0348 |
| `pilot-m4-forecast-ef.json` | EF-EWC | 0 | 10 | - | hourly,weekly,daily | TransformerEncoder d=64, L=2, H=4, ff=128 | 48->12 | train=4, eval=- | 3 | - | 1.9959 | 0.7133 |
| `pilot-m4-forecast-iewc.json` | IEWC | 0 | 10 | - | hourly,weekly,daily | TransformerEncoder d=64, L=2, H=4, ff=128 | 48->12 | train=4, eval=- | 3 | - | 2.0085 | 0.7272 |
| `pilot-m4-forecast-sequential.json` | Sequential | 0 | - | - | hourly,weekly,daily | TransformerEncoder d=64, L=2, H=4, ff=128 | 48->12 | train=4, eval=- | 3 | - | 2.0826 | 0.8395 |
| `pilot-m4-transformer-ef-l1-4freq-e4.json` | EF-EWC | 0 | 1 | - | hourly,weekly,daily,monthly | TransformerEncoder d=64, L=2, H=4, ff=128 | 48->12 | train=4, eval=- | 4 | - | 2.9904 | 1.7264 |
| `pilot-m4-transformer-ef-l10-4freq-e4.json` | EF-EWC | 0 | 10 | - | hourly,weekly,daily,monthly | TransformerEncoder d=64, L=2, H=4, ff=128 | 48->12 | train=4, eval=- | 4 | - | 3.0066 | 1.7422 |
| `pilot-m4-transformer-ef-l100-4freq-e4.json` | EF-EWC | 0 | 100 | - | hourly,weekly,daily,monthly | TransformerEncoder d=64, L=2, H=4, ff=128 | 48->12 | train=4, eval=- | 4 | - | 3.163 | 1.9097 |
| `pilot-m4-transformer-ef-l3-4freq-e4.json` | EF-EWC | 0 | 3 | - | hourly,weekly,daily,monthly | TransformerEncoder d=64, L=2, H=4, ff=128 | 48->12 | train=4, eval=- | 4 | - | 2.992 | 1.7274 |
| `pilot-m4-transformer-ef-l30-4freq-e4.json` | EF-EWC | 0 | 30 | - | hourly,weekly,daily,monthly | TransformerEncoder d=64, L=2, H=4, ff=128 | 48->12 | train=4, eval=- | 4 | - | 3.0561 | 1.7959 |
| `pilot-m4-transformer-iewc-l1-4freq-e4.json` | IEWC | 0 | 1 | - | hourly,weekly,daily,monthly | TransformerEncoder d=64, L=2, H=4, ff=128 | 48->12 | train=4, eval=- | 4 | - | 2.9907 | 1.7268 |
| `pilot-m4-transformer-iewc-l10-4freq-e4.json` | IEWC | 0 | 10 | - | hourly,weekly,daily,monthly | TransformerEncoder d=64, L=2, H=4, ff=128 | 48->12 | train=4, eval=- | 4 | - | 3.0034 | 1.7358 |
| `pilot-m4-transformer-iewc-l100-4freq-e4.json` | IEWC | 0 | 100 | - | hourly,weekly,daily,monthly | TransformerEncoder d=64, L=2, H=4, ff=128 | 48->12 | train=4, eval=- | 4 | - | 3.1436 | 1.8704 |
| `pilot-m4-transformer-iewc-l3-4freq-e4.json` | IEWC | 0 | 3 | - | hourly,weekly,daily,monthly | TransformerEncoder d=64, L=2, H=4, ff=128 | 48->12 | train=4, eval=- | 4 | - | 2.9925 | 1.7279 |
| `pilot-m4-transformer-iewc-l30-4freq-e4.json` | IEWC | 0 | 30 | - | hourly,weekly,daily,monthly | TransformerEncoder d=64, L=2, H=4, ff=128 | 48->12 | train=4, eval=- | 4 | - | 3.0505 | 1.781 |
| `pilot-m4-transformer-sequential-4freq-e4.json` | Sequential | 0 | - | - | hourly,weekly,daily,monthly | TransformerEncoder d=64, L=2, H=4, ff=128 | 48->12 | train=4, eval=- | 4 | - | 2.9753 | 1.7058 |
| `smoke-ett-iewc-gss.json` | IEWC-GSS | 0 | 1 | ett | ETTh1,ETTh2 | TransformerEncoder d=16, L=1, H=2, ff=32 | 24->6 | train=8, eval=4 | 1 | - | 3.1772 | 0 |
| `smoke-forecast-iewc.json` | IEWC | 0 | 10 | - | hourly,weekly | TransformerEncoder d=16, L=1, H=2, ff=32 | 24->6 | train=1, eval=- | 1 | - | 4.5911 | 0 |
| `smoke-forecast-sequential.json` | Sequential | 0 | - | - | hourly,weekly | TransformerEncoder d=16, L=1, H=2, ff=32 | 24->6 | train=1, eval=- | 1 | - | 4.5911 | 0 |
| `smoke-longhorizon-patchtst-sequential.json` | Sequential | 0 | - | long_horizon | Weather,ECL | PatchTST d=64, L=2, H=4, ff=128, p=16, s=8 | 96->24 | train=4, eval=2 | 1 | series | 0.6779 | 0 |

## NLP Text Classification

Total runs: `99`.

| artifact | method | seed | lambda | model | adaptation | distributions | train/eval cap | epochs | batch | final avg accuracy | forgetting |
| --- | --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-ef-seed0-lam1.json` | EF-EWC | 0 | 1 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5525 | 0.2405 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-ef-seed0-lam10.json` | EF-EWC | 0 | 10 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5505 | 0.2412 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-ef-seed0-lam100.json` | EF-EWC | 0 | 100 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5525 | 0.2373 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-ef-seed0-lam2700.json` | EF-EWC | 0 | 2700 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5914 | 0.1716 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-ef-seed0-lam3.json` | EF-EWC | 0 | 3 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5492 | 0.2432 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-ef-seed0-lam30.json` | EF-EWC | 0 | 30 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5492 | 0.2422 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-ef-seed0-lam300.json` | EF-EWC | 0 | 300 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5616 | 0.2207 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-ef-seed0-lam900.json` | EF-EWC | 0 | 900 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.575 | 0.1938 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-ewc_dr-seed0-lam1.json` | EWC-DR | 0 | 1 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5499 | 0.2422 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-ewc_dr-seed0-lam10.json` | EWC-DR | 0 | 10 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5486 | 0.2442 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-ewc_dr-seed0-lam100.json` | EWC-DR | 0 | 100 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5574 | 0.2271 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-ewc_dr-seed0-lam2700.json` | EWC-DR | 0 | 2700 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5893 | 0.1777 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-ewc_dr-seed0-lam3.json` | EWC-DR | 0 | 3 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5486 | 0.2442 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-ewc_dr-seed0-lam30.json` | EWC-DR | 0 | 30 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5492 | 0.2412 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-ewc_dr-seed0-lam300.json` | EWC-DR | 0 | 300 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5737 | 0.1992 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-ewc_dr-seed0-lam900.json` | EWC-DR | 0 | 900 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5874 | 0.175 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-iewc-seed0-lam1.json` | IEWC | 0 | 1 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5499 | 0.2434 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-iewc-seed0-lam10.json` | IEWC | 0 | 10 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5479 | 0.2451 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-iewc-seed0-lam100.json` | IEWC | 0 | 100 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.56 | 0.2241 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-iewc-seed0-lam2700.json` | IEWC | 0 | 2700 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5899 | 0.1777 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-iewc-seed0-lam3.json` | IEWC | 0 | 3 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5499 | 0.2422 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-iewc-seed0-lam30.json` | IEWC | 0 | 30 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5492 | 0.2412 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-iewc-seed0-lam300.json` | IEWC | 0 | 300 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5743 | 0.1958 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-iewc-seed0-lam900.json` | IEWC | 0 | 900 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5885 | 0.174 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-iewc_fromp-seed0-lam1.json` | IEWC-FROMP | 0 | 1 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5499 | 0.2434 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-iewc_fromp-seed0-lam10.json` | IEWC-FROMP | 0 | 10 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5473 | 0.2451 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-iewc_fromp-seed0-lam100.json` | IEWC-FROMP | 0 | 100 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5593 | 0.2263 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-iewc_fromp-seed0-lam2700.json` | IEWC-FROMP | 0 | 2700 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5886 | 0.1777 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-iewc_fromp-seed0-lam3.json` | IEWC-FROMP | 0 | 3 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5492 | 0.2432 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-iewc_fromp-seed0-lam30.json` | IEWC-FROMP | 0 | 30 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5505 | 0.2393 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-iewc_fromp-seed0-lam300.json` | IEWC-FROMP | 0 | 300 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5735 | 0.1972 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-iewc_fromp-seed0-lam900.json` | IEWC-FROMP | 0 | 900 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.59 | 0.1728 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-iewc_gss-seed0-lam1.json` | IEWC-GSS | 0 | 1 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5505 | 0.2425 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-iewc_gss-seed0-lam10.json` | IEWC-GSS | 0 | 10 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5479 | 0.2451 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-iewc_gss-seed0-lam100.json` | IEWC-GSS | 0 | 100 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5585 | 0.2263 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-iewc_gss-seed0-lam2700.json` | IEWC-GSS | 0 | 2700 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5886 | 0.1777 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-iewc_gss-seed0-lam3.json` | IEWC-GSS | 0 | 3 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5512 | 0.2412 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-iewc_gss-seed0-lam30.json` | IEWC-GSS | 0 | 30 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5499 | 0.2393 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-iewc_gss-seed0-lam300.json` | IEWC-GSS | 0 | 300 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5722 | 0.198 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-iewc_gss-seed0-lam900.json` | IEWC-GSS | 0 | 900 | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5854 | 0.1777 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-1024-e3-sequential-seed0.json` | Sequential | 0 | - | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5505 | 0.2491 |
| `nlp-t5-lora-glue-sst2-mrpc-qqp-sequential-seed0-1024-e3.json` | Sequential | 0 | - | t5-small | lora | sst2->mrpc->qqp | train=1024, eval=512 | 3 | 16 | 0.5505 | 0.2491 |
| `nlp-t5-lora-glue3-sequential-seed0-1024-e3.json` | Sequential | 0 | - | t5-small | lora | sst2->mrpc->rte | train=1024, eval=512 | 3 | 16 | 0.5763 | 0.1143 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-ef-seed0-lam24300.json` | EF-EWC | 0 | 2.43e+04 | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.7235 | 0.0257 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-ef-seed0-lam2700.json` | EF-EWC | 0 | 2700 | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.7558 | 0.0294 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-ef-seed0-lam300.json` | EF-EWC | 0 | 300 | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.7377 | 0.0777 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-ef-seed0-lam8100.json` | EF-EWC | 0 | 8100 | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.743 | 0.0245 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-ef-seed0-lam900.json` | EF-EWC | 0 | 900 | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.752 | 0.0417 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-ef-seed1-lam2700.json` | EF-EWC | 1 | 2700 | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.7469 | 0.0233 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-ef-seed2-lam2700.json` | EF-EWC | 2 | 2700 | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.7315 | 0.0515 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-ewc_dr-seed0-lam100.json` | EWC-DR | 0 | 100 | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.7174 | 0.0882 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-ewc_dr-seed0-lam11p1111.json` | EWC-DR | 0 | 11.1111 | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.6771 | 0.1706 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-ewc_dr-seed0-lam24300.json` | EWC-DR | 0 | 2.43e+04 | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.685 | 0.0147 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-ewc_dr-seed0-lam2700.json` | EWC-DR | 0 | 2700 | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.6905 | 0.06 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-ewc_dr-seed0-lam300.json` | EWC-DR | 0 | 300 | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.7104 | 0.076 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-ewc_dr-seed0-lam33p3333.json` | EWC-DR | 0 | 33.3333 | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.7184 | 0.096 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-ewc_dr-seed0-lam8100.json` | EWC-DR | 0 | 8100 | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.6894 | 0.038 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-ewc_dr-seed0-lam900.json` | EWC-DR | 0 | 900 | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.6965 | 0.0699 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-ewc_dr-seed1-lam33p3333.json` | EWC-DR | 1 | 33.3333 | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.7339 | 0.0814 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-ewc_dr-seed2-lam33p3333.json` | EWC-DR | 2 | 33.3333 | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.711 | 0.1071 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-iewc-seed0-lam24300.json` | IEWC | 0 | 2.43e+04 | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.7045 | 0.0311 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-iewc-seed0-lam2700.json` | IEWC | 0 | 2700 | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.744 | 0.0196 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-iewc-seed0-lam300.json` | IEWC | 0 | 300 | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.7483 | 0.0441 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-iewc-seed0-lam8100.json` | IEWC | 0 | 8100 | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.7168 | 0.0282 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-iewc-seed0-lam900.json` | IEWC | 0 | 900 | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.7587 | 0.0233 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-iewc-seed1-lam900.json` | IEWC | 1 | 900 | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.742 | 0.0368 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-iewc-seed2-lam900.json` | IEWC | 2 | 900 | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.7322 | 0.0515 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-iewc_fromp-seed0-lam24300.json` | IEWC-FROMP | 0 | 2.43e+04 | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.691 | 0.027 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-iewc_fromp-seed0-lam2700.json` | IEWC-FROMP | 0 | 2700 | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.7107 | 0.0404 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-iewc_fromp-seed0-lam300.json` | IEWC-FROMP | 0 | 300 | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.7266 | 0.0723 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-iewc_fromp-seed0-lam8100.json` | IEWC-FROMP | 0 | 8100 | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.6976 | 0.0428 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-iewc_fromp-seed0-lam900.json` | IEWC-FROMP | 0 | 900 | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.7312 | 0.0404 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-iewc_fromp-seed1-lam900.json` | IEWC-FROMP | 1 | 900 | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.7156 | 0.0576 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-iewc_fromp-seed2-lam900.json` | IEWC-FROMP | 2 | 900 | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.6883 | 0.0956 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-iewc_gss-seed0-lam24300.json` | IEWC-GSS | 0 | 2.43e+04 | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.6914 | 0.0233 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-iewc_gss-seed0-lam2700.json` | IEWC-GSS | 0 | 2700 | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.7085 | 0.0429 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-iewc_gss-seed0-lam300.json` | IEWC-GSS | 0 | 300 | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.723 | 0.0735 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-iewc_gss-seed0-lam8100.json` | IEWC-GSS | 0 | 8100 | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.6989 | 0.0392 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-iewc_gss-seed0-lam900.json` | IEWC-GSS | 0 | 900 | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.7266 | 0.0404 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-iewc_gss-seed1-lam900.json` | IEWC-GSS | 1 | 900 | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.7053 | 0.0674 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-iewc_gss-seed2-lam900.json` | IEWC-GSS | 2 | 900 | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.6856 | 0.0906 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-sequential-seed0.json` | Sequential | 0 | - | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.7028 | 0.1333 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-sequential-seed1.json` | Sequential | 1 | - | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.7091 | 0.1331 |
| `nlp-t5base-lora-glue-sst2-mrpc-qqp-2048-e3-highlambda-sequential-seed2.json` | Sequential | 2 | - | t5-base | lora | sst2->mrpc->qqp | train=2048, eval=1024 | 3 | 8 | 0.6693 | 0.1966 |
| `pilot-glue-roberta-lora-ef.json` | EF-EWC | 0 | 100 | roberta-base | lora | sst2->mrpc | train=64, eval=64 | 1 | 8 | 0.5781 | 0.0312 |
| `pilot-glue-roberta-lora-iewc.json` | IEWC | 0 | 100 | roberta-base | lora | sst2->mrpc | train=64, eval=64 | 1 | 8 | 0.5781 | 0.0312 |
| `pilot-glue-roberta-lora-sequential.json` | Sequential | 0 | - | roberta-base | lora | sst2->mrpc | train=64, eval=64 | 1 | 8 | 0.5781 | 0.0312 |
| `pilot-glue-t5-small-lora-sequential.json` | Sequential | 0 | - | google-t5/t5-small | lora | sst2->mrpc | train=32, eval=32 | 1 | 4 | 0.5 | 0 |
| `pilot-nlp-roberta-lora-ef-l100-sst2-mrpc-256x2.json` | EF-EWC | 0 | 100 | roberta-base | lora | sst2->mrpc | train=256, eval=128 | 2 | 16 | 0.6172 | 0.2031 |
| `pilot-nlp-roberta-lora-iewc-l100-sst2-mrpc-256x2.json` | IEWC | 0 | 100 | roberta-base | lora | sst2->mrpc | train=256, eval=128 | 2 | 16 | 0.6172 | 0.2031 |
| `pilot-nlp-roberta-lora-sequential-3task-256x2.json` | Sequential | 0 | - | roberta-base | lora | sst2->mrpc->rte | train=256, eval=128 | 2 | 16 | 0.5677 | 0.1055 |
| `pilot-nlp-roberta-lora-sequential-sst2-mrpc-256x2.json` | Sequential | 0 | - | roberta-base | lora | sst2->mrpc | train=256, eval=128 | 2 | 16 | 0.6367 | 0.2031 |
| `pilot-nlp-roberta-lora-sequential-sst2-mrpc-qnli-256x2.json` | Sequential | 0 | - | roberta-base | lora | sst2->mrpc->qnli | train=256, eval=128 | 2 | 16 | 0.4115 | 0.332 |
| `pilot-nlp-t5-small-lora-ef-l100-sst2-mrpc-256x2.json` | EF-EWC | 0 | 100 | t5-small | lora | sst2->mrpc | train=256, eval=128 | 2 | 8 | 0.6133 | 0.0625 |
| `pilot-nlp-t5-small-lora-iewc-l100-sst2-mrpc-256x2.json` | IEWC | 0 | 100 | t5-small | lora | sst2->mrpc | train=256, eval=128 | 2 | 8 | 0.6133 | 0.0625 |
| `pilot-nlp-t5-small-lora-sequential-sst2-mrpc-256x2.json` | Sequential | 0 | - | t5-small | lora | sst2->mrpc | train=256, eval=128 | 2 | 8 | 0.6133 | 0.0625 |
| `smoke-nlp-iewc.json` | IEWC | 0 | 100 | roberta-base | lora | synthetic_a->synthetic_b | train=16, eval=16 | 1 | 8 | 0.5 | 0 |
| `smoke-nlp-sequential.json` | Sequential | 0 | - | roberta-base | full | synthetic_a->synthetic_b | train=16, eval=16 | 1 | 8 | 0.5 | 0 |
| `smoke-nlp-t5-gss.json` | IEWC-GSS | 0 | 1 | t5-small | lora | sst2->mrpc | train=8, eval=8 | 1 | 2 | 0.5625 | 0 |
