# AMML Summer 2026 

This project rebuilds both assignment tasks around the official brief.

## What the code does

### Task 1: evaluate the three supplied pre-trained VAEs

The code **does not retrain or rename** `model0`, `model1` or `model2`. It loads the three official weight files into the single supplied `VariationalAutoencoder` architecture and evaluates them under identical conditions.

It produces:

- deterministic reconstruction metrics for every test image: BCE, MSE, MAE and SSIM;
- per-class and overall reconstruction summaries;
- paired sample-level t-tests, Wilcoxon tests, Holm-adjusted p-values, effect sizes and Friedman tests;
- identical side-by-side qualitative reconstruction grids;
- posterior-sampling variability and prior-generation diversity measures;
- PCA and t-SNE latent-space visualisations;
- latent diversity and class-separation diagnostics;
- downstream classification from posterior means using the same standardised logistic-regression pipeline for all three models;
- ten paired classification runs by default, with mean, standard deviation and 95% confidence intervals;
- paired statistical tests across seeds;
- normalised confusion matrices and high-confidence misclassification examples.

### Task 2: small, imbalanced holdout dataset

Task 2 loads **only** `holdout_dataset.pt`. It never accesses the Task 1 test set.

The model is a compact multitask VAE classifier. Its VAE is initialised from supplied `model0` weights. A small classifier head operates on the posterior mean. The model therefore supports both required tasks:

1. image reconstruction;
2. digit classification.

Two paired variants are trained on the same fixed stratified split:

- `baseline`: ordinary cross-entropy;
- `weighted_ce`: inverse-frequency class-weighted cross-entropy.

No synthetic images are used to balance the data. This avoids introducing the recursive self-generated-data risk highlighted by Task 1.

The code reports:

- original and split-level class distributions;
- explicit train/validation/test indices;
- the exact class weights;
- the full multitask loss and every training hyperparameter;
- correctly averaged training and validation losses;
- early stopping and learning-rate reduction;
- accuracy, balanced accuracy, macro precision, macro recall, macro-F1, weighted-F1 and MCC;
- per-class recall and reconstruction results;
- dominant-versus-minority class comparisons;
- reconstruction metrics and qualitative reconstructions;
- prediction variability across seeds;
- paired baseline-versus-weighted tests;
- normalised confusion matrices and error examples.




## Main files for the report

### Task 1

- `task1_reconstruction_summary.csv`
- `task1_reconstruction_by_class.csv`
- `task1_reconstruction_pairwise_tests.csv`
- `task1_stochastic_variability_summary.csv`
- `task1_latent_diagnostics.csv`
- `task1_classification_summary.csv`
- `task1_classification_pairwise_tests.csv`
- `task1_reconstruction_same_samples.png`
- `task1_latent_space_pca.png`
- `task1_latent_space_tsne.png`
- the three normalised confusion matrices

### Task 2

- `task2_holdout_class_distribution.csv`
- `task2_fixed_stratified_split_indices.csv`
- `task2_class_weights.csv`
- `task2_test_results_summary.csv`
- `task2_baseline_vs_weighted_paired_tests.csv`
- `task2_per_class_results_summary.csv`
- `task2_dominant_vs_minority_summary.csv`
- `task2_prediction_variability_summary.csv`
- `task2_dominant_and_minority_reconstructions.png`
- `task2_per_class_recall.png`
- the two normalised confusion matrices
