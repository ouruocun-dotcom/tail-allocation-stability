# Tail allocation and stability in heavy-tailed networks

Reproduction code for **Tail allocation controls stability in heavy-tailed networks with identical spectral bulk**. Author: Ruocun Ou. Version: 2026.10.13.

## Install and run

Tested with Python 3.12. Install the numerical libraries (tested with NumPy 2.5.4, SciPy 1.18.1 and Matplotlib 3.11.2):

```sh
python -m pip install numpy==2.5.4 scipy==1.18.1 matplotlib==3.11.2
```

From the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python code/reproduce.py all
python code/supplementary.py
```

The first command generates all 576 primary matrices and the four main figures. The second generates 288 amplitude-copula matrices, the supplementary summaries, Figure S1, and the final three-panel replacement for Figure 4. Use `reproduce.py run` or `reproduce.py figures` for separate stages. After computing both scans, `supplementary.py --figures-only` redraws the final figures without generating matrices. The final Figure 4 is the one produced by the second command. Runtime depends on hardware; dense spectra at size 1024 dominate the primary scan.

No external data, precomputed inputs or network access are needed after installation. Run from any directory; outputs are relative to the scripts' repository root. A writable `MPLCONFIGDIR` may be needed on restricted systems.

## Model and fixed seeds

Primary scan: Pareto magnitudes with exponent 2.5, gain 0.65 and body boundaries 1.5 and 8. Independent unordered pairs have equal reverse magnitudes and symmetric signs. Two body probabilities preserve the reinforcing fraction 1/2 and zero population reciprocal covariance while tail reinforcement is changed. The four allocations are delta=0,1,2 and the hard-tail control. No magnitude truncation or sample-moment renormalization is used.

Sizes 64,128,256,512 each use 32 independent pair-array blocks; size 1024 uses 16. Four allocations share each block: 144 independent blocks and 576 matrices. Seeds are `2026101201 + 10000*N + replicate`, starting at replicate 0.

Amplitude-copula scan: sizes 128,256,512 each use 16 blocks, paired across kappa=0,0.5,1 and delta=0,2. Seeds are `2026101291 + 10000*N + replicate`, disjoint from the primary scan. Reverse magnitudes are independent with probability 1-kappa and equal with probability kappa. Signs are allocated by their geometric mean and body probabilities are recalibrated to maintain matching. These are 48 independent blocks and 288 matrices, not 288 independent comparisons.

## Outputs and manuscript mapping

- `results/scan.json`: every primary realization; local stability, spectral abscissa, symmetric-part sufficient test, pair magnitude, eigenvector mass/IPR, sample reciprocal covariance and reinforcing fraction.
- `results/spectra.npz` and `results/leading_vectors.npz`: full primary spectra and leading normalized right eigenvectors.
- `results/review_diagnostics.json`: Table S1 exact pair-threshold probabilities/counts; Table S2 primary medians/test counts; Table S3 covariance and fraction quantiles; Table S4 copula counts/medians; every copula realization; body probabilities; paired bootstrap summaries.
- `results/copula_spectra.npz`: full spectra for all copula settings.
- `figures/`: four main figure PDFs and the supplementary slow-branch PDF. Figure 4 is updated by the second command.

The stability fractions use pointwise Wilson intervals. Paired delta2-minus-delta0 differences resample realization blocks 10,000 times, seed 2026101299; these are pointwise percentile intervals. Table S3 gives empirical 10th/50th/90th percentiles, not confidence intervals. Sample covariances are not forced to match and have infinite variance despite a finite population mean.

The exact maximum-pair law is distinct from the full-network instability probability. Floating-point sufficient tests are not interval arithmetic; failing a test does not imply stability. The copula calculations delimit sensitivity and do not extend the equal-amplitude theorem to arbitrary reciprocal dependence. This source package generates all synthetic numerical results; it does not replace the mathematical proofs in the manuscript and supplement.
