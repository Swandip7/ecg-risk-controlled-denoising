# Risk-Controlled ECG Denoising for Downstream Decision Preservation

Code, frozen artifacts, and result tables for the paper *Risk-Controlled ECG Denoising for Downstream Decision Preservation* (DOI: to be added).

## What this repository contains

1. **Decision-level evaluation.** Harm (raw prediction correct, denoised prediction wrong), rescue, and net benefit, computed against a frozen classifier.
2. **Decision-consistency loss.** `loss = reconstruction + STFT + λ · KL(f(clean) ‖ f(denoised))` for three denoiser families (FCN-DAE, DRNN-style, DeepFilter-style, all our re-implementations).
3. **Conformal risk-controlled gate.** Chooses between the denoised and the raw prediction and bounds expected harm at a user-set level α (patient-level splits for atrial fibrillation, record-level for heartbeats).
4. **Generative noise model.** One diffusion model per NSTDB noise type (bw, em, ma), compared with spliced and additive NSTDB noise.

| Task | Dataset | Input | Label | Evaluation pool |
|---|---|---|---|---|
| Atrial fibrillation vs sinus | PTB-XL v1.0.3 | 10 s, lead II (V1 secondary) | rhythm | 3,645 windows (303 AF), folds 9-10 |
| Heartbeat class (N/S/V/F) | MIT-BIH Arrhythmia | 2 s beat window + RR features | AAMI class | 49,652 beats, 22 DS2 records |

## Data (not included)

Download from PhysioNet and do not commit raw signals.

| Dataset | Version | Link |
|---|---|---|
| PTB-XL | 1.0.3 | https://physionet.org/content/ptb-xl/1.0.3/ |
| MIT-BIH Arrhythmia | 1.0.0 | https://physionet.org/content/mitdb/1.0.0/ |
| MIT-BIH Noise Stress Test | 1.0.0 | https://physionet.org/content/nstdb/1.0.0/ |

Set these before running:
```bash
export ECG_DATA_ROOT=/path/to/datasets   # folder that contains the three datasets
export ECG_WORK=/path/to/workdir         # where data/, models/, results/ are written
```
The three dataset folder names are set at the top of the notebook (setup cell).

## Quickstart

```bash
git clone https://github.com/<user>/<repo>.git && cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional, to reproduce the paper numbers exactly (see "Frozen artifacts"):
#   download the Zenodo archive and unpack it into $ECG_WORK
jupyter notebook notebooks/ecg_risk_controlled_denoising.ipynb
```

**Runtime (Kaggle GPU).** From scratch: about 4 to 5 hours. Checkpoint timestamps in our run span about 3 hours of training, plus evaluation. With all frozen artifacts in place the notebook skips training; we have not timed that path. Efficiency numbers in the paper come from one thread of an Intel Xeon CPU at 2.00 GHz (`results/FINAL_E8_efficiency.csv`).

## Frozen artifacts and reproducibility

Results are from a **single execution**. GPU training is not bit-reproducible, so a rerun from scratch will give slightly different values, and **the evaluation pool also changes** because it is built from sampled generative noise. To reproduce the paper numbers, use the archived files (Zenodo, DOI to be added):

| File | Needed for |
|---|---|
| `data/twin_pools.npz` | generated noise windows used to build every pool |
| `data/gate_pool_AF.npz` | the exact calibration and score-model pool (3,645 + score-model windows) |
| `models/*.pt` (all checkpoints, about 70 files) | classifiers (`af_II`, `af_V1`, `beat_clf`), the three generators (`g2_*`), 56 AF denoisers, 10 beat denoisers |

Use **all** checkpoints or **none**. The notebook loads any checkpoint that exists and trains the rest, and mixing the two gives inconsistent numbers.

The NSTDB content-disjoint split (`splits/nstdb_block_split.pkl`) is recomputed deterministically on the CPU (seed 0). The file is shipped only for audit.

With the archive in place, deterministic cells (classifier evaluation, band-pass baselines, efficiency) agree to floating-point tolerance. Training-dependent numbers agree within the seed-to-seed spread reported in the paper.

## Paper item → output file

| Paper item | Output file(s) in `results/` |
|---|---|
| Table I (AF, noise source × λ) | `FINAL_E4_AF_consistency.csv`, `FINAL_E4_AF_paired_lambda.csv` |
| Table II (architectures, Kendall τ) | `FINAL_E2b_architectures_mean.csv`, `FINAL_E2_RQ1_kendall.csv`, `FINAL_E2b_kendall_architectures.csv` |
| Table III (gate validity) | `FINAL_E5_AF_gate.csv`, `FINAL_E5c_gate_architectures.csv` |
| Table IV (heartbeat task) | `FINAL_E4_beat_consistency.csv`, `FINAL_E4_beat_per_class.csv`, `FINAL_E5_beat_gate.csv` |
| Text only: learned vs divergence-only score | `FINAL_E5a_AF_gate_learned_vs_KL.csv` |
| Text only: real-only calibration | `FINAL_E5b_AF_gate_real_only.csv` |
| Text only: band-pass vs denoiser | `FINAL_E4b_bandpass_vs_denoiser_pool.csv` |
| Text only: loss-type ablation | `FINAL_ablation_e.csv` |
| Text only: noise type, real flags, lead V1 | `FINAL_E7a_per_noise_type_lead_II.csv`, `FINAL_E7b_real_noise_flags.csv`, `FINAL_E7c_per_lead.csv` |
| Text only: efficiency | `FINAL_E8_efficiency.csv` |
| Text only: realism test | `E1a_realism_live.csv` |
| Fig. 3 (realism curve) | `figures/FIG3_realism_auc.pdf` |
| Fig. 4 (generated noise and spectra) | `figures/FIG4_twin_waveforms_psd.pdf` |
| Fig. 5 (harm vs ΔSNR per model) | `figures/FIG1_scatter.pdf` |
| Fig. 6 (per-split harm histograms) | `figures/FIG2_gate_per_split_hist.pdf` |

Figs. 1 and 2 (framework and study design) are hand-drawn and are not generated by the code. Figure file names follow the notebook, not the paper numbering.

## Reference numbers for spot-checking

Atrial fibrillation, generative noise, mean ± sd over five seeds (pool raw accuracy 93.44%):

| λ | ΔSNR@6 dB | harm % | harm on real noisy % | accuracy % |
|---|---|---|---|---|
| 0 | 4.37 ± 0.08 | 2.84 ± 0.10 | 2.23 ± 0.41 | 95.40 ± 0.13 |
| 1 | 1.71 ± 0.10 | 1.61 ± 0.07 | 0.86 ± 0.21 | 96.51 ± 0.12 |

Gate at λ = 1, 200 patient-level splits (five-seed mean):

| α % | test harm % | denoised % | accuracy % |
|---|---|---|---|
| 0.25 | 0.21 | 93.3 | 96.1 |
| 0.50 | 0.47 | 95.5 | 96.6 |
| 1.00 | 0.96 | 98.2 | 96.8 |

Heartbeats, five-seed mean (raw accuracy 88.39%): harm 10.22% at λ = 0 and 6.28% at λ = 1; band-pass 8.17%. Record-level gate at α = 0.5%: harm 0.59% (above target; window-level 0.50%).

Realism test, AUC for separating real flagged-noisy PTB-XL windows from corrupted clean windows (lower = harder to tell apart; clean-record floor 0.71):

| SNR centre | generative | spliced | additive |
|---|---|---|---|
| 6 dB | 0.95 | 0.98 | 0.99 |
| 12 dB | 0.86 | 0.91 | 0.94 |

Small deviations from GPU nondeterminism are expected in a fresh run.

## Environment

Numbers were produced with Python 3.13 on Kaggle. The first notebook cell prints the exact versions of `torch`, `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`, and `wfdb`, and the CPU and GPU names. `requirements.txt` pins those versions.

## Known limitations

- Single execution. Seed spread covers denoiser-training variability only; classifiers, generator, and pool are fixed.
- Baselines are our own re-implementations, not the original authors' code.
- The heartbeat gate does not hold at record level (22 records). See the paper.
- No clinical validity is claimed. This is a research prototype.


## License

Code: MIT, see `LICENSE`. The datasets keep their PhysioNet licenses; check each dataset page. No raw data is redistributed.
