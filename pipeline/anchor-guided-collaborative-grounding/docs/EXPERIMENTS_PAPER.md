# 논문용 하이퍼파라미터 실험 매핑

## (1) α, β, γ — pairwise exclusion 가중치

코드: `src/model_infonce_excl.py` 의 `anchor_exclusion_loss`

\[
\lambda_{ij} = \alpha(1-s^{ent}) + \beta(1-s^{reg}) + \gamma(1-s^{type})
\]

| 기호 | 환경변수 | 비고 |
|------|-----------|------|
| α | `EXCL_PAIR_ALPHA` | |
| β | `EXCL_PAIR_BETA` | |
| γ | (기본) `1-α-β` | `EXCL_PAIR_GAMMA` 미설정 시 |
| γ (명시) | `EXCL_PAIR_GAMMA` | 설정 시 `(α,β,γ)`를 **양수 합으로 정규화** |

**스윕 스크립트:** `run_paper_hyperparam_sweep.sh`

- `SWEEP_PHASE=alpha` — `FIX_BETA` 고정, `SWEEP_ALPHA` 스윕 (implicit γ)
- `SWEEP_PHASE=beta` — `FIX_ALPHA` 고정, `SWEEP_BETA` 스윕
- `SWEEP_PHASE=gamma` — `FIX_ALPHA_RAW`, `FIX_BETA_RAW` 고정, `SWEEP_GAMMA_RAW` 스윕 (명시 γ + 정규화)

**다중 엔티티 / 유사 엔티티:** 로그의 `[Dev-by-EntCount]`, `[TEST-by-EntCount]` 에서 `E=2,3,...` 버킷을 같이 보고하면 됨.

## (2) λ_comp vs λ_nce

\[
L = \lambda_{comp} L_{comp} + \lambda_{nce} L_{nce}
\]

| 기호 | 환경변수 |
|------|-----------|
| λ_comp | `LOSS_EXCL_W` |
| λ_nce | `LOSS_INFO_NCE_W` |

**스윕:** `SWEEP_PHASE=loss_ratio` — 기본적으로 `LOSS_INFO_NCE_W=1` 고정 후 `SWEEP_LOSS_EXCL_W` 스윕.

## (3) Joint temperature τ (경쟁 분포 날카로움)

\[
P \propto \exp(s/\tau)
\]

| 환경변수 | `TEMPERATURE` |

**스윕:** `SWEEP_PHASE=joint_temp` — `SWEEP_JOINT_TEMP` (기본: `0.05 0.1 0.5 0.8 1.0`).  
매우 작은 값은 gradient/수치 불안정 가능.

## (4) InfoNCE temperature (대조 학습)

| 환경변수 | `INFO_NCE_TAU` |

**스윕:** `SWEEP_PHASE=nce_tau` — `SWEEP_NCE_TAU` (기본: `0.01 0.03 0.05 0.07 0.1`).

## 실행 예시

```bash
cd /workspace/ancor/pipeline/anchor-guided-collaborative-grounding

SWEEP_PHASE=alpha RESUME_SWEEP=1 bash run_paper_hyperparam_sweep.sh
SWEEP_PHASE=loss_ratio RESUME_SWEEP=1 bash run_paper_hyperparam_sweep.sh
```

tmux:

```bash
tmux new -s hp_alpha \
'cd /workspace/ancor/pipeline/anchor-guided-collaborative-grounding && SWEEP_PHASE=alpha RESUME_SWEEP=1 bash run_paper_hyperparam_sweep.sh'
```

`SWEEP_PHASE=all` 은 위 단계를 순서대로 전부 돌림 (총 실행 횟수 많음).
