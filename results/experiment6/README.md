# Experiment 6: Zero-Shot System One Decision Model (`Jev`)

Evaluate TypeSafe AI's non-autoregressive System One model (**`jev-1.14.0`**) zero-shot on the held-out **native Bengali** test split (`bn-test`, $n=490$) and the **English** test split (`en-test`, $n=356$), with no task-specific training.

| Condition | Model | Primitive | Rubric | Decision Rule | Test Splits |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **6a (`jev_choice`)** | `jev-1.14.0` | `Choice` (4 discrete classes) | `unified` (DSM-5) | $\arg\max_k p_k$ | `bn-test` ($n=490$), `en-test` ($n=356$) |
| **6b (`jev_score`)** | `jev-1.14.0` | `Score` (4-level ordered rubric) | `unified` (DSM-5) | $\text{round}(E[Y])$ and $\arg\max_k p_k$ | `bn-test` ($n=490$), `en-test` ($n=356$) |
| **6c (`naseem_dda`)** | `jev-1.14.0` | `Choice` & `Score` | `naseem_dda` (Reddit DDA) | $\arg\max_k p_k$ and $\text{round}(E[Y])$ | `en-test` ($n=356$) |

Setup: Each social media post is evaluated in a single forward pass via `typesafe-sdk` (`src/jev_client.py`), querying both `Choice` and `Score`. Conditions `6a` and `6b` use the unified 4-tier DSM-5 clinical severity criteria across both languages; Condition `6c` is an English schema-alignment ablation using the Depressive Disorder Annotation (DDA) descriptors from Naseem et al. (2022, Fig. 1 & Table 2). Because `jev-1.14.0` is deterministic, metrics are single-pass point estimates accompanied by 1,000-sample bootstrap 95% confidence intervals (`seed=42`).

## Artefacts

- `6a_jev_choice_{bn,en}/eval_test.json` — full ordinal metric suite for the `Choice` primitive (`unified` rubric) on `bn-test` and `en-test`.
- `6b_jev_score_{bn,en}/eval_test.json` — full ordinal metric suite for the `Score` primitive (`unified` rubric, rounded expected value $E[Y] = \sum_{k=1}^4 k \cdot p_k$, plus `score_argmax_summary`) on `bn-test` and `en-test`.
- `6c_jev_{choice,score}_naseem_dda_en/eval_test.json` — full ordinal metric suite for the `naseem_dda` rubric ablation on `en-test`.

Entry point: [`evaluate_jev.py`](../../evaluate_jev.py) and [`notebooks/experiment6_jev.ipynb`](../../notebooks/experiment6_jev.ipynb).

## Results

### 1. Held-Out Native Bengali Test Split (`bn-test`, $n=490$)

| Metric | `6a Choice (BN)` | `6b Score E[Y] (BN)` | `6b Score argmax (BN)` |
| :--- | :---: | :---: | :---: |
| **`macro-F1` (95% CI)** | `0.5873 [0.5495, 0.6246]` | `0.5703 [0.5346, 0.6072]` | **`0.5960`** |
| **`QWK` (95% CI)** | `0.6062 [0.5344, 0.6672]` | **`0.6161 [0.5526, 0.6718]`** | **`0.6161`** |
| **`MAE`** | `0.5633` | `0.5633` | **`0.5490`** |
| **`accuracy`** | `0.5306` | `0.5143` | **`0.5429`** |
| **`Graded FScore` (`GP / GR`)** | `0.6933 (0.5677 / 0.8904)` | `0.6792 (0.5408 / 0.9130)` | **`0.7037 (0.5708 / 0.9172)`** |
| **`downgrade_rate` (`pred < gold`)** | `0.0653 (32/490)` | **`0.0490 (24/490)`** | **`0.0490 (24/490)`** |
| **`upgrade_rate` (`pred > gold`)** | `0.4041 (198/490)` | `0.4367 (214/490)` | `0.4082 (200/490)` |
| **`downgrade_share_of_errors`** | `0.1391` | `0.1008` | `0.1071` |
| **`mean_signed_error`** | `+0.4163` | `+0.4612` | `+0.4122` |
| **`ECE (10-bin)`** | `0.2556` | `0.2666` | — |

Per-class performance on `bn-test` ($n=490$):

| Class | True Support | `6a Choice` Pred Count | `6a Choice` F1 (P / R) | `6b Score E[Y]` Pred Count | `6b Score E[Y]` F1 (P / R) | `6b Score argmax` F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`minimal` (1)** | `210` | `68` | `0.3453 (0.7059 / 0.2286)` | `45` | `0.2824 (0.8000 / 0.1714)` | `0.3663` |
| **`mild` (2)** | `154` | `240` | `0.5381 (0.4417 / 0.6883)` | `261` | `0.5301 (0.4215 / 0.7143)` | `0.5459` |
| **`moderate` (3)** | `71` | `121` | `0.5521 (0.4380 / 0.7465)` | `126` | `0.5482 (0.4286 / 0.7606)` | `0.5657` |
| **`severe` (4)** | `55` | `61` | `0.9138 (0.8689 / 0.9636)` | `58` | `0.9204 (0.8966 / 0.9455)` | `0.9060` |

### 2. Held-Out English Test Split (`en-test`, $n=356$): Unified Rubric (`6a/6b`) vs. Naseem-DDA Rubric (`6c`)

| Metric | `6a Choice (unified)` | `6b Score E[Y] (unified)` | `6b Score argmax (unified)` | `6c Choice (naseem_dda)` | `6c Score E[Y] (naseem_dda)` | `6c Score argmax (naseem_dda)` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`macro-F1`** | `0.4291` | `0.3949` | `0.4188` | **`0.4523`** | `0.4094` | `0.4426` |
| **`macro-F1` 95% CI** | `[0.3618, 0.4904]` | `[0.3316, 0.4572]` | — | `[0.3883, 0.5158]` | `[0.3500, 0.4673]` | — |
| **`QWK`** | `0.3690` | `0.4206` | `0.4233` | `0.4939` | `0.4876` | **`0.5038`** |
| **`QWK` 95% CI** | `[0.2588, 0.4785]` | `[0.3169, 0.5265]` | — | `[0.3880, 0.5914]` | `[0.3873, 0.5760]` | — |
| **`MAE`** | `0.5534` | `0.5787` | `0.5534` | **`0.4972`** | `0.5365` | `0.5028` |
| **`accuracy`** | `0.6573` | `0.5758` | `0.6376` | **`0.6910`** | `0.6236` | `0.6798` |
| **`Graded Precision (GP)`** | `0.7723` | `0.6788` | `0.7419` | **`0.7859`** | `0.7115` | `0.7654` |
| **`Graded Recall (GR)`** | `0.8153` | `0.7915` | `0.8195` | `0.8512` | `0.8346` | **`0.8551`** |
| **`Graded FScore`** | `0.7932` | `0.7308` | `0.7788` | **`0.8173`** | `0.7682` | `0.8078` |
| **`downgrade_rate`** | `0.1489 (53)` | `0.1517 (54)` | `0.1404 (50)` | `0.1208 (43)` | `0.1236 (44)` | **`0.1152 (41)`** |
| **`upgrade_rate`** | `0.1938 (69)` | `0.2725 (97)` | `0.2219 (79)` | **`0.1882 (67)`** | `0.2528 (90)` | `0.2051 (73)` |
| **`mean_signed_error`** | `+0.0028` | `+0.0955` | `+0.0562` | `+0.0927` | `+0.1489` | `+0.1124` |
| **`ECE (10-bin)`** | `0.1565` | `0.2270` | — | `0.1609` | `0.2178` | — |

Per-class performance on `en-test` ($n=356$):

| Class (Support) | `6a Choice (unified)` F1 (Pred) | `6b Score E[Y] (unified)` F1 (Pred) | `6c Choice (naseem_dda)` F1 (Pred) | `6c Score E[Y] (naseem_dda)` F1 (Pred) | `6c Score argmax` F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`minimal` (259)** | `0.8121 (236)` | `0.7577 (195)` | **`0.8455 (233)`** | `0.8052 (203)` | `0.8412` |
| **`mild` (29)** | **`0.2697 (60)`** | `0.2044 (108)` | `0.2121 (37)` | `0.1942 (74)` | `0.1714` |
| **`moderate` (40)** | `0.3488 (46)` | `0.3250 (40)` | `0.4404 (69)` | `0.4000 (65)` | **`0.4464`** |
| **`severe` (28)** | `0.2857 (14)` | `0.2927 (13)` | **`0.3111 (17)`** | `0.2381 (14)` | **`0.3111`** |

## Interpretation

### 1. Ordinal Separability and Clinical Downgrade Protection on `bn-test`
On `bn-test` ($n=490$), `jev-1.14.0` reaches **`0.5873`** `macro-F1` (`Choice`), **`0.5703`** (`Score` rounded expectation $E[Y]$), and **`0.5960`** (`Score` argmax), with `QWK = 0.6062–0.6161`. The median expected severity score $E[Y]$ rises monotonically across true severity tiers (`2.000` for `minimal`, `2.155` for `mild`, `2.900` for `moderate`, and `3.980` for `severe`). Detection is strongest on `severe` (`F1 = 0.9138–0.9204`, identifying `53/55` cases in `Choice`) and consistent across the interior tiers (`mild` `F1 = 0.5301–0.5459`, `moderate` `F1 = 0.5482–0.5657`). Clinically dangerous downgrades (`pred < gold`) occur on only **`4.90%–6.53%`** of Bengali test items.

### 2. Upward Boundary Shift Between `minimal` and `mild` on `bn-test`
Errors on `bn-test` are predominantly 1-tier upgrades (`upgrade_rate = 0.4041–0.4367`; `mean_signed_error = +0.4122` to `+0.4612`), driven by `124/210` (`Choice`) and `139/210` (`Score`) true `minimal` posts being predicted as `mild`. Because the Bengali corpus (`Kabir et al., 2022`) was collected from mental health counseling forums where even `Level 1 (minimal)` posts frequently describe situational crying, grief, or heartbreak, a zero-shot clinical rubric maps those expressions of sadness to `mild`.

### 3. `Choice` vs. `Score` Primitives (`E[Y]` Rounding vs. Argmax)
Across both languages and rubrics, taking the mode (`argmax`) of the `Score` rubric distribution outperforms rounding the continuous expectation $E[Y] = \sum_{k=1}^4 k \cdot p_k$ on `macro-F1` (`0.5960` vs. `0.5703` on `bn-test`; `0.4426` vs. `0.4094` on `en-test` in `6c`). When probability mass is split between `minimal` (`1`) and `moderate` (`3`), the expected value $E[Y]$ falls near `2.0` and rounds into `mild`, inflating false positives on the `mild` tier.

### 4. Why `en-test` Macro-F1 Remains Capped Even Under the `6c` Naseem-DDA Rubric
Adapting the rubric to Naseem et al.'s (2022) DDA descriptors (`6c`) substantially improves ordinal alignment on `en-test`: `QWK` jumps by **`+0.1249`** (`0.3690` $\to$ `0.4939` in `Choice`, and `0.5038` in `Score argmax`), `MAE` drops from `0.5534` to **`0.4972`**, `moderate` F1 rises from `0.3488` to **`0.4404`**, `accuracy` rises from `0.6573` to **`0.6910`**, and Naseem et al.'s `Graded FScore` reaches **`0.8173`**. However, unweighted `macro-F1` only rises from `0.4291` to **`0.4523`**.

Inspecting the source paper ([Naseem et al., 2022](https://doi.org/10.1145/3485447.3512128)) and the misclassified rows in `en_test.csv` identifies structural reasons why unweighted `macro-F1` on `en-test` is inherently capped:
1. **5-Sentence Window Truncation from *Dreaddit*:** The English dataset (`D1`) slices long Reddit posts into fixed ~5-sentence excerpts (~80 words). Several excerpts labeled `severe (4)` in `en_test.csv` contain only the introductory setup or a relationship `TL;DR` without the depressive content that appeared elsewhere in the thread (e.g., Row 299, labeled `4`: *"Hey y'all. Throw away for obvious reasons... I started dating my boyfriend two years ago... He was funny, intelligent, and \*very\* charming"*; Row 20, labeled `4`: *"TL;DR I want to focus on my priorities in life but I find myself getting wrapped up in my thoughts about... what my girlfriend is doing"*).
2. **Extreme Minority-Class Imbalance:** The CSV is 72.8% `minimal` (`259/356` test items), leaving only `29` `mild`, `40` `moderate`, and `28` `severe` test items. Because `macro-F1` weights a 28-item minority class equally with the 259-item `minimal` class, ~15–20 spillover predictions from the 259 `minimal` posts into a 29-item bucket mathematically caps minority-class precision (`0.19–0.41`), even when overall directional `Graded FScore` is high (`0.8173`).
