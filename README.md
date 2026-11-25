# What school phone bans measure, and what adolescents think of them

Analysis code and derived estimates for two manuscripts using the PISA 2022 international survey.

- **Paper A.** *What School Phone Bans Measure When the Measured Effect Is a System Response*
- **Paper B.** *Adolescents in Schools That Ban Phones Endorse Device Restriction Specifically and Not Rules in General*

Author: Sewon Min, Sahmyook University.

Every number in either manuscript can be re-derived from the public PISA files with the scripts here. If you find one that cannot, please open an issue.

## Getting the data

The raw PISA 2022 files are public and are not redistributed here. Download them from the OECD file server, which requires no registration:

```
https://webfs.oecd.org/pisa2022/STU_QQQ_SPSS.zip   # student questionnaire, 651 MB
https://webfs.oecd.org/pisa2022/SCH_QQQ_SPSS.zip   # school questionnaire, 6 MB
```

Unzip both into `data/`. The scripts expect `data/CY08MSP_STU_QQQ.SAV` and `data/CY08MSP_SCH_QQQ.SAV`.

The student file is about 2 GB once unpacked. Every script reads only the columns it needs, so the analyses run in a few gigabytes of memory rather than loading the whole file.

Requirements: Python 3.11 with `pandas`, `numpy`, `pyreadstat` and, for the figures, `matplotlib`.

The scripts read and write a working directory called `out/`, and later scripts read what earlier ones wrote there. The released copies of those files are in `results/`, so to run a script without re-running the whole chain, copy or link them first:

```
mkdir -p out figs && cp results/* out/
```

## Code

Run in this order. `12_fix.py` produces the main estimates for both papers on one consistent sample; everything after it is a separate question or a test.

| Script | Purpose | Feeds |
|---|---|---|
| `code/12_fix.py` | the adjustment ladder, three domains, six device policies singly and mutually adjusted, and the Paper B main models, all on one sample | A Tables 1–4, B Tables 1 and 3 |
| `code/13_country.py` | country-specific estimates, the compliance gap pooled and within country | A Section 4.1, 4.3 |
| `code/14_dose.py` | regulation count, observed adjusted means by rule count | B Section 4.2 |
| `code/15_dose2.py` | the three tests that dismantle the dose response | B Section 4.2 |
| `code/30_identify.py` | Oster bound, equivalence test, split-school meta-regression, country inclusion audit, the six device items jointly, person-centred outcomes, the discretion moderator | A Sections 3.2, 4.2, 4.4; B Sections 4.3, 4.4 |
| `code/31_acq.py` | the placebo batteries | B Section 4.3, Table S7 |
| `code/11_extra.py` | supporting estimates referenced in the supplementary material | both |
| `code/20_figsA.py`, `code/21_figsB.py`, `code/32_figsB34.py` | figures | both |

Country comparisons use weighted within-country centring rather than indicator variables. The two are algebraically equivalent; the first is about forty times faster at this scale, which matters when each estimate requires 810 weighted fits.

## Variance estimation

Estimates follow the PISA convention. Each coefficient is computed for each plausible value and for the final weight; sampling variance comes from the eighty Fay replicate weights with a factor of 0.5, giving a divisor of twenty; imputation variance is the between-plausible-value variance inflated by 1.1; the two are added. Sampling variance exceeds imputation variance by roughly a factor of nine in these models.

Paper B's outcomes are single survey items rather than plausible values, so there is no imputation component and the whole standard error is sampling variance.

Two-sided p values are computed with `math.erfc`, not `1 - norm.cdf`, which underflows to exactly zero above about |z| = 8.2. Several estimates here exceed that.

## Three coding rules that change the answer

1. **A blank policy answer is not a "no".** `(col == 1).astype(float)` sends `NaN` to `0.0`, which silently moves about 1,750 non-responding schools into the comparison group. For one of the six policy items that would have exceeded the number of genuine negative answers. Missing stays missing throughout; see `yesno()` in `12_fix.py`.
2. **A count is missing if any of its parts is.** `df[cols].sum(axis=1)` treats a missing item as a zero; `min_count=6` is what makes the rule count missing instead.
3. **Use the same heterogeneity estimator on both sides of an R-squared.** Mixing a DerSimonian-Laird unconditional variance with a method-of-moments residual variance inflates the proportion of between-country variance explained. Both are method-of-moments here.

See `CHANGELOG.md` for what these corrected and by how much.

## Results

`results/` holds the derived estimates, which are small enough to read directly.

| File | Contents |
|---|---|
| `fixed_20261124.json` | the main estimates for both papers |
| `country_fixed.json`, `country_fixed.csv` | country-specific achievement estimates, compliance |
| `compliance_fixed.csv` | the compliance gap by country |
| `metareg_fixed.json` | the meta-regression of the achievement gap on the compliance gap |
| `paperB_country_brr.csv` | country-specific endorsement estimates with replicate standard errors |
| `dose_20261124.json`, `dose2_20261124.json` | the regulation count, and the tests that dismantle it |
| `ident_20261125.json` | Oster, equivalence, split-school refits, inclusion audit, six items, person-centred, discretion |
| `acq_20261125.json` | every placebo item with its estimate |
| `extra_20261124.json` | supporting estimates |

## Licence

MIT, see `LICENSE`. The PISA data are the OECD's and are governed by the OECD's own terms.
