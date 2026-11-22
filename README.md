# What school phone bans measure, and what adolescents think of them

Analysis code and derived estimates for two manuscripts using the PISA 2022 international survey.

- **Paper A.** *What School Phone Bans Measure When the Measured Effect Is a System Response*
- **Paper B.** *Adolescents Endorse the Device Rule They Live Under, Independently of How Much They Use Their Phone*

Author: Sewon Min, Sahmyook University.

## Getting the data

The raw PISA 2022 files are public and are not redistributed here. Download them from the OECD file server, which requires no registration:

```
https://webfs.oecd.org/pisa2022/STU_QQQ_SPSS.zip   # student questionnaire, 651 MB
https://webfs.oecd.org/pisa2022/SCH_QQQ_SPSS.zip   # school questionnaire, 6 MB
```

Unzip both into `data/`. The scripts expect `data/CY08MSP_STU_QQQ.SAV` and `data/CY08MSP_SCH_QQQ.SAV`.

The student file is about 2 GB once unpacked. Every script reads only the columns it needs, so the analyses run in a few gigabytes of memory rather than loading the whole file.

## Code

| Script | Purpose |
|---|---|
| `code/03_brr.py` | pooled estimates with ten plausible values and eighty replicate weights |
| `code/04_country_brr.py` | country-specific estimates with the same variance treatment |
| `code/06b_fast.py` | robustness across domains and across six device policies |
| `code/07_paperB.py` | policy and attitude, with and without the control for own phone use |
| `code/09_paperB2.py` | dose response, convergent items, country heterogeneity |
| `code/08_reconcile.py` | checks that two specifications that should agree do agree |
| `code/05_figs.py`, `code/10_figsB.py` | figures |

Country comparisons use weighted within-country centring rather than indicator variables. The two are algebraically equivalent; the first is about forty times faster at this scale, which matters when each estimate requires 810 weighted fits.

## Variance estimation

Estimates follow the PISA convention. Each coefficient is computed for each plausible value and for the final weight; sampling variance comes from the eighty Fay replicate weights with a factor of 0.5; imputation variance is the between-plausible-value variance inflated by 1.1; the two are added.

This matters more than it might appear. Standard errors from a naive weighted regression that ignores the replicate structure were between five and six times too small in these models, and sampling variance exceeded imputation variance by roughly a factor of nine.

## Derived results

`results/` holds the country-level tables and the pooled estimates that the manuscripts report, so the figures and tables can be rebuilt without re-running the heavy estimation.

| File | Contents |
|---|---|
| `brr_pooled.json` | pooled estimates at four levels of adjustment |
| `supp.json` | three domains and six device policies |
| `country_brr.csv` | per-country achievement difference, standard error, compliance gap |
| `country_compliance.csv` | per-country difference in heavy phone use |
| `country_robust.csv` | per-country estimates under three control sets |
| `paperB.json`, `paperB2.json`, `paperB_country.csv` | the attitude analyses |

## Note on what these data can support

Every estimate here is cross-sectional. No causal claim is made for any of them; the argument of Paper A is precisely about what an observational association of this kind contains.

## Licence

MIT. See `LICENSE`.
