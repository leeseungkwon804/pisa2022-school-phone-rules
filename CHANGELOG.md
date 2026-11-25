# Changelog

## 2026-11-25

Three coding errors were found in the analysis published here on 22 November 2026 and are corrected in this release. All three were found before submission; no manuscript was submitted on the earlier numbers. The direction and the conclusions of both papers are unchanged, but several figures moved and one claim in Paper B was withdrawn.

### 1. Missing policy answers were counted as "no"

`(col == 1).astype(float)` evaluates `NaN == 1` as `False` and returns `0.0`. Principals who left a device-policy item blank were therefore counted as principals reporting that the policy was absent. About 1,750 schools were affected. For `SC190Q06JA` the genuine "no" group is 1,417 schools, so the fabricated negatives outnumbered the real ones.

Present in `03_brr.py`, `04_country_brr.py`, `06b_fast.py`, `07_paperB.py`, `08_reconcile.py` and `09_paperB2.py`, all of which are superseded by `12_fix.py`.

Effect: the analytic sample falls from 264,770 to 243,148 students for Paper A and from 245,533 to 229,966 for Paper B, and the country count from 48 to 45. Point estimates move by a few hundredths to a few tenths of a point. The meta-regression of the achievement gap on the compliance gap moves from a slope of 0.457 to 0.506.

### 2. Two-sided p values underflowed to exactly zero

`2*(1 - scipy.stats.norm.cdf(abs(z)))` returns `0.0` for |z| above about 8.2, because `norm.cdf` reaches 1.0 in double precision. Paper B's main table contained a reported p of 0, where the correct value is 1.8e-37. Replaced throughout by `math.erfc(abs(z)/sqrt(2))`.

### 3. The chi-square upper tail was wrong above the series or continued-fraction crossover

The continued-fraction branch of the regularised incomplete gamma function initialised `c` to `FPMIN` instead of `1/FPMIN`, returning values of order 1e297 instead of a probability. This affected one printed statistic, the Wald test that all six device-regulation items carry the same coefficient, which was reported as `p = 5.6e+297` and interpreted as "not rejected". The correct value is chi-square 12.71 on 5 degrees of freedom, p = 0.026, and the hypothesis of a uniform shift **is** rejected. Corrected in `30_identify.py`; the test is now verified against four known values at import.

### Separately: a claim withdrawn

The 22 November version of Paper B reported that endorsement rises with the number of device rules a school imposes, and read that as a dose response. It does not survive inspection, and the manuscript now reports the null as a finding. Holding the ban constant, the five other rules give 0.003 points per rule, standard error 0.005, p = 0.45; the departure from linearity is significant, Wald 24.99 on 5 degrees of freedom, p = 0.00014; and the same null holds inside ban schools and inside no-ban schools separately. The apparent gradient is the ban, which correlates with the rule count at 0.41. See `15_dose2.py`.
