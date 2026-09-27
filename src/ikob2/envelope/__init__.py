"""
The reference-budget envelope: from source tables to the per-tour money
budgets of `reference_budgets.csv`.

A Python re-derivation of `X_M calc.R`, in stages that each take tables and
return tables:

  sources   the published inputs (Nibud basket, Warnaar anchors, CBS income
            percentiles, car costs), one CSV each with its source;
  odin      ODiN microdata -> aggregate tables (tour rates, household
            composition, commuting); the only stage that reads microdata;
  nibud     minimum and example baskets, the anchor residuals, the residual
            per household type, income decile and rent scenario;
  income    the income axis: the income of each decile from CBS percentiles;
  xm        tours per month, commuting costs and the scenario grid of X_M
            (EUR per home-based tour); its minimum and maximum per cell are
            the envelope.

With the parameters of `defaults.toml` [envelope] the result reproduces
`data/envelope/reference_budgets.csv`.
"""
