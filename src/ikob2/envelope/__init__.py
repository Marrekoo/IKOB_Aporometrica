"""
The reference-budget envelope: from source tables to the per-journey money
budgets of `reference_budgets.csv`.

The method of the envelope script `X_M calc.R`, in stages that each take
tables and return tables:

  sources   the published inputs (Nibud basket, Warnaar anchors, CBS income
            percentiles, car costs), one CSV each with its source;
  odin      ODiN microdata -> aggregate tables (tour rates, household
            composition, commuting); the only stage that reads microdata;
  nibud     minimum and example baskets, the anchor residuals, the residual
            per household type, income decile and rent scenario;
  income    the income axis: the income of each decile from CBS percentiles;
  xm        journeys (or tours) per month, commuting costs and the scenario
            grid of X_M, the budget per journey (tour); per cell its range
            over gamma (or over the whole grid) is the envelope.

With the parameters of `defaults.toml` [envelope] the result is
`data/envelope/reference_budgets.csv`.
"""
