# 04_agreement.R ----------------------------------------------------------
# Do the specifications agree with M2? Correlation across origin x segment cells
# (decile 1 excluded) of the baseline levels and of the gains of each scenario.
source("scripts/00_setup.R")
d <- load_cache()
section("Agreement between specifications")

measures <- c(levels = "Baseline levels", gain_s1 = "Gain S1", gain_s1a = "Gain S1a",
              gain_s4 = "Gain S4", gain_s2c = "Gain S2c", gain_s2t = "Gain S2t")
cor <- d$correlation |>
  filter(method == "pearson", spec_b == "m2", spec_a != "m2", measure %in% names(measures)) |>
  rename(spec = spec_a)
cb <- spec_band(cor, "correlation", "measure") |>
  mutate(measure = factor(measures[measure], levels = measures))
save_table(cb |> arrange(measure, spec) |> select(measure, spec, correlation, lo, hi),
           "correlation_with_m2", "Pearson correlation with M2 across origin x segment cells", "Section 4.2-4.3",
           report = function(df) mutate(df, across(c(correlation, lo, hi), ~ round(.x, 2))))

# one row per measure, the specifications side by side within it
off <- c(-0.22, 0, 0.22, 0); names(off) <- c(SPEC_LABEL[["m0"]], SPEC_LABEL[["m1"]], SPEC_LABEL[["m3"]], SPEC_LABEL[["m2"]])
cb <- cb |> mutate(y = length(measures) + 1 - as.integer(measure) + off[as.character(spec)])
p <- ggplot(cb, aes(correlation, y, colour = spec)) +
  geom_vline(xintercept = 0, colour = "grey60") +
  geom_errorbar(data = filter(cb, spec == SPEC_LABEL[["m3"]]), aes(xmin = lo, xmax = hi),
                orientation = "y", width = 0.12) +
  geom_point(size = 2.6) +
  scale_colour_manual(values = PAL_SPEC[SPEC_LABEL[c("m0", "m1", "m3")]], breaks = unname(SPEC_LABEL[c("m0", "m1", "m3")]), name = NULL) +
  scale_y_continuous(breaks = rev(seq_along(measures)), labels = unname(measures), minor_breaks = NULL) +
  coord_cartesian(xlim = c(-0.2, 1)) +
  labs(title = "Agreement with M2: levels agree, gains of price cuts do not",
       subtitle = "Pearson correlation across origin x segment cells, decile 1 excluded; M3 as the range over theta",
       x = "Correlation with M2", y = NULL, caption = CAP) +
  guides(colour = guide_legend(ncol = 1))
save_plot(p, "fig_agreement_with_m2", "Correlation with M2 of levels and gains", "Section 4.2-4.3",
          w = 8, h = 5)
