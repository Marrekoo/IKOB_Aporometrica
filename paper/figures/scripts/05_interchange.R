# 05_interchange.R --------------------------------------------------------
# The interchangeability ratio R = gain(price cut) / gain(hubs) for S1/S2c
# (citywide) and S4/S2t (15 buurten). R is bimodal across segments (0 where the
# price cut gives nothing, large where it opens the money gate), so it is
# reported as: its dispersion across the segments of an origin (CV), the share
# of segments where the price cut adds nothing, and by decile.
source("scripts/00_setup.R")
d <- load_cache()
section("The interchangeability ratio R")

PAIR <- c("s1/s2c" = "S1 / S2c (citywide)", "s4/s2t" = "S4 / S2t (15 buurten)")
keep <- c(CFG$specs, CFG$m3)
summ <- d$r_summary |> filter(spec %in% keep) |>
  transmute(pair = PAIR[pair], comparison, spec, pooled_cv, share_a_zero, pooled_median_R,
            pairs, pairs_undefined)
save_table(summ, "R_summary", "R: CV across segments, share where the price cut adds nothing, undefined pairs",
           "Section 4.1",
           report = function(df) mutate(df, pooled_cv = round(pooled_cv, 2), share_a_zero = round_pct(share_a_zero),
                                        pooled_median_R = signif(pooled_median_R, 2)))

cls <- d$r_class |> filter(spec %in% keep, comparison == "full") |>
  mutate(pair = factor(PAIR[pair], levels = PAIR))
save_table(cls |> select(pair, spec, income_class, share_a_zero, share_b_zero, median_R, median_R_both),
           "R_by_decile", "R by decile: shares of segments where one intervention adds nothing, medians",
           "Section 4.1",
           report = function(df) mutate(df, across(c(share_a_zero, share_b_zero), round_pct),
                                        across(c(median_R, median_R_both), ~ signif(.x, 2))))

cv <- spec_band(d$r_summary |> filter(spec %in% keep), "pooled_cv", c("pair", "comparison")) |>
  mutate(pair = factor(PAIR[pair], levels = PAIR),
         comparison = factor(comparison, c("controlled", "full"), c("Controlled", "Full")))
p1 <- ggplot(cv, aes(pooled_cv, spec, colour = spec)) +
  geom_errorbar(orientation = "y", data = filter(cv, spec == SPEC_LABEL[["m3"]]), aes(xmin = lo, xmax = hi), width = 0.3) +
  geom_point(size = 2.6) + facet_grid(comparison ~ pair) +
  scale_colour_manual(values = PAL_SPEC, guide = "none") +
  scale_y_discrete(limits = rev(unname(SPEC_LABEL))) +
  labs(title = "Dispersion of R across the segments of an origin",
       subtitle = "CV of R pooled over origins; 0: one ratio for all segments.\nControlled: common jobs and costs; full: the application",
       x = "CV of R", y = NULL)
# decile 1 has no segment where the hubs add something: its share is undefined
z <- spec_band(filter(cls, !is.na(share_a_zero), income_class != "D1"), "share_a_zero", c("pair", "income_class")) |>
  mutate(income_class = decile_f(income_class))
p2 <- ggplot(z, aes(income_class, share_a_zero, colour = spec, group = spec)) +
  geom_ribbon(data = filter(z, spec == SPEC_LABEL[["m3"]]), aes(ymin = lo, ymax = hi, fill = spec),
              colour = NA, alpha = 0.25) +
  geom_line(linewidth = 0.8) + geom_point(size = 1.4) + facet_wrap(~pair) +
  scale_x_discrete(limits = DECILES[-1]) +
  scale_colour_manual(values = PAL_SPEC, name = NULL) + scale_fill_manual(values = PAL_SPEC, guide = "none") +
  scale_y_continuous(labels = percent) +
  labs(title = "Where the price cut adds nothing (R = 0)",
       subtitle = "Share of segments with a hub gain but no gain from the price cut\n(full application; decile 1 has no hub gain)",
       x = "Income decile", y = "Share of segments", caption = CAP) +
  guides(colour = guide_legend(ncol = 2))
save_plot(p1 / p2 + plot_layout(heights = c(1, 1.2)), "fig_R", "R: dispersion across segments and zero shares by decile",
          "Section 4.1", w = 9, h = 9)
