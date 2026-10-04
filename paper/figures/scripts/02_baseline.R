# 02_baseline.R -----------------------------------------------------------
# Baseline accessibility by income decile for M0, M1, M2 and the range of M3.
source("scripts/00_setup.R")
d <- load_cache()
section("Baseline accessibility by decile")

b <- spec_band(d$baseline, "accessibility", "income_class") |>
  mutate(income_class = decile_f(income_class))
save_table(b |> arrange(spec, income_class) |> select(spec, income_class, accessibility, lo, hi),
           "baseline_by_decile", "Baseline accessibility by decile and specification (jobs per person)",
           "Section 4.2",
           report = function(df) {
             u <- ifelse(df$spec == SPEC_LABEL[["m2"]], precision_u(d, "level", "s0", as.character(df$income_class)), NA)
             round_cols(df, c("accessibility", "lo", "hi"), u, CFG$level_digits)
           })

p <- ggplot(b, aes(income_class, accessibility / 1e3, colour = spec, group = spec)) +
  geom_ribbon(data = filter(b, spec == SPEC_LABEL[["m3"]]),
              aes(ymin = lo / 1e3, ymax = hi / 1e3, fill = spec), colour = NA, alpha = 0.25) +
  geom_line(linewidth = 0.9) + geom_point(size = 1.8) +
  scale_x_discrete(limits = DECILES, drop = FALSE) +
  scale_colour_manual(values = PAL_SPEC, name = NULL) +
  scale_fill_manual(values = PAL_SPEC, guide = "none") +
  labs(title = "Baseline accessibility by income decile",
       subtitle = "Public transport with shared bicycles; M3 as the range over theta",
       x = "Income decile", y = "Acceptable jobs per person (thousands)", caption = paste0(CAP, precision_note(d, "level"))) +
  guides(colour = guide_legend(ncol = 2))
save_plot(p, "fig_baseline_by_decile", "Baseline accessibility by decile and specification",
          "Section 4.2", w = 8, h = 5.5)
