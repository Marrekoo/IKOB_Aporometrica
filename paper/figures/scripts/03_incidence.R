# 03_incidence.R ----------------------------------------------------------
# Who gains: gain per person by income decile for the price cuts and the hubs,
# citywide (S1 against S2c) and in the 15 buurten (S4 against S2t); the other
# scenarios in an appendix figure.
source("scripts/00_setup.R")
d <- load_cache()
section("Incidence by decile")

inc <- d$incidence |> filter(by == "income_class") |> rename(income_class = group)
band <- function(scens) {
  spec_band(filter(inc, scenario %in% scens), "gain_per_person", c("scenario", "income_class")) |>
    mutate(income_class = decile_f(income_class),
           scenario = factor(SCEN_LABEL[scenario], levels = SCEN_LABEL))
}
all <- band(names(SCEN_LABEL))
save_table(all |> arrange(scenario, spec, income_class) |>
             select(scenario, spec, income_class, gain_per_person, lo, hi),
           "gain_by_decile", "Gain per person by scenario, decile and specification", "Section 4.3",
           report = function(df) {
             u <- ifelse(df$spec == SPEC_LABEL[["m2"]],
                         precision_u(d, "gain_per_person", SCEN_CODE[as.character(df$scenario)],
                                     as.character(df$income_class)), NA)
             round_cols(df, c("gain_per_person", "lo", "hi"), u, floor = CFG$gain_floor)
           })

plot_gain <- function(df, title, ncol) {
  ggplot(df, aes(income_class, gain_per_person, colour = spec, group = spec)) +
    geom_ribbon(data = filter(df, spec == SPEC_LABEL[["m3"]]),
                aes(ymin = lo, ymax = hi, fill = spec), colour = NA, alpha = 0.25) +
    geom_line(linewidth = 0.8) + geom_point(size = 1.4) +
    scale_x_discrete(limits = DECILES, drop = FALSE) +
    facet_wrap(~scenario, ncol = ncol, scales = "free_y") +
    scale_colour_manual(values = PAL_SPEC, name = NULL) +
    scale_fill_manual(values = PAL_SPEC, guide = "none") +
    labs(title = title, x = "Income decile", y = "Gain in acceptable jobs per person", caption = paste0(CAP, precision_note(d, "gain_per_person"))) +
    guides(colour = guide_legend(ncol = 2))
}
# rows: citywide (S1 against S2c) and the 15 buurten (S4 against S2t); columns: price, hubs
main <- filter(all, scenario %in% SCEN_LABEL[c("s1", "s2c", "s4", "s2t")]) |>
  mutate(scenario = factor(as.character(scenario), SCEN_LABEL[c("s1", "s2c", "s4", "s2t")]))
save_plot(plot_gain(main, "Who gains: price cuts against extra hubs", 2),
          "fig_gain_price_vs_hubs", "Gain by decile: S1 against S2c (citywide), S4 against S2t (15 buurten)",
          "Section 4.3", w = 9, h = 7)
app <- filter(all, scenario %in% SCEN_LABEL[c("s1a", "s4a", "s3c", "s3t")]) |>
  mutate(scenario = factor(as.character(scenario), SCEN_LABEL[c("s1a", "s4a", "s3c", "s3t")]))
save_plot(plot_gain(app, "Who gains: targeted price cuts and combinations", 2),
          "fig_gain_other_scenarios", "Gain by decile: S1a, S4a, S3c, S3t", "Appendix", w = 9, h = 7)
