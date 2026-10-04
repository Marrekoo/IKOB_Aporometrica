# 07_gap.R ----------------------------------------------------------------
# The reachability gap: jobs that pass the time gate and fail the money gate, as a
# share of the time-acceptable jobs, by decile, at baseline and after the price cuts.
source("scripts/00_setup.R")
d <- load_cache()
section("Reachability gap")

GAP_SCEN <- c(s0 = "S0 baseline", s1 = "S1 everyone", s1a = "S1a D2-D4", s4 = "S4 15 buurten",
              s4a = "S4a D2-D4 in 15 buurten")
g <- d$gap |> filter(spec %in% c("m2", CFG$m3), scenario %in% names(GAP_SCEN))
gb <- spec_band(g, "gap_share", c("scenario", "income_class")) |>
  mutate(income_class = decile_f(income_class), scenario = factor(GAP_SCEN[scenario], GAP_SCEN))
save_table(gb |> select(scenario, spec, income_class, gap_share, lo, hi) |> arrange(scenario, spec, income_class),
           "gap_share_by_decile", "Reachability gap as a share of the time-acceptable jobs (M2, M3 range)",
           "Section 4.5",
           report = function(df) mutate(df, across(c(gap_share, lo, hi), round_pct)))

# (a) the gap at baseline; (b) the share of it that each price cut closes
g0 <- filter(gb, scenario == GAP_SCEN[["s0"]])
pa <- ggplot(g0, aes(income_class, gap_share, colour = spec, group = spec)) +
  geom_ribbon(data = filter(g0, spec == SPEC_LABEL[["m3"]]), aes(ymin = lo, ymax = hi, fill = spec),
              colour = NA, alpha = 0.25) +
  geom_line(linewidth = 0.8) + geom_point(size = 1.6) +
  scale_x_discrete(limits = DECILES, drop = FALSE) +
  scale_colour_manual(values = PAL_SPEC, name = NULL) + scale_fill_manual(values = PAL_SPEC, guide = "none") +
  scale_y_continuous(labels = percent) +
  labs(title = "(a) The gap at baseline", subtitle = "Share of the time-acceptable jobs that fail the money gate",
       x = "Income decile", y = "Gap")
closed <- g |> filter(spec == "m2") |> select(scenario, income_class, gap) |>
  left_join(g |> filter(spec == "m2", scenario == "s0") |> select(income_class, gap0 = gap), by = "income_class") |>
  filter(scenario != "s0", income_class %in% c("D2", "D3", "D4", "D5")) |>
  mutate(closed = 1 - gap / gap0, scenario = factor(GAP_SCEN[scenario], GAP_SCEN),
         income_class = decile_f(income_class))
save_table(closed |> arrange(scenario, income_class), "gap_closed_by_price_cuts",
           "Share of the baseline gap closed by each price cut, D2-D5 (M2)", "Section 4.5",
           report = function(df) mutate(df, across(c(gap, gap0), ~ signif(.x, 2)), closed = round_pct(closed)))
pb <- ggplot(closed, aes(income_class, closed, fill = scenario)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.75) +
  scale_fill_manual(values = setNames(c(OI[["vermillion"]], OI[["orange"]], OI[["purple"]], "#40004B"),
                                      GAP_SCEN[-1]), name = NULL) +
  scale_y_continuous(labels = percent) +
  labs(title = "(b) Share of the gap closed by halving the Lime price (M2)",
       subtitle = "The gap is mostly the public transport fare: a cheaper shared bicycle closes little of it",
       x = "Income decile", y = "Share of the baseline gap closed", caption = CAP) +
  guides(fill = guide_legend(nrow = 2))
p <- pa / pb + plot_layout(heights = c(1, 1))
save_plot(p, "fig_gap_by_decile", "Reachability gap at baseline and the share closed by the price cuts",
          "Section 4.5", w = 8, h = 9)
