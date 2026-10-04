# 06_targeting.R ----------------------------------------------------------
# Person-based (by income) against location-based (by address) price cuts, and the
# extra hubs: who is eligible, what it costs, who gains (M2).
source("scripts/00_setup.R")
d <- load_cache()
section("Targeting: by income, by address, hubs")

ORDER <- c("S1", "S1a", "S4", "S4a", "S2c", "S2t")
LAB <- c(S1 = "S1 everyone", S1a = "S1a D2-D4, citywide", S4 = "S4 residents of 15 buurten",
         S4a = "S4a D2-D4 residents of 15 buurten", S2c = "S2c 27 hubs, citywide",
         S2t = "S2t 12 hubs in 15 buurten")
t <- d$target |> filter(scenario %in% ORDER) |>
  mutate(scenario = factor(scenario, ORDER)) |> arrange(scenario) |>
  transmute(scenario, label = LAB[as.character(scenario)], hubs,
            target_eligible_share, eligible_outside_target_share,
            price_cost_eur_year, hub_cost_eur_year, cost_eur_year,
            cost_share_target, cost_share_zone,
            gain_job_persons, gain_per_eur, gain_share_target, gain_share_zone)
save_table(t, "targeting_summary",
           "Targeting: coverage of D2-D4, inclusion error, cost, gain, gain per euro (M2)", "Section 4.4",
           report = function(df) {
             code <- tolower(as.character(df$scenario))
             df |> mutate(across(c(target_eligible_share, eligible_outside_target_share, cost_share_target,
                                   cost_share_zone, gain_share_target, gain_share_zone), round_pct),
                          across(c(price_cost_eur_year, hub_cost_eur_year, cost_eur_year), ~ signif(.x, 2)),
                          gain_job_persons = round_report(gain_job_persons, precision_u(d, "gain_job_persons", code)),
                          gain_per_eur = round_report(gain_per_eur, precision_u(d, "gain_per_eur", code)))
           })

GROUP <- function(ic) case_when(ic == "D1" ~ "D1", ic %in% c("D2", "D3", "D4") ~ "D2-D4", TRUE ~ "D5-D10")
cells <- d$target_cells |> filter(scenario %in% c("S1a", "S4", "S2t")) |>
  mutate(group = GROUP(income_class)) |>
  group_by(scenario, group, place) |>
  summarise(gain_per_person = sum(gain) / sum(population), population = sum(population), .groups = "drop") |>
  mutate(scenario = factor(LAB[scenario], LAB[c("S1a", "S4", "S2t")]),
         place = factor(place, c("zone", "rest"), c("15 buurten", "rest of the city")),
         group = factor(group, c("D5-D10", "D2-D4", "D1")))
save_table(cells, "targeting_cells", "Gain per resident by income group and place (M2)", "Section 4.4",
           report = function(df) round_cols(df, "gain_per_person", floor = CFG$gain_floor) |> mutate(population = signif(population, 2)))

p <- ggplot(cells, aes(place, group, fill = gain_per_person)) +
  geom_tile(colour = "white", linewidth = 1) +
  geom_text(aes(label = format(round_report(gain_per_person, floor = CFG$gain_floor), big.mark = ",", trim = TRUE)), size = 3.6) +
  facet_wrap(~scenario) +
  scale_fill_gradient(low = "grey95", high = OI[["blue"]], trans = "sqrt", name = "Gain per resident") +
  labs(title = "Who gains: a price cut by income, by address, and extra hubs",
       subtitle = "Acceptable jobs per resident, M2; hubs serve journeys that end near them",
       x = NULL, y = NULL, caption = CAP) +
  theme(panel.grid = element_blank(), legend.position = "right")
save_plot(p, "fig_targeting_cells", "Gain per resident by income group and place: S1a, S4, S2t",
          "Section 4.4", w = 10, h = 4)
