# 08_map.R ----------------------------------------------------------------
# The same 15 buurten, two interventions: half price for their residents (S4)
# against 12 extra hubs in them (S2t). Gain per resident (all segments, M2).
source("scripts/00_setup.R")
d <- load_cache()
section("Map: S4 against S2t")

RUN_LAB <- setNames(c("S4: Lime half price for residents of the 15 buurten",
                      "S2t: 12 extra hubs in the 15 buurten"),
                    c(CFG$map_runs[["price"]], CFG$map_runs[["hubs"]]))
m <- d$buurten |> inner_join(d$map_gain, by = "buurtcode") |>
  mutate(run = factor(RUN_LAB[run], RUN_LAB))
zone <- d$buurten |> filter(buurtcode %in% d$zone) |> summarise(geometry = st_union(geom))
save_table(st_drop_geometry(m) |> select(run, buurtcode, buurtnaam, population, gain) |> arrange(run, buurtcode),
           "map_gain_by_buurt", "Gain per resident by buurt: S4 and S2t (M2)", "Section 4.4",
           report = function(df) round_cols(df, "gain", floor = CFG$gain_floor))

p <- ggplot() +
  geom_sf(data = m, aes(fill = gain), colour = "white", linewidth = 0.1) +
  geom_sf(data = zone, fill = NA, colour = "black", linewidth = 0.6) +
  geom_sf(data = d$hubs, aes(shape = kind), size = 1.3, colour = "grey15") +
  facet_wrap(~run) +
  scale_fill_viridis_c(option = "magma", direction = -1, trans = "pseudo_log",
                       name = "Gain per resident\n(acceptable jobs)") +
  scale_shape_manual(values = c(existing = 1, s2t = 17), labels = c("existing hub", "S2t hub"), name = NULL) +
  labs(title = "Price by address against hubs in the same 15 buurten",
       subtitle = "Outline: Overvecht and Kanaleneiland/Transwijk; public transport with shared bicycles, M2",
       caption = CAP) +
  theme_void() + theme(legend.position = "right", strip.text = element_text(face = "bold"),
                       plot.title = element_text(face = "bold"))
save_plot(p, "fig_map_s4_s2t", "Map: gain per resident, S4 against S2t", "Section 4.4", w = 11, h = 5.5)
