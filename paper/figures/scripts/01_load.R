# 01_load.R --------------------------------------------------------------
# Reads the model's paper tables and the inputs of the map, and caches them.
source("scripts/00_setup.R")
section("Loading the model's paper tables")

rd <- function(dir, f) {
  p <- file.path(dir, f)
  if (!file.exists(p)) stop("Missing ", p, ": run `python -m ikob2.cli.paper_tables`.", call. = FALSE)
  read_csv(p)
}
d <- list(
  baseline    = rd(CFG$specs_dir, "baseline_by_spec.csv"),
  incidence   = rd(CFG$specs_dir, "incidence_by_spec.csv"),
  correlation = rd(CFG$specs_dir, "correlation_by_spec.csv"),
  r_summary   = rd(CFG$specs_dir, "interchange_by_spec.csv"),
  r_class     = rd(CFG$specs_dir, "interchange_by_class.csv"),
  gap         = rd(CFG$specs_dir, "gap_by_spec.csv"),
  target      = rd(CFG$target_dir, "targeting_summary.csv"),
  target_cells = rd(CFG$target_dir, "targeting_cells.csv")
)

d$precision <- if (file.exists(CFG$precision_file)) read_csv(CFG$precision_file) else NULL
message(if (is.null(d$precision)) "  no precision table yet (python -m ikob2.cli.precision): display rounding only"
        else paste("  precision table:", nrow(d$precision), "statistics"))

section("Loading the map inputs")
run_file <- function(run, f) {
  p <- file.path(CFG$runs_dir, run, f)
  if (!file.exists(p)) stop("Missing ", p, call. = FALSE)
  p
}
d$buurten <- st_read(run_file(CFG$map_runs[["base"]], "origins.gpkg"), layer = "origins", quiet = TRUE) |>
  select(buurtcode, buurtnaam, population)
acc <- function(run) {
  read_csv(run_file(run, "accessibility.csv")) |>
    filter(mode == CFG$mode) |> select(buurtcode, segment, population, accessibility)
}
base <- acc(CFG$map_runs[["base"]])
d$map_gain <- map_dfr(c("price", "hubs"), function(k) {
  acc(CFG$map_runs[[k]]) |>
    inner_join(select(base, buurtcode, segment, a0 = accessibility), by = c("buurtcode", "segment")) |>
    group_by(buurtcode) |>
    summarise(gain = sum((accessibility - a0) * population) / sum(population), .groups = "drop") |>
    mutate(run = CFG$map_runs[[k]])
})
d$zone <- read_csv(file.path(CFG$data_root, "inputs", "tariffs", CFG$zone_file),
                   col_types = cols(buurtcode = "c"))$buurtcode
d$hubs <- imap_dfr(CFG$hub_files, function(f, kind) {
  read_csv(file.path(CFG$data_root, f)) |> select(hub, lat, lon) |> mutate(kind = kind)
}) |> st_as_sf(coords = c("lon", "lat"), crs = 4326) |> st_transform(st_crs(d$buurten))

saveRDS(d, CACHE)
message("Cached ", length(d), " objects to ", CACHE)
