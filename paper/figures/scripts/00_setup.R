# 00_setup.R -------------------------------------------------------------
# Configuration, packages, labels, theme and helpers shared by all scripts.
# Every script starts with `source("scripts/00_setup.R")` (run_all.R does this).
#
# The statistics come from the Python model (IKOB Aporometrica): its paper tables in
# <data root>/outputs/comparisons/specs and .../targeting (python -m ikob2.cli.paper_tables)
# and, for the map, three scenario runs. This pipeline only selects, reshapes and plots them.

# ---- packages -------------------------------------------------------------
required <- c("dplyr", "tidyr", "readr", "purrr", "stringr", "tibble", "ggplot2",
              "scales", "sf", "patchwork", "jsonlite")
missing_req <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_req)) {
  stop("Install the missing packages first: install.packages(c(",
       paste0('"', missing_req, '"', collapse = ", "), "))", call. = FALSE)
}
suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(readr); library(purrr); library(stringr)
  library(tibble); library(ggplot2); library(scales); library(sf); library(patchwork)
  library(jsonlite)
})
options(bitmapType = "cairo", dplyr.summarise.inform = FALSE, readr.show_col_types = FALSE)

# ---- CONFIG ---------------------------------------------------------------
data_root <- Sys.getenv("IKOB_DATA_ROOT")
if (!nzchar(data_root))
  stop("Set IKOB_DATA_ROOT to the data folder, e.g. in ~/.Renviron: IKOB_DATA_ROOT=/path/to/data",
       call. = FALSE)
CFG <- list(
  data_root = data_root,
  # figures, tables, cache and logs; default <data root>/outputs/paper_figures
  out       = Sys.getenv("IKOB_FIGURES_OUT", file.path(data_root, "outputs", "paper_figures")),
  # specifications of the main text; M3 is shown as the range over its theta values
  specs     = c("m0", "m1", "m2"),
  m3        = c("m3t1.25", "m3t1.5", "m3t2", "m3t4", "m3tinf"),
  # scenario runs of the map (python run plan paper/runs.toml)
  map_runs  = c(base = "scen_s0", price = "scen_s4", hubs = "scen_s2t"),
  zone_file = "lime_price_zones_overvecht_kanaleneiland.csv",   # under inputs/tariffs
  hub_files = c(existing = "inputs/hubs/utrecht_hubs.csv",
                s2t = "intermediate/hubs/utrecht_hubs_s2t.csv"),
  mode      = "pt_v2",                                          # PT chains with shared bicycles
  # reporting (the review on significant figures): at most this many significant digits in
  # the tables for the paper, and never more than the data support (precision.csv)
  display_digits = 2,
  level_digits   = 3,                                           # accessibility levels (jobs)
  # resolution: a gain below half a job per person, or a share below 0.1%, is reported as 0
  gain_floor     = 0.5,
  pct_floor      = 0.1
)
CFG$runs_dir <- file.path(CFG$data_root, "outputs", "runs")
CFG$specs_dir <- file.path(CFG$data_root, "outputs", "comparisons", "specs")
CFG$target_dir <- file.path(CFG$data_root, "outputs", "comparisons", "targeting")
CFG$precision_file <- file.path(CFG$data_root, "outputs", "comparisons", "precision", "precision.csv")
for (d in c("figures", "tables", "tables_paper", "cache", "logs"))
  dir.create(file.path(CFG$out, d), recursive = TRUE, showWarnings = FALSE)
if (!dir.exists(CFG$specs_dir))
  stop("No paper tables in ", CFG$specs_dir, ": run `python -m ikob2.cli.paper_tables` ",
       "or set IKOB_DATA_ROOT.", call. = FALSE)

# ---- labels and palettes --------------------------------------------------
DECILES <- paste0("D", 1:10)
SPEC_LABEL <- c(m0 = "M0 (cut-off on generalised time)", m1 = "M1 (exponential, one value of time)",
                m2 = "M2 (gates, independent)", m3 = "M3 (gates, dependent: theta 1.25 to inf)")
SCEN_LABEL <- c(s1 = "S1 Lime half price, everyone", s1a = "S1a Lime half price, D2-D4",
                s2c = "S2c 27 extra hubs, citywide", s2t = "S2t 12 extra hubs, 15 buurten",
                s3c = "S3c S1 + S2c", s3t = "S3t S1 + S2t",
                s4 = "S4 Lime half price, residents of 15 buurten",
                s4a = "S4a Lime half price, D2-D4 residents of 15 buurten")
SCEN_CODE <- setNames(names(SCEN_LABEL), SCEN_LABEL)
# Okabe-Ito, colour-blind safe
OI <- c(black = "#000000", orange = "#E69F00", sky = "#56B4E9", green = "#009E73",
        yellow = "#F0E442", blue = "#0072B2", vermillion = "#D55E00", purple = "#CC79A7")
PAL_SPEC <- setNames(c("#7F7F7F", OI[["orange"]], OI[["blue"]], OI[["vermillion"]]), SPEC_LABEL)
PAL_SCEN <- setNames(c(OI[["vermillion"]], OI[["orange"]], OI[["green"]], OI[["sky"]],
                       "#B2182B", OI[["blue"]], OI[["purple"]], "#40004B"), SCEN_LABEL)

theme_sb <- function(base_size = 11) {
  theme_minimal(base_size = base_size) +
    theme(plot.title = element_text(face = "bold", size = base_size + 2),
          plot.subtitle = element_text(colour = "grey30"),
          plot.caption = element_text(colour = "grey40", size = base_size - 3, hjust = 0),
          panel.grid.minor = element_blank(), strip.text = element_text(face = "bold"),
          legend.position = "bottom", legend.title = element_text(face = "bold"))
}
theme_set(theme_sb())
CAP <- "Source: IKOB Aporometrica paper runs (paper/runs.toml); 111 origin buurten, municipality of Utrecht"

# ---- helpers ---------------------------------------------------------------
INDEX <- file.path(CFG$out, "cache", "index.csv")
register <- function(file, title, paper) {
  row <- tibble(file = file, title = title, paper = paper)
  old <- if (file.exists(INDEX)) read_csv(INDEX, col_types = "ccc") else row[0, ]
  write_csv(bind_rows(filter(old, file != !!file), row), INDEX)
}
save_plot <- function(p, name, title, paper, w = 8, h = 5) {
  base <- file.path(CFG$out, "figures", name)
  ggsave(paste0(base, ".png"), p, width = w, height = h, dpi = 300, bg = "white")
  ggsave(paste0(base, ".pdf"), p, width = w, height = h,
         device = if (capabilities("cairo")) cairo_pdf else "pdf")
  register(file.path("figures", paste0(name, ".png")), title, paper)
  message("  figure: ", name)
  invisible(p)
}
# Tables: `df` at full precision in <out>/tables; `report(df)`, the rounded
# version for the paper, in <out>/tables_paper.
save_table <- function(df, name, title, paper, report = NULL) {
  write_csv(df, file.path(CFG$out, "tables", paste0(name, ".csv")))
  register(file.path("tables", paste0(name, ".csv")), title, paper)
  if (!is.null(report)) {
    write_csv(report(df), file.path(CFG$out, "tables_paper", paste0(name, ".csv")), na = "")
    register(file.path("tables_paper", paste0(name, ".csv")), paste(title, "(rounded for the paper)"), paper)
  }
  message("  table: ", name, " (", nrow(df), " rows)")
  invisible(df)
}

# ---- rounding (the review on significant figures) ---------------------------
# Significant digits of x with standard uncertainty u: the last digit at the leading digit
# of u (JCGM 100:2008, 7.2.6); NA where u is unknown or zero.
digits_from_u <- function(x, u) {
  ok <- is.finite(x) & is.finite(u) & u > 0 & x != 0
  out <- rep(NA_real_, length(x))
  out[ok] <- pmax(1, floor(log10(abs(x[ok]))) - floor(log10(u[ok])) + 1)
  out
}
# Report x with at most CFG$display_digits significant digits (two effective digits for
# reading) and no more than its uncertainty u supports.
round_report <- function(x, u = NA_real_, n_display = CFG$display_digits, floor = 0) {
  u <- rep_len(u, length(x))
  n <- digits_from_u(x, u)
  n <- ifelse(is.na(n), n_display, pmin(n, n_display))
  ifelse(abs(x) < floor, 0, signif(x, n))
}
# Shares as percentages: whole percents, two significant digits below 10%, 0 below the floor.
round_pct <- function(x) {
  p <- 100 * x
  ifelse(abs(p) < CFG$pct_floor, 0, ifelse(abs(p) < 10, signif(p, 2), round(p)))
}
# Data-rounding uncertainty (sd over the perturbed M2 runs) of a statistic per scenario code
# and income class; NA where the perturbed runs do not cover it. Vectorised.
precision_u <- function(d, statistic, scenario, income_class = "all") {
  n <- max(length(scenario), length(income_class))
  if (is.null(d$precision)) return(rep(NA_real_, n))
  p <- d$precision[d$precision$statistic == statistic, ]
  unname(setNames(p$sd, paste(p$scenario, p$income_class))[paste(scenario, income_class)])
}
# Round columns `cols` of df with round_report; `u` only for the M2 rows.
round_cols <- function(df, cols, u = NA_real_, n_display = CFG$display_digits, floor = 0) {
  df |> mutate(across(all_of(cols), ~ round_report(.x, u, n_display, floor)))
}
# Caption note: the largest relative spread of a statistic over the perturbed runs.
precision_note <- function(d, statistic) {
  if (is.null(d$precision)) return("")
  p <- d$precision |> filter(.data$statistic == !!statistic, is.finite(rel_sd))
  if (nrow(p) == 0) return("")
  sprintf("\nRounding of the input data moves these values by at most %s%% (sd over %d draws).",
          format(signif(100 * max(p$rel_sd), 1), scientific = FALSE), max(p$draws))
}
CACHE <- file.path(CFG$out, "cache", "data.rds")
load_cache <- function() {
  if (!file.exists(CACHE)) stop("Run scripts/01_load.R first.", call. = FALSE)
  readRDS(CACHE)
}
decile_f <- function(x) factor(x, levels = DECILES)
section <- function(txt) message("\n== ", txt)
# one row per specification of the main text: M0, M1, M2 as they are, M3 as min and max over theta
spec_band <- function(df, value, by) {
  main <- df |> filter(spec %in% CFG$specs) |>
    mutate(spec = factor(SPEC_LABEL[spec], levels = SPEC_LABEL), lo = .data[[value]], hi = .data[[value]])
  m3 <- df |> filter(spec %in% CFG$m3) |> group_by(across(all_of(by))) |>
    summarise(lo = min(.data[[value]]), hi = max(.data[[value]]), mid = median(.data[[value]]),
              .groups = "drop") |>
    mutate(spec = factor(SPEC_LABEL[["m3"]], levels = SPEC_LABEL), !!value := mid) |> select(-mid)
  bind_rows(main, m3)
}
message("Setup OK: data ", CFG$data_root, " | output ", CFG$out)
