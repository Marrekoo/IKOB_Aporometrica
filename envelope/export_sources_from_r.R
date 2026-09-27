# Provenance of data/envelope/sources/: writes the literal tables of X_M calc.R
# (run in the folder of that script; writes sources/). The repository tables
# add source and note columns; their numbers are these, unchanged.
suppressPackageStartupMessages(library(tidyverse))
src <- readLines("X_M calc.R")
grab <- function(start_rx, stop_rx) {
  i <- grep(start_rx, src)[1]; j <- i + grep(stop_rx, src[(i+1):length(src)])[1]
  txt <- sub("\\|>\\s*$", "", paste(src[i:j], collapse = "\n")); eval(parse(text = txt), envir = globalenv())
}
CFG <- list(use_table16_for_cc_wml = TRUE)
grab("^HHSAM_MAP <- tribble", "^  8L, NA_character_")
grab("^EQV_CBS <- c", "couple_children = 1.91\\)")
grab("^MVB_HH <- tribble", "\"wlz_aow\"")
grab("^POST_META <- tribble", "\"vrijetijd_sp\"")
grab("^MVB_WIDE <- tribble", "\"vrijetijd_sp\"")
grab("^NIBUD_ANCHORS <- tribble", "price_base = \"warnaar\"\\)")
grab("^CBS_PCTL_RAW <- tribble", "\"disp\", 2024L")
grab("^BUNDLES_BASE <- tribble", "\"motor\"")
o <- "sources"; dir.create(o, showWarnings = FALSE)
write_csv(HHSAM_MAP, file.path(o, "odin_household_types.csv"), na = "")
write_csv(tibble(hh_type = names(EQV_CBS), factor = unname(EQV_CBS)), file.path(o, "equivalence_cbs.csv"))
write_csv(MVB_HH, file.path(o, "nibud_households.csv"))
write_csv(POST_META, file.path(o, "nibud_posts.csv"))
write_csv(MVB_WIDE |> pivot_longer(-post, names_to = "hh_key", values_to = "eur"), file.path(o, "nibud_basket.csv"), na = "")
write_csv(NIBUD_ANCHORS |> select(-price_base), file.path(o, "warnaar_anchors.csv"))
write_csv(CBS_PCTL_RAW |> pivot_longer(matches("^p[0-9]+$"), names_to = "p", values_to = "k_eur_year") |>
            mutate(p = as.numeric(sub("^p","",p))/100), file.path(o, "cbs_income_percentiles.csv"))
write_csv(BUNDLES_BASE, file.path(o, "car_bundles.csv"))
cat("ok\n")
