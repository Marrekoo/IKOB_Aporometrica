# 09_index.R --------------------------------------------------------------
# An index of every figure and table, with the section of the paper it belongs to.
source("scripts/00_setup.R")
section("Index")
idx <- read_csv(INDEX, col_types = "ccc") |> arrange(paper, file)
lines <- c("# Figures and tables of the paper", "",
           paste0("Made ", format(Sys.time(), "%Y-%m-%d %H:%M"), " from ", CFG$data_root, "."), "",
           "| Paper | File | Content |", "|---|---|---|",
           sprintf("| %s | `%s` | %s |", idx$paper, idx$file, idx$title))
writeLines(lines, file.path(CFG$out, "README_output.md"))
message("Index of ", nrow(idx), " files: ", file.path(CFG$out, "README_output.md"))
