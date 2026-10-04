# run_all.R --------------------------------------------------------------------------------------
# Runs the analysis pipeline:
#   Rscript run_all.R               # everything
#   Rscript run_all.R 3 5           # only these steps (step 1 must have run once)
# Steps: 1 load  2 baseline  3 incidence  4 agreement  5 R  6 targeting  7 gap  8 map  9 index
args <- commandArgs(trailingOnly = TRUE)
file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
root <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg[1]))) else getwd()
setwd(root)
source("scripts/00_setup.R")

steps <- c("01_load", "02_baseline", "03_incidence", "04_agreement", "05_interchange",
           "06_targeting", "07_gap", "08_map", "09_index")
sel <- if (length(args)) as.integer(args) else seq_along(steps)
if (!1 %in% sel && !file.exists(CACHE)) sel <- union(1, sel)
if (1 %in% sel) unlink(INDEX)
log <- file.path(CFG$out, "logs", paste0("run_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".log"))
say <- function(...) { m <- paste0(...); cat(m, "\n"); cat(m, "\n", file = log, append = TRUE) }

status <- data.frame(step = character(), ok = logical(), seconds = numeric(), note = character())
for (i in sel) {
  say("\n>>> ", steps[i])
  t0 <- Sys.time()
  res <- tryCatch({
    withCallingHandlers(source(file.path("scripts", paste0(steps[i], ".R")), local = new.env()),
                        message = function(m) cat(conditionMessage(m), file = log, append = TRUE))
    list(ok = TRUE, note = "")
  }, error = function(e) list(ok = FALSE, note = conditionMessage(e)))
  secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  status[nrow(status) + 1, ] <- list(steps[i], res$ok, round(secs, 1), res$note)
  say(if (res$ok) "    ok" else paste("    FAILED:", res$note), sprintf(" (%.1f s)", secs))
  if (!res$ok && i == 1) { say("Stopping: loading must succeed."); break }
}
say("\nSummary:"); for (r in seq_len(nrow(status)))
  say(sprintf("  %-16s %s %6.1f s %s", status$step[r], if (status$ok[r]) "ok    " else "FAILED",
              status$seconds[r], status$note[r]))
if (!all(status$ok)) quit(status = 1)
