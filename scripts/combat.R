#!/usr/bin/env Rscript
# combat.R — sva::ComBat batch correction for Exhalo-Scan
# Usage: Rscript combat.R <input_csv> <metadata_csv> <batch_col> <output_csv>
#
# Requires: sva (Bioconductor), optionally installed via:
#   if (!requireNamespace("BiocManager")) install.packages("BiocManager")
#   BiocManager::install("sva")

suppressPackageStartupMessages({
  args <- commandArgs(trailingOnly = TRUE)
})

if (length(args) < 4) {
  cat("Usage: Rscript combat.R <input.csv> <metadata.csv> <batch_col> <output.csv>\n")
  quit(status = 1)
}

input_csv   <- args[1]
meta_csv    <- args[2]
batch_col   <- args[3]
output_csv  <- args[4]

if (!requireNamespace("sva", quietly = TRUE)) {
  cat("ERROR: sva package not installed. Install via BiocManager::install('sva')\n")
  quit(status = 2)
}

library(sva)

dat  <- read.csv(input_csv,  row.names = 1, check.names = FALSE)
meta <- read.csv(meta_csv,   row.names = 1, check.names = FALSE)

# Align metadata to data order
meta <- meta[rownames(dat), , drop = FALSE]

if (!batch_col %in% colnames(meta)) {
  cat(sprintf("ERROR: Batch column '%s' not found in metadata.\n", batch_col))
  quit(status = 3)
}

batch <- meta[[batch_col]]

if (length(unique(batch)) < 2) {
  cat("WARNING: Only one batch found — ComBat skipped, writing input unchanged.\n")
  write.csv(dat, output_csv, quote = FALSE)
  quit(status = 0)
}

# sva::ComBat expects features as rows, samples as columns
edata <- t(as.matrix(dat))

# Model matrix (intercept only — no covariates)
mod <- model.matrix(~1, data = meta)

tryCatch({
  corrected <- ComBat(
    dat    = edata,
    batch  = as.factor(batch),
    mod    = mod,
    par.prior = TRUE,
    prior.plots = FALSE
  )
  corrected_t <- as.data.frame(t(corrected))
  # Restore SampleID column
  corrected_t <- cbind(SampleID = rownames(corrected_t), corrected_t)
  write.csv(corrected_t, output_csv, row.names = FALSE, quote = FALSE)
  cat(sprintf("ComBat correction complete. Output: %s\n", output_csv))
}, error = function(e) {
  cat(sprintf("ERROR in ComBat: %s\n", conditionMessage(e)))
  quit(status = 4)
})
