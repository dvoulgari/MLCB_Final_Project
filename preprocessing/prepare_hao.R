%%R
# Load final Seurat object
hao <- qread("./data/Hao/pbmc_hao_ref_up.qs")

# Show the number of cells per label.
table(hao$harmonized_celltype)

# Load and filter signature genes
our_genes <- read.table("./discovery/immdisc_aybey_final_list.tsv", sep = "\t", header = TRUE)
selected_annotations <- c("Plasma", "NK", "T CD8", "B", "Monocytes", "DC", "T CD4", "pDC")
filtered_genes <- our_genes %>% filter(Annotation %in% selected_annotations)

selected_hao_genes <- intersect(rownames(hao), filtered_genes$genes)

# Filter Hao to remove "Other cells" as in the original pipeline
hao <- hao[, hao$harmonized_celltype != "Other cells"]

# Normalization was already applied in the original preprocessing script
# No need to normalize again here
# e.g. hao <- NormalizeData(hao)
# hao <- ScaleData(hao, features = selected_hao_genes)

# Extract hao expression matrix
# expr_mat_hao <- as.data.frame(t(GetAssayData(hao, layer = "data")))
expr_mat_hao <- as.data.frame(t(GetAssayData(hao[selected_hao_genes, ], layer = "data")))
expr_mat_hao$label <- hao$harmonized_celltype

# Save
write.csv(expr_mat_hao, file = "hao.csv", row.names = TRUE)
