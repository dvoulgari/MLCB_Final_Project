library(Seurat)
library(dplyr)
library(qs)

# Load final Seurat object
sobj <- qread("./data/Kotliarov/kotliarov_pbmc.qs")

# Show the number of cells per label.
print(table(sobj$harmonized_celltype))

# Load and filter signature genes
our_genes <- read.table("./discovery/immdisc_aybey_final_list.tsv", sep = "\t", header = TRUE)
selected_annotations <- c("Plasma", "NK", "T CD8", "B", "Monocytes", "DC", "T CD4", "pDC")
filtered_genes <- our_genes %>% filter(Annotation %in% selected_annotations)
selected_genes <- intersect(rownames(sobj), filtered_genes$genes)

# Normalization was already applied in the original preprocessing script
# No need to normalize again here
# sobj <- NormalizeData(sobj)
# sobj <- ScaleData(sobj, features = selected_genes)

# Extract expression matrix
# expr_mat <- as.data.frame(t(GetAssayData(sobj, layer = "scale.data")))
expr_mat <- as.data.frame(t(GetAssayData(sobj[selected_genes, ], layer = "data")))
expr_mat$label <- sobj$harmonized_celltype

# Save
write.csv(expr_mat, file = "kotliarov.csv", row.names = TRUE)
