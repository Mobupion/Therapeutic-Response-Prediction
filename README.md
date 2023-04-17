# Therapeutic-Response-Prediction
Paper: Deep Learning of 2D-Restructured Gene Expression Representations for Improved Low-Sample Therapeutic Response Prediction

All datasets used in this paper are accessible in the "0_datasets" folder, and essential codes and results in this study are provided in the “codes” and "results" folders respectively. (`Note`: "CN-AML" is an old name of "Chemo-AML" as well as "LNN-BRCA" to "CTR-BRCA") 

# AggMap environment creation
It's necessary to install AggMap (https://github.com/shenwanxiang/bidd-aggmap) before result reproduction and make sure that related files are accessible in these codes.

# Overview
![Clinical therapeutic response prediction and biomarker identification based on multi-channel 2D-GERs](https://github.com/Mobupion/Therapeutic-Response-Prediction/raw/main/workflow.jpg)
### Clinical therapeutic response prediction and biomarker identification based on multi-channel 2D-GERs.
a)	Flowchart of manifold-guided restructuring of gene expression data into 2D-GREs for each clinical dataset and the subsequent CNN-based model construction for patient response prediction and biomarker discovery; 
b)	A multi-channel 2D-GERs on which each pixel represents the expression level of a specific gene. Transcriptome data from each clinical dataset are transformed to 4D vector, (samples in batch size, channel, height and width of 2D-GER) and then input to the 2D CNN-based AggMapNet architecture; 
c)	FI (feature importance) score measured by the increase of model loss due to the change of expression level for each gene to background value, and the importance map (saliency map) from PD-Bor training set and associated model is shown as an example. The top-rank important genes in red yellow and green colours are concentrated in a few hot-zones. 
