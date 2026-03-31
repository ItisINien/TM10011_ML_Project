**Machine Learning Analysis for the Differentiation of Benign and Malignant Solid Liver Tumors** 

**Authors**
Lieve Bron (L.Bron-1@student.tudelft.nl)
Nienke Hendriksen (N.hendriksen@student.tudelft.nl)
Loïs Staleman (l.c.m.staleman@student.tudelft.nl)
Jos Vork (j.w.vork@student.tudelft.nl)

**Description**
This repository contains code for training and evaluating machine learning models (Random Forest, Support Vector Machine, Logistic Regression) to differentiate benign and malignant solid liver tumors using radiomic features extracted from T2-weighted MRI scans.

**Dataset**
Dataset: Worcliver (186 patients, 493 features)

**Code**
The main execution script is final.py, which orchestrates the pipeline by calling functions from the other modular scripts. For transparency, all individual scripts and their corresponding code are also available in the main branch of this repository.

Please note: The Random Forest optimization process is computationally intensive and may take a significant amount of time to complete.

**Notes**
Based on our comparative analysis, the Random Forest model using correlation and constant feature filtering (Forest_Cor_Const.py) achieved the highest performance across all tested strategies and classifiers.

**Contact**
For questions, contact authors (mail adresses on top).
