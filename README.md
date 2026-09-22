
# Spectroscopy to structure, prediction of acetone and water geometries using FTIR, QM and ML methods 

# Project Overview
This project combines Computational Chemistry, FTIR Spectroscopy, and Machine Learning to predict the distribution of water geometries around acetone from experimental FTIR data.

# Background
Dilution of acetone with either water or carbon tetrachloride shifts the carbonyl band red or blue, respectively
https://assets.thermofisher.com/TFS-Assets/CAD/Application-Notes/AN50733_E.pdf

"The strong bathochromic shifts observed on methanol OH and acetone CO stretch IR bands are related to hydrogen bonds between these groups. Factor analysis separates the spectra into four acetone and four methanol principal factors." 
https://doi.org/10.1063/1.1790431

"Analysis  of  IR  spectra  of  ethylene  glycol  shows  that  there are only a few contributing bands with solidly fixed vibrational frequencies,  which  only  change  in  relative  intensity  when temperature is changed. It did not show any clear evidence of an intrinsic frequency shift indicating the gradual weakening of hydrogen bonding interaction. Only the relative population of species,  e.g.,  strongly  bonded  and  dissociated  or  much  more weakly  bonded  groups,  seems  to  be  changing.  IR  spectra  of acetone   in   a   mixed   solvent   of   CHCl3/CCl4with   varying composition  also  show  that  intrinsic  IR  frequency  does  notshift  appreciably.  Instead,  only  the  relative  contributions  of highly overlapped adjacent bands are changing."
https://doi.org/10.1366/000370210792434396

# Introduction

Water forms a small number of discrete hydrogen bonding states around the carbonyl on acetone. Generative AI methods such as Variational Auto encoders encode latent space allowing for predictions on the range of possible geometries for each state compared with regression methods producing only singular values. This work first uses a delta matrix made from the difference between vibrations of QM optimised geometries and experimental FTIR data to understand these discrete states and then includes input from Molecular Dynamics simulations to include thermally realistic variations. Initially the VAE is conditioned only on the carbonyl vibration. 

# Method

Mixtures of acetone and water were prepared across a range of volume ratios from 10:90 to 90:10 (acetone:water) and the FTIR spectrum was taken for each. The FTIR has been collected on a Nicolet iD7 with a resolution of 2cm-1. 

The carbonyl peak position at each concentration is determined using Voigt peak fitting (voigt_peak_centre.py). The difference between the experimental peak position and the scaled QM calculated vibrations is used to create the "delta matrix". 

# References

Graph neural networks for materials science and
chemistry
https://www.nature.com/articles/s43246-022-00315-6.pdf

Infrared Spectra Prediction Using Attention-Based Graph Neural Networks. Digital Discovery, Royal Society of Chemistry.
https://doi.org/10.1039/D3DD00254C

Excess Gibbs Free Energy Graph Neural Networks for Predicting Composition-Dependent Activity Coefficients of Binary Mixtures. arXiv preprint.
https://arxiv.org/abs/2407.18372

Representation Learning with a β-Variational Autoencoder for Infrared Spectroscopy. 
https://www.researchgate.net/publication/361453151

Anomaly Detection in Fourier Transform Infrared Spectroscopy of Pharmaceutical Tablets Using Variational Autoencoders. Chemometrics and Intelligent Laboratory Systems, 
https://doi.org/10.1016/j.chemolab.2023.104781

Infrared Spectroscopy of Acetone–Water Liquid Mixtures: Molecular Organization and Hydrogen Bonding. 
https://pubmed.ncbi.nlm.nih.gov/15267555

Spectroscopy from Machine Learning by Accurately Representing Molecular Structures. Nature Communications
https://www.nature.com/articles/s41467-023-36957-4

Learning Molecular Mixture Properties Using Chemistry-Aware Graph Neural Networks.
https://doi.org/10.1103/PRXEnergy.3.023006

Auto-Encoding Variational Bayes.
https://arxiv.org/abs/1312.6114

Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules.
https://doi.org/10.1021/acscentsci.7b00572

A Generative Model for Molecular Distance Geometry.
https://proceedings.neurips.cc/paper/2020/hash/fb60d411a5c5c4c7bd16c6d0bd1780b9-Abstract.html

GemNet: Universal Directional Graph Neural Networks for Molecules.
https://arxiv.org/abs/2106.08903
 






