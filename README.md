# CF-HistoGAN

Implementation of the CF-HistoGAN. The code is written in pytorch-lightning, a lightweight wrapper for pytorch.

This method is based on the VAGAN by Baumgartner, C.F. et al. (2018), with implementation available on GitHub at https://github.com/baumgach/vagan-code.

## ABSTRACT
Understanding the interactions of different cell types inside the immune tumor microenvironment (iTME) is crucial for the development of immunotherapy treatments as well as for predicting their outcomes. Highly multiplexed tissue imaging (HMTI) technologies offer a tool which can capture cell properties of tissue samples while preserving their spatial location. HMTI technologies can be used to gain insights into the iTME and in particular how the iTME differs for different patient groups of interest (e.g., treatment responders vs. non-responders). However, such analyses are inherently limited by the fact that samples from patient groups of interest are always unpaired, i.e., we can never observe the same tissue sample with two different diagnoses. Here, we present CF-HistoGAN, a machine learning framework that employs generative adversarial networks (GAN) to create artificial counter-factual tissue samples that resemble the original tissue samples as closely as possible but appear to come from a different patient group. Specifically, we leverage recent developments in machine learning to "translate" HMTI samples from one patient group to appear like samples from another patient group to create artificial paired samples. We show that this approach allows to directly study the effects of a variable (e.g., diagnosis or outcome) on the iTME for individual tissue samples. On the one hand, this can serve as an explorative tool for understanding iTME effects on the pixel level. On the other hand, we show that our method can be used to identify parameters that correlate with the studied patient groups with greater test power compared to conventional approaches, which may help to uncover effects that would be otherwise undetectable.

## Architecture
This project is divided into three parts:

#### Preprocessing
The CRC and CTCL CODEX data has to be preprocessed. The scripts remove empty or redundant channels and split the images into smaller patches. In the scripts you can define the size of the patches, the overlapping strides and the down sample factor.

#### Training
Train the network by running the train.py script. There are a bunch of options that can be adapted in the parser.py file.

#### Postprocessing
A bunch of scripts that can be used to evaluate and visualize the results:

- meanAndAbsolutePixels.py Calculates the mean and absolute pixel change per channel for all samples.

- statTest.py computes the paired and un-paired t-tests. This is used to see if our method finds significant changes that are not present when directly comparing the images of both classes.

- multiNetColoring.py colores the channels with the highest absolute changes and composites them into a plot.

- multiNetHE.py visualizes the artifically generated HE images.
