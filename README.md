# Code for Coreset for Rational Functions
This folder contains the official code for all the algorithms and examples that appeared in the thesis.
## Construction
- **def_ex.py**, code to generate the illustration for claim 1.
- **Example.py**, code to generate the illustration at Fig. 1 of the thesis.
- **algorithms.py**, Implementation of the algorithms used at the tests, as stated at the paper.
- **l_inf_rational_approx.py**, Conversion to python of RationalMinMaxOpt.m from https://github.com/nirsharon/rational_approx, that implements the
manuscript: "Rational approximation and its application to improving deep learning classifiers", by
V. Peiris, N. Sharon, N. Sukhorukova J. Ugon. See (arXiv link) https://arxiv.org/abs/2002.11330.

## Unified
- **UnifiedFramework-master**, fork of [official implementation of **Coresets for Near-Convex Functions**](https://github.com/muradtuk/UnifiedFramework).
- **Compute_sen.py**, interface to run the sensitivity bound from as defined in Algorithm 3 of the paper; used only for debugging.

## Tests
- **Helpers**, contains helper functions for the tests:
  - **GenOpts.py**, a helper library to mostly generate the options for the compression schemes considered.
  Includes the function compute_options, which is a pre-computing function that computes all the approximations for 
    the compression schemes; or sensitivities at the case of the use of *Coresets for Near-Convex Functions*.
    Includes as well functions to compute the coresets from the approximations/sensitivities computed above.
  - **loss.py**, a helper library that contains helper functions to compute the loss of the options generated above.
- **CoreSetSyntheticNoise.py**, contains the tests for the synthetic data mentioned at the thesis.
- **Beijing.py**, contains the tests for the [Beijing data set](https://archive-beta.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data) mentioned at the thesis.
- **Coreset_airQ.py**, contains the tests for the [Gas Multisensor Dataset](https://archive-beta.ics.uci.edu/dataset/360/air+quality) mentioned at the thesis.

## Results
Folder containing the results of the tests. Containing, split to Optimal solution (opt) and worst query (def):
  - **airQ**, all the results for [Gas Multisensor Dataset](https://archive-beta.ics.uci.edu/dataset/360/air+quality) mentioned at the thesis.
  - **Beijing**, a folder for the city *Aotizhongxin*, which contains all the results for [Beijing data set](https://archive-beta.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data) mentioned at the thesis.
  - **Synthetic**, all the results for the synthetic tests.

## Datasets
Folder containing the (real-life) datasets used at the tests, each consisting of a save_folder and original data (extracted at the corresponding tests).
- **airQ**, the data set [Gas Multisensor Dataset](https://archive-beta.ics.uci.edu/dataset/360/air+quality) mentioned at the thesis.
- **Beijing**, the data set for [Beijing data set](https://archive-beta.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data) mentioned at the thesis.
### Notes
The code uses PyTorch, and as stated in their official site, 
> It is recommended, but not required, that your Windows system has an NVIDIA GPU in order to harness the full power of PyTorch’s CUDA support.

(with similar explanation to other operating systems). Therefore, we recommend using an NVIDIA GPU with the corresponding cuda installation of torch (see https://pytorch.org/get-started/locally). <br/>
We use the default cuda driver, if you wish to change this change the values of the global variables both named `cuda` in **algorithms_part_1** and **hmm_algorithms**. <br/>
In case that you do not have an NVIDIA GPU the code will not use the GPU at all, and to run this change the global variables both named `cuda` to `torch.device("cpu")`.


## Writen by
- David Denisov, the sole writer all the code (unless stated otherwise).


### Honorable mentions
- Prof. Dan Feldman, the thesis supervisor that was always eager to answer any question regarding the research or writing, and to provide a constructive feedback.
- Dr. Ibrahim Jubran, the second writer of the thesis, which helped significantly improve the writing, gave ideas on missing parts in the tests,
and find bugs in the proofs.
Unfortunately since I (David Denisov) had a lot of bugs the last part was more significant than I would have hoped too, but
  this is entirely on my part, and while I would many times tell him about an exiting solution for the previously wrong ideas
  he was very eager to her it even if it would usually contain other bugs which he would very humbly point out.

