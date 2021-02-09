# Upsampling Artifacts in Neural Audio Synthesis

This repository contains a markdown copy of our article ([ARTICLE.md](ARTICLE.md)) with code to experiment with its figures:
* **[Figures 1 and 9](Figures/Fig1_9.ipynb)**: experiment with several neural audio synthesizers that can produce upsampling artifacts.
* **Figures 2, 3, 4**: conceptual figures, with no code to experiment with them.
* **[Figure 5](Figures/Fig5.ipynb)**: understand that any transposed convolution setup, even with full or no overlap setups, produces a poor initialization (with tonal artifacts) due to the weights initalization.
* **[Figure 6 (left column)](Figures/Fig6_left.ipynb)**: experiment with nearest neighbor interpolation upsampling layers, that introduce filtering artifacts.
* **[Figure 6 (right column)](Figures/Fig6_right.ipynb)**: experiment with linear interpolation upsampling layers, that introduce filtering artifacts.
* **[Figure 7](Figures/Fig7.ipynb)**: experiment with subpixel convolution upsampling layers, that can introduce tonal artifacts.
* **[Figure 8 (left column)](Figures/Fig8_left.ipynb)**: experiment with transposed convolution upsampling layers, that can introduce tonal artifacts.
* **[Figure 8 (right column)](Figures/Fig8_right.ipynb)**: experiment with linear interpolation layers, without the interleaved (learnable) convolutions. This figure is used as reference to discuss the spectral replicas of filtering artifacts.

#### Abstract:

A number of recent advances in neural audio synthesis rely on upsampling layers, which can introduce undesired artifacts. In computer vision, upsampling artifacts have been studied and are known as checkerboard artifacts (due to their characteristic visual pattern). However, their effect has been overlooked so far in audio processing. Here, we address this gap by studying this problem from the audio signal processing perspective. 

In our study, we show that the main sources of upsampling artifacts are: (i) the tonal and filtering artifacts introduced by problematic upsampling operators, and (ii) the spectral replicas that emerge while upsampling. 
We then compare different upsampling layers, showing that nearest neighbor upsamplers can be an alternative to the problematic (but state-of-the-art) transposed and subpixel convolutions which are prone to introduce tonal artifacts.

See the complete article at [ARTICLE.md](ARTICLE.md) or on [arXiv](https://arxiv.org/pdf/2010.14356.pdf).

#### Reference:
```
@inproceedings{pons2021upsampling,
  title={Upsampling artifacts in neural audio synthesis},
  author={Pons, Jordi and Pascual, Santiago and Cengarle, Giulio and Serr{\`a}, Joan},
  booktitle={IEEE international conference on acoustics, speech and signal processing (ICASSP)},
  year={2021},
  organization={IEEE}
}
```
