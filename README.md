# Language Neuron Identification and Manipulation

This repository provides code for identifying and manipulating language-specific neurons in transformer-based language models using the **LAPE (Language Probability Entropy)** method introduced by [Tang et al. (2024)](https://arxiv.org/abs/2402.16438).

## Overview

We provide a complete pipeline for:

- **Data Preparation**: Scripts for collecting and formatting multilingual datasets for activation extraction.
- **Activation Extraction**: Python and shell scripts for collecting MLP activations from transformer models.
- **Neuron Identification**: Code for extracting neurons associated with specific languages (using LAPE method).
- **Neuron Manipulation**: Scripts for applying **additive interventions** to steer model behavior using the identified neurons.


## Citation

The paper with our results for 20 typologically diverse languages will be released soon!

If you use this repository, please consider citing:

```bibtex
@misc{tang2024languagespecificneuronskeymultilingual,
      title={Language-Specific Neurons: The Key to Multilingual Capabilities in Large Language Models}, 
      author={Tianyi Tang and Wenyang Luo and Haoyang Huang and Dongdong Zhang and Xiaolei Wang and Xin Zhao and Furu Wei and Ji-Rong Wen},
      year={2024},
      eprint={2402.16438},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2402.16438}, 
}
```
