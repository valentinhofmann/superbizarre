# Superbizarre Is Not Superb

This repository contains the code and data for the ACL paper [Superbizarre Is Not Superb: Derivational Morphology Improves BERT's Interpretation of Complex Words](https://aclanthology.org/2021.acl-long.279.pdf).

# Dependencies

The code requires `Python>=3.6`, `numpy>=1.18`, `torch>=1.2`, and `transformers>=2.5`.

# Data

The three datasets used in the experiments can be found in `data`.
The datasets contain derivatives with corresponding semantic classes (sentiment and topicality).
Please refer to the paper for details about the datasets.
The labeling of the datasets is as follows:

- Amazon (sentiment): 0 -- negative (e.g., _overpriced_, _crappy_), 1 -- positive (e.g., _megafavorite_, _applausive_)
- ArXiv (topicality): 0 --  physics (e.g., _semithermal_, _ozoneless_), 1 -- computer science (e.g., _autoencoded_, _rankable_)
- Reddit (topicality): 0 -- entertainment (e.g., _supervampires_, _spoilerful_), 1 -- knowledge (e.g., _antirussian_, _immigrationism_)

The datasets are provided as csv files and as segmentation-specific pickled PyTorch datasets that can be easily loaded for model training.
The repository also contains the code for generating the different segmentations in `src`.

# Usage

To replicate the hyperparameter search for the learning rate, run the script `start_hs.sh` in `src`.
To train the models using different segmentations, run the script `start_main.sh` in `src`.

# Citation

If you use the code or data in this repository, please cite the following paper:

```
@inproceedings{hofmann2021superbizarre,
    title = {Superbizarre Is Not Superb: Derivational Morphology Improves {BERT}{'}s Interpretation of Complex Words},
    author = {Hofmann, Valentin and Pierrehumbert, Janet and Sch{\"u}tze, Hinrich},
    booktitle = {Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics},
    year = {2021}
}
```
