# VL-detector-eval
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repo contains the implementation of WACV 2024 paper "Towards Addressing the Misalignment of Object Proposal Evaluation for Vision-Language Tasks via Semantic Grounding".

arxiv: https://arxiv.org/abs/2309.00215

### Overview
This project aims to adapt detector evaluation metrics to Vision-Language tasks using semantic grounding. The repo is divided into two directories: 'detector_benchmark' which provides an easy to use VL-specific benchmark for a proposed detector architecture (shown in Table 2 of the WACV 2024 paper) and 'semantic_grounding' which contains the implementation of the semantic grounding algorithm (described in Section 3.4 of the WACV 2024 paper). Human survey data including templates, annotation comparison images, and AMT responses (corresponding to Section 4.2 of the WACV 2024 paper) can also be found in 'semantic_grounding/survey_data/'.

### Requirements
You can run requirements/install.sh to quickly install all the requirements in an Anaconda environment. The requirements are:
- python3
- pycocotools
- numpy
- shapely
- spacy
- gensim
- opencv-python
- pygsp

### Detector Benchmark Usage
Our benchmark is intended to be used for detectors utilizing the Visual Genome dataset splits and categories from the paper https://arxiv.org/abs/1701.02426 and corresponding repository https://github.com/danfeiX/scene-graph-TF-release. Save the top 30 most confident proposals from your detector for the image ids provided in 'detector_benchmark/detector_input/vg_test_ids.txt' (urls for these images can be found in 'detector_benchmark/detector_input/image_data.json') as a binary file in the location 'detector_benchmark/baseline_proposals/my_detector_proposals.bin'. Binary files for the baselines used in the paper can also be found in 'detector_benchmark/baseline_proposals/' (their performance is reported in Table 2 of the WACV 2024 paper). You can then evaluate your proposals or the baseline proposals by running 'python3 evaluate_proposals.py DETECTOR_PROPOSALS_FILE ANNOTATION_FILE' (ex: python3 evaluate_proposals.py motifs2018_proposals.bin vg_gt_thres30.json) where the ANNOTATION_FILE refers to one of the thresholded annotation sets found in 'detector_benchmark/thresholded_annotations'. **We recommend reporting Precision, Recall, and F1 score with both a threshold of 0.075 and 0.30 corresponding to 'vg_gt_thres075.json' and 'vg_gt_thres30.json' files, respectively.** Scores for the baseline detectors are reported in the table below:

| Detector | Precision (T=0.075) | Recall (T=0.075) | F1 Score (T=0.075) | Precision (T=0.30) | Recall (T=0.30) | F1 Score (T=0.30) |
| :-------- | :-------: | :--------: | :-------: | :-------: | :--------: | ------- |
| Neural Motifs (https://github.com/rowanz/neural-motifs) | 18.0 | 37.7 | 24.3 | 5.9 | 46.7 | 10.6 |
| Unbiased SGG (https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) | 22.9 | 18.7 | 40.0 | 25.5 | 5.2 | 47.2 | 9.4 |
| Microsoft SGG Benchmark (https://github.com/microsoft/scene_graph_benchmark)   | 24.5 | 20.0 | 41.7 | 27.0 | 5.7 | 50.9 | 10.2 |

### Semantic Grounding Algorithm Usage
Run 'python3 generate_human_survey_importance_scores.py' to generate a dictionary in './generated_COCO_survey_score.json' containing COCO survey image information and "weight" scores corresponding to annotation importance. 

### Author/Maintainer:
Joshua Feinglass (https://scholar.google.com/citations?user=V2h3z7oAAAAJ&hl=en)

If you find this repo useful, please cite:
```
@inproceedings{feinglass2024vldet,
  title={Towards Addressing the Misalignment of Object Proposal Evaluation for Vision-Language Tasks via Semantic Grounding},
  author={Joshua Feinglass and Yezhou Yang},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year={2024},
  url={https://arxiv.org/abs/2309.00215}
}
```
