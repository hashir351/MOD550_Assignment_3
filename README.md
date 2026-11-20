# MOD550 – Assignment 3: Engineering Machine Learning with Real-World Data

Part I: SVM classification (`Part_I_SVM.ipynb`) · Part II: Deep learning, 1-D CNN (`Part_II_Deep_Learning.ipynb`)

## Dataset
Paderborn University Bearing DataCenter (KAt-DataCenter), Chair of Design and Drive Technology, Paderborn University.
- Bearings used: K001 (healthy), KA01 (outer-race damage), KI01 (inner-race damage); 80 recordings each (240 in total), channel `vibration_1`.
- Page: https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/data-sets-and-download
- Download: https://groups.uni-paderborn.de/kat/BearingDataCenter/
- Citation: Lessmeier, C., Kimotho, J. K., Zimmer, D., Sextro, W. (2016). Condition Monitoring of Bearing Damage in Electromechanical Drive Systems by Using Motor Current Signals of Electric Motors: A Benchmark Data Set for Data-Driven Classification. European Conference of the PHM Society, Bilbao.
- License: CC BY-NC 4.0 (non-commercial academic use with citation).

The raw data is not included in this repository (too large). Download the three bearing archives and extract the `.mat` files into a folder named `dataset/` next to the notebooks.

## Dependencies
Python 3.14 with TensorFlow 2.22 (release candidate). Install with `pip install -r requirements.txt`
(numpy, pandas, scipy, matplotlib, scikit-learn, tensorflow/keras, jupyter, ipykernel).

## How to run
1. Put the `.mat` files in `dataset/`.
2. Run `Part_I_SVM.ipynb` and `Part_II_Deep_Learning.ipynb` from top to bottom (random seed 42), or from a terminal:
   `python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 Part_II_Deep_Learning.ipynb`
3. The first run of the Part II notebook reads all `.mat` files and writes `segments_cache_all.npz` (ignored by git) to speed up re-runs.
4. Part II trains on every 10th training segment for at most 15 epochs (CPU time limit); this takes roughly 15 minutes on a laptop CPU. Figures are saved in `figures/` and test metrics in `results/ii3_metrics.json`.

## Part II results (test set: 6144 segments from 48 unseen recordings)
1-D CNN: accuracy 0.954, macro F1 0.954. RBF-SVM on 4 hand-crafted features (same split): accuracy 0.900, macro F1 0.898.
All CNN errors are inner-race segments predicted as healthy, concentrated at operating condition N09_M07_F10. Details are in the notebook.

## AI log (code, debugging and figures)
| Tool | Purpose | What was kept / changed |
|---|---|---|
| Claude (Anthropic) | Code for Part II: loading and segmenting the recordings, recording-level 60/20/20 split, 1-D CNN, training and evaluation, RBF-SVM comparison | Kept the architecture, split and evaluation code. Changed the training setup (every 10th training segment, at most 15 epochs) because the full run was too slow on a CPU. |
| Claude (Anthropic) | Debugging the setup and runs: Python/TensorFlow install, Jupyter kernel problem, PowerShell and git commands, notebook run settings | Used the suggested commands; ran the notebook from the terminal with nbconvert instead of the VS Code kernel picker. |
| Claude (Anthropic) | Figures for Task II.3 and II.4: learning curves, confusion matrix, accuracy by condition, misclassified examples | Kept the generated plotting code unchanged. |
| [Royar / Shishir: add your own entries for Part I] | [..] | [..] |

## Group contributions
- Royar Kurdman: Part I Tasks I.4 and I.5, Part II Task II.1
- Hashir Javed: Part II Tasks II.2, II.3 and II.4
- Shishir Dahal: [fill in the remaining tasks]

Individual answers in each notebook were written by the student named in the "Individual Analysis" heading.
