# AI651 Assignment 2: river forecasting proposal and Moving MNIST ConvLSTM

Bimal Rehman, roll number 25280081. The graded answers are in the PDF report on the LMS. This repository holds the code, the saved results, the figures, the GIFs and the record of the run.

## Question 1. River forecasting proposal

Question 1 is a written design. It needs no data, training or GPU.

| File | Content |
|---|---|
| `report/q1.md` | The full answer under the six required headings |
| `report/q1_diagram.mmd`, `report/q1_diagram.png` | Figure Q1, the architecture diagram, as Mermaid source and image |
| `report/q1_counts.py` | Parameter and multiply add counts quoted in Section 5 of the answer |

`python report/q1_counts.py` prints 415,709 parameters, 93,010,944 multiply adds per forecast and 289,013,760 multiply adds for five joint attention blocks.

The course package also has an optional, ungraded companion notebook. Nothing from it is submitted and it is not part of this repository. To run it, unzip `DL4STG_PA2.zip` from the course page and use the `Question1` folder:

```
cd Question1
pip install -r requirements.txt
jupyter lab Question1.ipynb
```

Run the cells in order. The models are already trained in `Question1/checkpoints`, so no GPU is needed. The interactive controls need a live Jupyter kernel.

## Question 2. Motion or memory

All Question 2 results come from one run on a Google Colab Tesla T4 GPU with PyTorch 2.11.0+cu130. The executed record is `25280081_PA2_AI651_BR_SOLVED_V4.ipynb`.

### What was done

1. Table 1 checks of mirror, rotation and the window rules on the synthetic tensors.
2. Figure 1 optical flow statistics on the first 50 training videos.
3. Model A, the ConvLSTM baseline, trained with seeds 0 and 1. It has 480,784 parameters and a reach of 8 pixels per step.
4. The Q2.5 prediction committed to this repository before any mirror, rotation or speed test.
5. Mirror, rotation and speed tests of model A for Tables 3 and 4 and Figures 2 to 5.
6. The Q2.11 prediction committed before models B, C and D were trained.
7. Models B, C and D trained with seeds 0 and 1. Model D uses dilation 3 in layer 2, which gives a reach of 16 pixels with the same 480,784 parameters. A validation comparison of other D designs was started and stopped by hand, so D uses this default design. No test video was used to choose it.
8. Every model evaluated plain and symmetrised over the eight square symmetries for Tables 5 and 6 and Figure 6. The final model is symmetrised D.

### Layout

| Path | Content |
|---|---|
| `pa2/transforms.py` | Mirror, rotation and the eight element group |
| `pa2/windows.py` | Window rules at 1×, 2× and 3× |
| `pa2/models.py` | ConvLSTM forecaster used by models A to D |
| `pa2/train.py` | Training loop with speed and symmetry augmentation |
| `pa2/metrics.py` | Errors, consistency gaps, symmetrised forecaster and timing |
| `pa2/flow.py` | Farneback optical flow for Figure 1 |
| `pa2/reach.py` | Reach count and impulse test |
| `pa2/checks.py`, `pa2/selection.py`, `pa2/diagnostics.py`, `pa2/figures.py`, `pa2/configs.py`, `pa2/analysis.py` | Table 1, video choice for the figures, blob statistics, plotting, model settings and the failure rule used by the Q2.11 draft |
| `run.py` | One command per stage |
| `predictions/Q2.5.md`, `predictions/Q2.11.md` | The committed predictions, unedited |
| `results/` | Every saved metric as JSON and every per video gap as NPZ |
| `figures/` | Figures 1 to 6 as PNG |
| `gifs/` | The four GIFs of Figure 3 and the four GIFs of Figure 6(a) |
| `report/make_report.py`, `report/results_io.py` | Builds the report docx from `report/q1.md` and `results/` |
| `tests/test_all.py` | Unit checks of the transforms, windows, reach, loss mask, consistency and flow sign |

### How to reproduce Question 2

Open a Colab notebook with a T4 GPU, clone this repository and install the packages.

```
pip install -r requirements.txt
python tests/test_all.py
python run.py table1
python run.py flow
python run.py reach
python run.py train --model A --seed 0
python run.py train --model A --seed 1
python run.py evaluate --models A
for m in B C D; do for s in 0 1; do python run.py train --model $m --seed $s; done; done
python run.py evaluate --models A B C D
python run.py select_videos
python run.py figures --final D
python run.py diagnostics --final D
python report/make_report.py
```

`tests/test_all.py` must print `ALL TESTS PASSED`. Moving MNIST downloads to `data/` on the first call. One training run of A, B or C took about 11 seconds per epoch on the T4 and one run of D about 17 seconds per epoch, so the eight training runs take about 52 minutes. Checkpoints go to `checkpoints/`, which is not tracked by git.

### How to rebuild the report

`python report/make_report.py` needs only `results/`, `figures/`, `predictions/` and `report/q1.md`, so it runs on a CPU. It writes `report/25280081_PA2_AI651_BR.docx`. It also stops with an error if any statement in the Question 2 text no longer matches the saved numbers.
