import json

P = "Part_II_Deep_Learning.ipynb"
nb = json.load(open(P, encoding="utf-8"))

# ---------- Royar's corrected Task II.1 and Model selection (240 recordings, all conditions) ----------
II1 = """## Task II.1 -- Engineering Problem and Dataset
### Question 1
The Part I dataset is reused: Paderborn bearing vibration signals (K001, KA01, KI01, all four operating conditions). The problem is bearing fault diagnosis, with a three-class target: healthy (0), outer-race damage (1) or inner-race damage (2).
### Question 2
The input is the raw `vibration_1` signal, cut into 2000-value segments. This gives an array of shape (30743, 2000, 1), from 240 recordings (about 10,250 segments per class).
### Question 3
A 1-D CNN is used because bearing faults appear as short, repeating impulses. Convolutional filters detect these local patterns at any position and learn features from the raw waveform."""

COMPUTE_NOTE = """

**Compute note.** To fit the available CPU time, the CNN was trained on every 10th training segment (1,846 of the 18,452 training segments, spread over all 144 training recordings) for at most 15 epochs. The validation and test sets are complete. The RBF-SVM baseline below was trained on all 18,452 training segments."""

OBS = """### Observations
- **Learning curves.** Training and validation loss both decrease steadily over all 15 epochs (validation loss 0.917 to 0.101, validation accuracy 0.555 to 0.962) and stay close together, so there is no sign of overfitting. The best epoch was epoch 15 of 15, so early stopping (patience 5) never triggered and the 15-epoch maximum ended training. The validation loss was still falling slowly, so the model is probably slightly under-trained.
- **Test performance** (6144 segments from 48 recordings never used in training or validation): accuracy 0.9539, macro precision 0.9595, macro recall 0.9540, macro F1 0.9537. Outer race: precision 1.000, recall 1.000. Healthy: precision 0.879, recall 1.000. Inner race: precision 1.000, recall 0.862.
- **Comparison with the feature baseline** on the same split: the RBF-SVM on the four Part I features reached accuracy 0.9001 and macro F1 0.8978, so the CNN is about 5.4 percentage points better in accuracy and 5.6 in macro F1. The SVM used default hyperparameters and all training segments; the CNN used one tenth of them. Each model was run once with one seed.
- **Generalisation gap.** Accuracy on the full training set is 0.9657, on validation 0.9624 and on test 0.9539, so the gap is small."""

DISC = """### Discussion
**Where the errors occur.** All 283 errors (4.61% of 6144 test segments) are inner-race segments predicted as healthy. Healthy and outer-race segments are classified perfectly. The errors sit in a single operating condition: N09_M07_F10 has accuracy 0.830 overall and only 0.558 for inner race, while the other three conditions are at 1.000. The five worst recordings are all KI01 (inner race) recordings at N09_M07_F10, with only 54-58% of their segments correct.

**What the misclassified signals look like.** The median RMS and peak of the wrongly classified inner-race segments (0.343 and 2.72) are close to the healthy segments (0.386 and 2.87) and much lower than the correctly classified inner-race segments (0.589 and 4.17). So the missed segments are low-amplitude and look healthy.

**Possible engineering explanations (hypotheses, not tested here).**
1. N09 is the lower rotational speed (900 rpm instead of 1500 rpm). Fault impacts are weaker and further apart at low speed, so a 2000-sample segment (about 31 ms) may contain very weak or no fault impacts.
2. Inner-race faults pass through the load zone of the bearing, so the impact strength varies and some segments look almost healthy.
3. The network may rely partly on signal amplitude, and low-amplitude segments then look healthy to it. It was also trained on only a tenth of the training segments.
4. Each class is one physical bearing, so one KI01 bearing at 900 rpm dominates the error.

**Conclusion.** The evidence shows that the errors are one-sided (inner race called healthy) and concentrated at 900 rpm. A leave-one-condition-out test, longer segments, and training on all segments would show which explanation is right."""

Q1 = """## Question II.1 -- Architecture Selection

I used a 1-D CNN because every sample is a single vibration signal, 2000 numbers recorded at 64 kHz, which is about 31 ms. A damaged bearing hits its crack again and again, and in the waveform this shows up as short sharp impulses. A convolution filter looks at a small window and slides along the signal, so it can learn to detect that kind of pattern wherever it happens. In my model the first Conv1D layer has 16 filters with kernel size 32 and the second has 32 filters with kernel size 16. Max pooling shortens the signal from 2000 to 500 and then 125 steps, and global average pooling averages each of the 32 filters over the whole segment. That matters because the segments are cut at arbitrary places, so an impulse can be anywhere and the model should not care where. The whole network has only 8,851 trainable weights.

An LSTM would have to go through the 2000 time steps one after another, which is slow on a CPU, and I don't think we need its long memory since the important patterns are short. The four hand-made features from Part I (RMS, peak, crest factor, kurtosis) throw away most of the signal before the model sees it. With the raw signal the CNN can choose its own features. In my run it did better than the SVM on the four features (accuracy 0.954 against 0.900)."""

Q2 = """## Question II.2 -- Model Performance

The training and validation curves look healthy to me. Over 15 epochs the training loss went from 1.04 to 0.13 and the validation loss from 0.92 to 0.10, and the two stayed close together the whole time. Validation accuracy rose from 0.56 to 0.962, and the final accuracies were 0.966 (train), 0.962 (validation) and 0.954 (test). So I see no overfitting. The best epoch was the last one, so early stopping never triggered and the 15-epoch limit stopped training. The validation loss was still going down slowly, so the model is probably a bit under-trained. I also trained on only every 10th training segment (1,846 of 18,452) because the full run was too slow on my CPU.

The CNN did better than the baseline. An RBF-SVM on the four Part I features, with the same recording-level split and default settings, got 0.900 accuracy and 0.898 macro F1. The CNN got 0.954 accuracy and 0.954 macro F1. That is about 5.4 percentage points, and the SVM even had all the training segments.

I don't want to oversell it, though. It is one run with one seed, the SVM was not tuned (the tuned SVM from Part I used a different split, so I can't compare with it), and the test set is only 48 recordings, so a few bad recordings change the score a lot. In fact all the errors came from inner-race recordings at one operating condition (N09_M07_F10). So the CNN is better in this experiment, but I can't say by exactly how much."""

Q3 = """## Question II.3 -- Engineering Applicability

I would recommend it for further validation, but not for real use yet. On the test recordings it got 0.954 accuracy and found every healthy and outer-race segment, but the mistakes are the worst kind. All 283 errors were inner-race faults classified as healthy, so recall for inner race is only 0.862. In a real machine that is a missed fault, which costs more than a false alarm. The errors are concentrated in one operating condition (900 rpm, N09_M07_F10), where only 56% of the inner-race segments were found, so I would not trust the model at low speed.

The most important limitation for me is the data. Each class is one physical bearing with artificially made damage, and the test set comes from the same three bearings. The network might have partly learned the identity of these bearings and not the fault type, so I can't say how it would do on a different bearing or on real wear damage. I also trained on only a tenth of the segments for 15 epochs, with one seed.

One experiment I would add is a leave-one-condition-out test: train on three operating conditions, test on the fourth, and repeat so each condition is the test condition once. I would also add other bearing codes from the same dataset so that the test bearings are never seen in training. Finally I would train on all segments with 3 seeds and use inner-race recall as the main number to compare."""

def text(c):
    return "".join(c["source"])

def setsrc(c, s):
    c["source"] = s.splitlines(True)

done = set()
for c in nb["cells"]:
    if c["cell_type"] != "markdown":
        continue
    t = text(c)
    if c["id"] == "7176acc0-9f2f-47cc-88c0-f517ead6fde8":
        setsrc(c, II1); done.add("II1")
    elif c["id"] == "608c1f9e-b942-4e30-be28-cd145bbf31ba":
        setsrc(c, t.replace("small dataset of 384 segments", "dataset of 30,743 segments from 240 recordings")); done.add("model_sel")
    elif t.startswith("## Task II.3 -- Training and Evaluation") and "Compute note" not in t:
        setsrc(c, t.rstrip() + COMPUTE_NOTE); done.add("note")
    elif t.startswith("### Observations"):
        setsrc(c, OBS); done.add("obs")
    elif t.startswith("### Discussion"):
        setsrc(c, DISC); done.add("disc")
    elif t.startswith("## Question II.1"):
        setsrc(c, Q1); done.add("q1")
    elif t.startswith("## Question II.2"):
        setsrc(c, Q2); done.add("q2")
    elif t.startswith("## Question II.3"):
        setsrc(c, Q3); done.add("q3")

for c in nb["cells"]:
    t = "".join(c["source"])
    if "Hasir" in t or "Javid" in t:
        c["source"] = t.replace("Hasir", "Hashir").replace("Javid", "Javed").splitlines(True)

json.dump(nb, open(P, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("patched:", sorted(done))
missing = {"II1", "model_sel", "note", "obs", "disc", "q1", "q2", "q3"} - done
print("MISSING:", sorted(missing) if missing else "none")
