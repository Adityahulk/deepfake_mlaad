# Prosody-Content Disentangled Audio Deepfake Detection

This project is an implementation of a novel, state-of-the-art architecture for audio deepfake detection, designed to achieve top-tier results on the **MLAAD dataset**.

The primary goal is to create a model that generalizes well against *unseen* spoofing attacks by focusing on fundamental prosodic cues rather than superficial acoustic artifacts. This approach is designed to be the foundation for an **A/A* conference-level paper*\* (e.g., ICASSP, INTERSPEECH).

## Core Thesis: Prosody-Content Disentanglement

The central hypothesis is that most deepfake detectors fail to generalize because they overfit to specific *acoustic artifacts* of known generators. These artifacts change with every new deepfake model.

A more robust and generalizable signal of a deepfake is its **unnatural prosody**. While an AI can create "clean" audio, it struggles to replicate the complex, semantically-tied rhythm, intonation, and stress of human speech.

This project trains a model to **"disentangle" prosody from content**. It learns to identify fakes by listening *only* to the prosody, while being actively *punished* for listening to the words being said.

## Architecture

This project uses an advanced "teacher-student" training framework.

1.  **The "Prosody Encoder" (Student):** This is the **actual deepfake detector**. It is a lightweight and fast TCN/ResNet model that learns to classify audio as `real` or `fake` from mel-spectrograms.
2.  **The "Content Encoder" (Teacher):** A large, **frozen Wav2Vec2 model**. Its only job is to provide a "content" embedding for the audio (i.e., *what* is being said).
3.  **The "Content Discriminator" (The Adversary):** This small network, equipped with a **Gradient Reversal Layer (GRL)**, attempts to predict the *content* using *only* the features from the Prosody Encoder.

### The Training "Trick"

The Prosody Encoder is trained to do two things at once:

  * **Task 1:** Get *better* at predicting `real` vs. `fake` (Standard spoofing loss).
  * **Task 2:** Get *worse* at helping the Discriminator guess the content (Adversarial disentanglement loss).

This process "scrubs" content information from the Prosody Encoder, forcing it to rely only on generalizable prosodic cues.

At inference time, **we discard the heavy Wav2Vec2 Teacher and the Discriminator**. The final model (`prosody_encoder_best.pth`) is just the lightweight, fast, and robust Prosody Encoder, making it ideal for edge devices.

## Datasets

This model is trained by combining two large-scale, streaming datasets:

  * **Fake Audio (Label 1):** [`mueller91/MLAAD`](https://huggingface.co/datasets/mueller91/MLAAD)
      * A massive, multi-lingual dataset containing audio from over 119 different spoofing models.
  * **Real Audio (Label 0):** [`librispeech_asr` (clean subsets)](https://huggingface.co/datasets/openslr/librispeech_asr)
      * A large corpus of high-quality, "bona fide" (real) speech.

The dataloader interleaves these two streams 50/50 to create balanced batches.

## How to Run

### 1\. Requirements

Install all necessary Python libraries.

```bash
pip install torch torchaudio numpy pandas scikit-learn
pip install transformers datasets
pip install torchcodec
conda install -c conda-forge ffmpeg
```

### 2\. Hugging Face Authentication

The `mueller91/MLAAD` dataset is "gated" and requires you to accept its terms.

1.  Go to the [MLAAD dataset page](https://huggingface.co/datasets/mueller91/MLAAD) and accept the terms and conditions.
2.  Log in to your Hugging Face account from your terminal:
    ```bash
    huggingface-cli login
    ```

### 3\. Run Training

Simply run the main Python script. The script handles all data downloading, streaming, processing, training, and validation.

```bash
python train.py
```

The script will:

  * Load the frozen Wav2Vec2 model and the trainable models.
  * Stream and interleave the MLAAD (fake) and LibriSpeech (real) datasets.
  * Train the model for the specified number of epochs.
  * At the end of each epoch, it will run a validation loop and report the **Equal Error Rate (EER)**.
  * It will automatically save the model with the best EER to `prosody_encoder_best.pth`.

-----

## To-Do List

Here is the step-by-step plan from "code works" to "paper submitted."

### Phase 1: Get the Code Running

1.  **Run a "Smoke Test":** Don't try to train for 50 epochs right away. Modify your training loop to run for just **100 steps** (`STEPS_PER_EPOCH = 100`) and validate for **20 steps** (`VAL_STEPS = 20`). Run this for just 2 epochs.

      * **Goal:** Just make sure the code doesn't crash, the loss goes down (even slightly), and the EER is not 50.0% (which is random guessing).

### Phase 2: 📈 The Main Experiments

3.  **Train the Baseline Model:** This is **the most important step** for a paper. You must prove your new idea is better than a simple model.

      * **How:** In `train.py`, find the line `total_loss = loss_spoof + loss_adv`.
      * **Change it** to `total_loss = loss_spoof`.
      * **Train this model** for the full 50 epochs and save its best EER. This is your "Baseline" (a standard TCN/ResNet without disentanglement).

4.  **Train the "Hero" Model (Your Model):**

      * **Change the loss back** to `total_loss = loss_spoof + loss_adv`.
      * **Train for 50-100 epochs.** This is your main experimental run. Let it cook for a day or two.
      * **Save the best model** (`prosody_encoder_best.pth`).

5.  **Hyperparameter Tuning:**

      * Your first run might not be perfect. The most important knob to turn is `LAMBDA_ADV`.
      * Try running experiments with `LAMBDA_ADV = 0.01`, `LAMBDA_ADV = 0.1` (current), and `LAMBDA_ADV = 1.0`.
      * Pick the one that gives the best EER on your validation set.

### Phase 3: 🌍 The Generalization Test (The A\* Proof)

This is the part that gets you into a top conference. You must show your model works on data it's **never seen before.**

6.  **Find a Test Set:** Download the **ASVspoof 2019 "Logical Access"** evaluation set. This is the gold standard for testing.
7.  **Write `test.py`:** Create a new, simple script that:
      * Loads your saved `prosody_encoder_best.pth` model (for both the Baseline and Hero).
      * Loads the ASVspoof 2019 audio.
      * Processes the audio (Mel-spec, padding, etc.) *exactly* as you did in training.
      * Feeds it to your model and calculates the final EER.
8.  **Create "The Table":** This is your paper's money-maker. It will look like this:

| Model | Training Data | Testing Data | EER (%) |
| :--- | :--- | :--- | :--- |
| Baseline (TCN) | MLAAD + LibriSpeech | MLAAD (Validation) | 10.5% |
| **Ours (Disentangled)** | MLAAD + LibriSpeech | MLAAD (Validation) | **8.2%** |
| Baseline (TCN) | MLAAD + LibriSpeech | **ASVspoof 2019 (Unseen)** | 45.2% |
| **Ours (Disentangled)** | MLAAD + LibriSpeech | **ASVspoof 2019 (Unseen)** | **12.1%** |

*(Note: These numbers are just examples, but this is what you're looking for. Your model (Ours) should be **dramatically** better on the unseen ASVspoof data).*

### Phase 4: ✍️ Write and Submit

9.  **Write the Paper:** Structure your paper around this story:

      * **Intro:** "Deepfake detectors overfit. This is a huge problem."
      * **Method:** "We propose a novel disentanglement architecture that separates prosody from content to solve this."
      * **Experiments:** "We trained on MLAAD. We show our model is good."
      * **Results:** "But most importantly, we tested on ASVspoof, and 'The Table' proves our model generalizes *far* better than a standard baseline."
      * **Conclusion:** "This is the right way to build detectors."

10. **Check Deadlines:** ICASSP and INTERSPEECH have *very* strict, non-negotiable deadlines. Find out when they are (e.g., INTERSPEECH is often around March) and plan your work backward from that date.