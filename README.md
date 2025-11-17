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
