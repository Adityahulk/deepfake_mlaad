# Training, Testing, and Evaluation Strategy

## 1\. Overview & Core Thesis

The primary objective of this project is to develop a lightweight, edge-deployable audio deepfake detector that achieves State-of-the-Art (SOTA) generalization results.

**The Problem:** Standard deepfake detectors (e.g., AASIST, RawNet) often overfit to specific **acoustic artifacts** present in the training data (e.g., background noise, specific vocoder glitches). When faced with a new, unseen generation method (e.g., a modern diffusion model), these detectors fail catastrophically.

**Our Solution (The Thesis):** We hypothesize that **prosody** (rhythm, intonation, stress) is a more robust and generalizable signal than acoustic artifacts. We propose a **Prosody-Content Disentanglement** framework. By forcing the model to "unlearn" semantic content (words), we compel it to learn pure prosodic representations, making it robust to unseen attacks.

-----

## 2\. Experimental Setup: Hero vs. Baseline

To demonstrate the efficacy of our Disentanglement method, we conduct a rigorous ablation study comparing our "Hero" model against a strictly controlled "Baseline."

### 2.1. The Comparison Configuration

| Feature | **Baseline Model (Control)** | **Ours ("Hero") Model** |
| :--- | :--- | :--- |
| **Backbone Architecture** | Lightweight TCN / ResNet | Lightweight TCN / ResNet (Identical) |
| **Input Features** | Mel-Spectrograms | Mel-Spectrograms |
| **Auxiliary Network** | **None** | **Wav2Vec2 (Frozen)** + Discriminator |
| **Loss Function** | $L_{spoof}$ (Cross-Entropy) | $L_{spoof} + \lambda L_{adv}$ (Adversarial) |
| **Training Objective** | "Classify Real vs. Fake" | "Classify Real vs. Fake" AND "Hide Content" |
| **Inference Speed** | Fast (Edge Ready) | Fast (Edge Ready) - *Auxiliary nets are discarded* |

### 2.2. The Baseline Definition

The Baseline represents the "standard approach." It uses the **exact same** lightweight TCN architecture and training data as the Hero model. However, it is trained **without** the Disentanglement framework.

  * **Purpose:** To prove that any performance gain in the Hero model is due strictly to the *disentanglement method*, not the model architecture.
  * **Mechanism:** The Wav2Vec2 branch and Discriminator are removed. The model is optimized solely on Binary Cross-Entropy.

-----

## 3\. Mathematical Framework (Loss Functions)

We utilize a Multi-Task Learning objective. Note that we do **not** optimize Equal Error Rate (EER) directly, as it is a non-differentiable metric based on sorting.

### 3.1. The Primary Loss: $L_{spoof}$

**Type:** Binary Cross-Entropy Loss.
$$L_{spoof} = -\frac{1}{N} \sum_{i=1}^{N} y_i \cdot \log(p(y_i)) + (1-y_i) \cdot \log(1-p(y_i))$$

  * **Role:** Teaches the Prosody Encoder to distinguish between Real (0) and Fake (1).
  * **Used By:** Both Baseline and Hero.

### 3.2. The Adversarial Loss: $L_{adv}$ (Novelty)

**Type:** Mean Squared Error (MSE) with Gradient Reversal.
$$L_{adv} = || D(E(x)) - C(x) ||^2$$

  * **Components:**
      * $E(x)$: The features from our Prosody Encoder.
      * $C(x)$: The content features from the frozen Wav2Vec2 teacher.
      * $D(\cdot)$: The Discriminator network.
  * **The Mechanism:** A **Gradient Reversal Layer (GRL)** flips the gradient during backpropagation.
      * The Discriminator tries to *minimize* this error (predict content accurately).
      * The Prosody Encoder tries to *maximize* this error (hide content information).
  * **Role:** Acts as a regularizer to "scrub" semantic information from the embeddings.
  * **Used By:** Hero Model **ONLY**.

-----

## 4\. Dataset Strategy

We employ a "Massive Fake / Massive Real" strategy to prevent overfitting to specific environments.

| Role | Dataset | Split Used | Details |
| :--- | :--- | :--- | :--- |
| **Real Audio** | **LibriSpeech ASR** | `train.100` | High-quality, diverse speakers. Provides a "gold standard" for natural prosody. |
| **Fake Audio** | **MLAAD** | `train` | Aggregated fakes from \~119 different models. Provides extreme diversity in attack types. |
| **Validation** | **MLAAD** | `validation` | Used for model checkpointing (saving the best model). |
| **TESTING** | **ASVspoof 2019 LA** | `eval` | **COMPLETELY UNSEEN.** Used only for final paper results. |

**Data Loading:**
The training loop streams these two datasets and interleaves them with a 50/50 probability ratio to create perfectly balanced batches ($B=16 \rightarrow 8 \text{ Real}, 8 \text{ Fake}$).

-----

## 5\. Training Protocol

### 5.1. Hyperparameters

  * **Optimizer:** Adam
  * **Learning Rate:** `1e-4`
  * **Batch Size:** 16
  * **Epochs:** 50 (Early stopping based on Validation EER)
  * **Adversarial Weight ($\lambda$):** `0.1` (Determines the strength of the "tug-of-war")

### 5.2. Training Steps

1.  **Smoke Test:** Run for 2 epochs with `STEPS_PER_EPOCH=100` to verify pipeline stability and loss convergence (Spoof Loss should drop below 0.69).
2.  **Baseline Run:** Train the model with `total_loss = loss_spoof`. Save as `baseline_model.pth`.
3.  **Hero Run:** Train the model with `total_loss = loss_spoof + 0.1 * loss_adv`. Save as `prosody_encoder_best.pth`.

-----

## 6\. The Generalization Test (The A\* Proof)

This is the critical evaluation step that defines the success of the research for top-tier conferences (ICASSP/INTERSPEECH).

**Hypothesis:** The Baseline will overfit to MLAAD artifacts and fail on ASVspoof. The Hero model will generalize via prosody and succeed on ASVspoof.

### 6.1. The Protocol

1.  Load the **ASVspoof 2019 Logical Access (LA)** evaluation dataset.
2.  Run `test.py` using `baseline_model.pth`. Record EER.
3.  Run `test.py` using `prosody_encoder_best.pth`. Record EER.

### 6.2. Target Results Table ("The Money Maker")

The final paper will present the following comparison table. A significant delta in the last column proves the hypothesis.

| Model | Training Data | Evaluation Data | EER (%) |
| :--- | :--- | :--- | :--- |
| **Baseline (TCN)** | MLAAD + LibriSpeech | MLAAD (Validation) | Low (e.g., \~10%) |
| **Ours (Disentangled)** | MLAAD + LibriSpeech | MLAAD (Validation) | Low (e.g., \~8%) |
| | | | |
| **Baseline (TCN)** | MLAAD + LibriSpeech | **ASVspoof 19 (UNSEEN)** | **High (e.g., \>40%)** |
| **Ours (Disentangled)** | MLAAD + LibriSpeech | **ASVspoof 19 (UNSEEN)** | **Low (e.g., \~12%)** |

*Note: The Baseline failing on ASVspoof is NOT a bug; it is a feature of the experiment. It highlights the difficulty of the generalization problem.*

-----

## 7\. Reproducibility

To reproduce these results:

1.  **Setup Environment:**
    ```bash
    pip install torch torchaudio transformers datasets torchcodec
    ```
2.  **Train Baseline:**
    Modify `train.py` to set `total_loss = loss_spoof`.
    ```bash
    nohup python -u train.py > baseline_logs.txt 2>&1 &
    ```
3.  **Train Hero:**
    Modify `train.py` to set `total_loss = loss_spoof + loss_adv`.
    ```bash
    nohup python -u train.py > hero_logs.txt 2>&1 &
    ```
4.  **Evaluate:**
    Ensure `LA.zip` is downloaded and unzipped.
    ```bash
    python test.py
    ```