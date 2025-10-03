## Anomaly detection

Goal: To detect outlier 

Common Constraints:
1. **Lack of Labeled Data**
- Problem: Most anomaly detection is unsupervised because anomalies are rare and expensive to label.
- Impact: Hard to evaluate or tune models without ground truth.

2. **Class Imbalance**
- Problem: Anomalies usually make up <1% of the dataset.
- Impact: Models may bias toward normal data and miss outliers (false negatives).

3. **Concept Drift**
- Problem: The definition of “normal” can change over time (e.g., in network traffic or user behavior).
- Impact: Static models become outdated and lose accuracy.

4. **High Dimensionality**
- Problem: In high dimensions, distance metrics become less meaningful due to curse of dimensionality.
- Impact: Makes models like k-NN or LOF less effective.

5. **Ambiguity of Anomalies**
- Problem: Anomalies can be contextual, collective, or point-based, and it's not always obvious what type you're dealing with.
- Impact: A model designed for one type may miss another (e.g., LOF may miss global outliers).

6. **Noise vs. True Anomalies**
- Problem: Distinguishing between noisy data and genuine anomalies is hard.
- Impact: May result in many false positives or missed true outliers.

7. **Domain Dependence**
- Problem: The definition of an anomaly depends on the domain context (e.g., in finance vs. health vs. IoT).
- Impact: Generic models may not generalize well without domain-specific tuning.

### Classical ML and Statistical Models

#### DBSCAN (Density‑Based Spatial Clustering of Applications with Noise)

Key Idea: DBSCAN groups together points in **high‑density regions** and marks points in **low‑density regions** as noise (outliers).  It does **not require specifying the number of clusters beforehand** and can discover clusters of **arbitrary shape** .

How it works:
1. Pick an unvisited point.
2. Retrieve all points within ε (its neighborhood).
3. If it's a core point, start a new cluster and recursively add all density‑reachable points:
   - A point B is **density-reachable** from A if there's a chain of core points connecting them
   - Add all core points and border points within this connected region
4. Continue until no more reachable points exist.
5. Repeat with the next unvisited point until all points are processed

Point Types & Rules:

- **Core Point**: ≥ minPts points (including itself) lie within ε.
- **Border Point**: fewer than minPts within ε, but within ε of a core point.
- **Noise Point**: neither core nor border → labeled as outlier.

Implementation Highlights:

- **ε (eps)**: the radius of the neighborhood.
- **minPts**: minimum number of points required within ε to consider a point a **core point**.

**Advantages:**
- Handles clusters of varying shapes and sizes
- Automatically determines number of clusters
- Robust to outliers

**Limitations:**  
- Sensitive to ε and minPts parameter choice
- Struggles with clusters of very different densities
- Can be computationally expensive for large datasets

#### Local Outlier Factor

Key Idea: Local Outlier Factor (LOF) identifies anomalies by comparing the local density of a data point to that of its neighbors. A point is considered an outlier if it resides in a sparse region compared to its neighbors — even if it's not far from them. LOF is especially effective in datasets with clusters of varying densities.

Intuition: LOF compares how "lonely" a point is compared to its neighbors. If a point's neighborhood is much sparser than its neighbors' neighborhoods, it's likely an outlier.

Anomaly Scoring:

LOF assigns a **score** to each data point that reflects how isolated it is relative to its neighbors. Here's how the score is computed:

1. **k-distance**  
For a point $( A )$, find the distance to its $k^{th}$ nearest neighbor.

2. **Reachability Distance**  
The reachability distance between point $( A )$ and its neighbor $( B )$ is defined as:

$[
\text{reach-dist}_k(A, B) = \max\left(\text{k-distance}(B),\ d(A, B)\right)
]$

This prevents sharp density spikes from skewing results.

3. **Local Reachability Density (LRD)**  
This is the inverse of the average reachability distance of point $(A)$:

$[
\text{lrd}_k(A) = \left( \frac{1}{|N_k(A)|} \sum_{B \in N_k(A)} \text{reach-dist}_k(A, B) \right)^{-1}
]$

4. **LOF Score**  
Finally, compare $( A )$’s density to its neighbors’:

$[
\text{LOF}_k(A) = \frac{1}{|N_k(A)|} \sum_{B \in N_k(A)} \frac{\text{lrd}_k(B)}{\text{lrd}_k(A)}
]$

- **LOF ≈ 1** → Normal (similar density to neighbors)  
- **LOF < 1** → Inlier (denser than neighbors)  
- **LOF > 1** → Outlier (sparser than neighbors)

Implementation Highlights:

- **Key parameters**:
  - `n_neighbors`: Number of neighbors $( k )$ (default is 20)
  - `contamination`: Proportion of expected outliers (for thresholding)
  - `novelty=True`: Use for **outlier detection on new/unseen data**

#### Isolation Forest

Key idea: Isolation Forest isolates anomalies directly, rather than modeling normal behavior. Since outliers are rare and distinct, they get isolated with fewer random splits. 

How it works:
- Build multiple Isolation Trees (iTrees) by repeatedly selecting a random feature and a random split value. 
- Trees split until stopping criteria are met (typically max depth or minimum samples)
- For each data point, compute the average path length across all trees. 

Anomaly scoring:
- Points that require fewer splits (shorter path lengths) to isolate are more likely anomalies.
- A formula converts path lengths into an anomaly score: values close to 1 indicate anomalies; scores near 0.5 or lower suggest normal behavior. 
- Formula: s(x) = 2^(-E(h(x))/c(n)) where E(h(x)) is average path length

Implementation highlights:
- You can tune parameters like contamination (to control the amount of anomalies you expect) when using implementations such as scikit-learn’s Isolation Forest.

### Anomalous Sound Detection

### Auto Encoders

An autoencoder is a type of neural network used for unsupervised representation learning. It learns to compress input data into a lower-dimensional representation (called the bottleneck) and then reconstruct the original input from this compressed form.

Autoencoders consist of three main components:

- **Encoder**: Transforms the input into a compressed latent space.
- **Bottleneck**: The most compact and informative representation of the input.
- **Decoder**: Reconstructs the input from the bottleneck representation.

The network is trained to minimize reconstruction loss, such as Mean Squared Error (MSE), between the original input and its reconstruction.


#### Variants of Autoencoders

Autoencoders come in many variations depending on the task and data type:

##### 1. **Denoising Autoencoder (DAE)**
- Trained to reconstruct clean inputs from noisy versions.
- Helps the model learn robust feature representations.

##### 2. **Sparse Autoencoder**
- Adds a sparsity constraint to the bottleneck (e.g. via L1 regularization).
- Encourages the network to learn only essential features.

##### 3. **Convolutional Autoencoder (Conv-AE)**
- Uses convolutional layers instead of fully connected layers.
- Effective for image data, as it preserves spatial structure.

##### 4. **Variational Autoencoder (VAE)**
- A probabilistic autoencoder that learns a distribution over the latent space.
- Useful for generating new data samples (e.g. images).
- Loss includes a KL divergence term in addition to reconstruction loss.

##### 5. **LSTM Autoencoder**
- Uses LSTM layers to handle sequence data like time series or text.
- Useful for tasks like anomaly detection in temporal data.


#### Applications of Autoencoders

- Dimensionality reduction
- Anomaly detection
- Image denoising
- Feature extraction
- Data compression
- Generative modeling (with VAEs)

### Tuning of Threshold

1. **Percentile-Based Thresholding**
- Choose the top `x%` of samples with the highest anomaly scores as anomalies.
- Example: Flag top 5% of points with the highest scores.

✅ Simple, no ground truth needed  
⚠️ Requires tuning based on tolerance for false positives

---

### 2. **Statistical Methods**
- Assume scores follow a known distribution (e.g., Gaussian).
- Flag values that fall outside a certain number of standard deviations (e.g., 3σ rule).

✅ Easy to implement  
⚠️ May fail if scores are not normally distributed

---

### 3. **Youden's J Statistic (from ROC Curve)**
- From the ROC curve, choose the threshold that maximizes:  
  **J = TPR − FPR**

✅ Balances sensitivity and specificity  
⚠️ Requires ground truth labels

---

### 4. **Precision-Recall Optimization**
- If you're more interested in catching true anomalies, optimize based on the **precision-recall curve** instead of ROC.

✅ More meaningful if anomalies are **extremely rare**  
⚠️ PR AUC is more sensitive to class imbalance

---

### 5. **Manual or Domain Expert Tuning**
- Set thresholds based on **business constraints**, expert review, or risk tolerance.

✅ Tailored to real-world cost  
⚠️ Not automatic or scalable

## Fabian's Reflection

Working with the **DCASE 2025 Anomalous Sound Detection (ASD)** dataset—specifically the bearing subset—has been an eye-opening experience in understanding the practical challenges and trade-offs involved in building anomaly detection systems. 

One of the most valuable lessons from this project was realizing that deep learning is not always the best solution, especially in anomaly detection tasks where data can be limited, imbalanced, or noisy. While deep models like fully connected autoencoders, convolutional autoencoders, and LSTM-based architectures are powerful, they did not consistently outperform simpler, classical models on this task.

For example, Local Outlier Factor (LOF),a density-based unsupervised model—performed surprisingly well on both spectrogram and mel-spectrogram features. It was able to capture subtle variations in local data structure that deep models struggled to learn or generalize. This outcome reinforced the idea that classical machine learning models can be highly competitive, especially when:
- The feature space is well-engineered or already meaningful (e.g., spectrograms),
- The dataset is small or highly imbalanced,
- Or when model interpretability and quick iteration are important.

This project taught me the importance of maintaining a model-agnostic mindset. Rather than defaulting to deep learning solutions, it’s more effective to experiment with a variety of models from classical algorithms to neural networks—and let the results guide the decision. Each model has its strengths, and the "best" model often depends heavily on the data characteristics and task constraints.

Working on the DCASE 2025 bearing dataset brought me back to the fundamental where understanding the data and thinking critically is more important than choosing a state of the art model without understanding its application. Simpler models with well-prepared features may outperform more complex deep architectures.