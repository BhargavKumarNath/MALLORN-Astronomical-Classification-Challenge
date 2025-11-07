# MALLORN Astronomical Classification Challenge

## *Final Public Leaderboard Score: 0.5353 (7th Place) (Date: 7 Nov, 2025)*

This repository hosts the full solution for the MALLORN Astronomical Classification Challenge. The project followed an iterative data science journey, starting with classical machine learning, exploring deep learning approaches, and finally achieving the best results by circling back to a finely tuned gradient boosting model enhanced with advanced feature engineering.

# 1. Problem Statement
The objective of this challange was to develop a machine learning model capable of identifying rare **Tidal Disruption Events (TDEs)** from simulated multi-band astronomical lightcurve data. The dataset was designed to emulate observations from the upcoming Vera C. Rubin Observatory's Legacy Survey of Space and Time (LSST).

## The Dataset
The data is comprised of 2 main componenets:
* **Metadata Files (`train_log.csv`, `test_log.csv`):** These files contain object IDs and static features for each astronomical object, such as redshift (`Z`) and galactic dust extinction (`EBV`).

* **Lightcurve Files:** Located in 20 sub-folders, these files contain the core time-series data: brightness (Flux) measurements taken at specific times (`Time (MJD)`) through six different optical filters (`u`, `g`, `r`, `i`, `z`, `y`).

## The Challange
The primary challange was a binary classification problem characterized by:

* **Severe Class Imbalance:** TDEs are extremely rare, representing only **4.86%** of the training set.

* **Sparse and Irregular Time-Series:** The lightcurves for each object have a variable number of observations sampled at irregular time intervals.

* **Evaluation Metric:** The competition was evaluated on the **F1 Score**, which is well-suited for imbalanced classification as it balances precision and recall.

# 2. Approach Overview
Our approach was an agile and interative loop of modeling, diagnosis, and strategic pivoting. 

1. **Baseline Modeling:** We began with a classical machine learning approach, using a LightGBM model on a set of simple, manually engineered statistical features.

2. **Advanced Feature Engineering:** To improve performance, we used the `tsfresh` library to automatically extract hundreds of time-series features. A key strategic decision was to generate these features on a **per-filter basis**, which proved to be highly effective.

3. **Hyperparameter Tuning:** We used the `Optuna` framework to perform a search for the optimal LightGBM hyperparameters, which gave our best model.

4. **Deep Learning Exploration:** To determine if end-to-end sequence modeling could outperform feature engineering, we developed two deep learning models in PyTorch:

    * A single-channel GRU with a Bahdanau-style attention mechanism.
    * A more complex multi-channel GRU with six parallel encoders, one for each filter.

5. **Final Diagnosis & Submission:** After rigorously testing the deep learning models and finding them to be significantly inferior, we concluded that the dataset's sparse and low-volume nature was better suited to a feature-engineering approach. We generated our final submission using the best-performing LightGBM pipeline.

# 3. Machine Learning Techniques
## Models & Features Sets

| Model    | Feature Set                     | Mean CV F1 Score | Key Insight                                             |
|----------|---------------------------------|-----------------|--------------------------------------------------------|
| LightGBM | Basic Statistical Aggregates     | 0.4281          | Threshold optimization is critical.                   |
| **LightGBM** | **Per-Filter `tsfresh` Features**      | **0.5225**          | **Best model.** Capturing per-filter dynamics is key.     |
| LightGBM | `tsfresh` + Interpolated Colors    | 0.4974          | Interpolating sparse data creates noise and degrades performance. |
| LightGBM | `tsfresh` + Robust Color Features  | 0.5205          | `tsfresh` had likely already captured the color signal. |

# Final Model: LightGBM with Autoated Feature Engineering
Our most successful model was a LightGBM classifier. Its performance was driven by a robust feature set and methodical tuning.

* **Feature Engineering:** We used the `tsfresh` library with `EfficientFCParameters` to generate features for each of the 6 optical filters independently. This created a rich tabular dataset that captured the unique temporal behavior within each color band. The `tsfresh.select_features` function was then used to select the **198 most statistically relevant features** based on the training data.

* **Hyperparameter Tuning:** A 50-trial study was conducted using **Optuna** to find the optimal hyperparameters for our LightGBM model. The best parameters were:

    `{'learning_rate': 0.0361, 'num_leaves': 120, 'max_depth': 11, ...}`

* **Feature Importance:** The model identified redshift (`Z`) and features related to signal-to-noise (`Flux_Ratio_skew`) and minimum brightness levels (`g_Flux_min`, etc.) as the most predictive, confirming that both static metadata and lightcurve shape are crucial for classification.

# 4. Deep Learning Approach
In an attempt to further improve our score, we pivoted to deep learning to see if a model could learn temporal patterns directly from the raw sequences.

## Architectures
1. **Single-Channel GRU with Attention:** This model processed a flattened sequence of all observations. It used a bidirectional GRU to learn temporal patterns and an attention mechanism to focus on the most relevant timesteps before classification.

2. **Multi-Channel GRU:** A more sophisticated model with six parallel bidirectional GRU encoders. Each encoder was an "expert" for a single filter (`u`, `g`, `r`, etc.). The final hidden states from all six encoders were concatenated with the static metadata and passed to a final classifier.

## Diagnosis of Performance
Despite proper data pre-processing (per-object scaling, relative time encoding) and a robust architecture, the deep learning models performed poorly, achieving a maximum F1 score of ~0.18.

**The reasons for this were twofold:**
* **Data Sparsity:** Standard RNNs like GRUs are designed for dense, regularly sampled sequences. They struggle to find patterns in sparse, irregularly sampled astronomical data.

* **Low Data Volume:** With only ~150 positive TDE examples in the training set, the deep learning models did not have sufficient data to learn the complex, non-linear patterns of a TDE lightcurve from scratch.

This experimental phase was a crucial diagnostic step, proving that for this particular problem, feature abstraction was a more effective strategy than sequence modeling.

# 5. Final Solution Pipeline
The final submission was generated using our best LightGBM pipeline, which can be summarized as follows:

1. **Load Data:** Load all `train_log.csv`, `test_log.csv`, and their corresponding lightcurve files.

2. **Feature Engineering:** Generate per-filter time-series features for the combined dataset using `tsfresh`.

3. **Feature Selection:** Isolate the training data features and use `tsfresh.select_features` to identify the most relevant feature columns.

4. **Data Preparation:**
    * Split the data back into training and test sets using the selected feature columns.

    * Merge static metadata (`Z`, `EBV`).

    * Sanitize feature names to be 
    compatible with LightGBM.
    * Scale all features using `StandardScaler` fitted only on the training data.

5. **Threshold Optimization:** Perform 5-Fold Stratified Cross-Validation on the training data to determine the mean optimal probability threshold (~0.35) for maximizing the F1 score.

6. **Final Training & Prediction:**
    * Train the Optuna-tuned LightGBM model **on 100% of the training data.**

    * Predict probabilities on the prepared test set.

    * Apply the optimal threshold to the probabilities to generate the final binary predictions.

7. **Submission:** Format the predictions into the required `submission.csv` file.

# 6. Technical Details
* **Libraries:** `Pandas`, `NumPy`, `Scikit-learn`, `LightGBM`, `tsfresh`, `Optuna`, `PyTorch`, `Matplotlib`, `Seaborn`.

* **Validation Strategy:** 5-Fold Stratified Cross-Validation was used throughout the project to ensure robust evaluation despite the heavy class imbalance.

* **Class Imbalance Handling:** The `scale_pos_weight` parameter in LightGBM was used to give more importance to the minority class (TDEs) during training.

* **Code Structure:** The project is organized into modular Python scripts for each stage: initial exploration, feature engineering, modeling, and submission generation.
