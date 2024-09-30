# LPLM: A Neural Language Model for Cardinality Estimation of LIKE-Queries

LPLM (LIKE Pattern Language Model) is a deep learning-based approach designed to estimate the cardinalities of SQL LIKE-queries. This model leverages neural language models to predict the cardinalities of LIKE queries, which can then be used to optimize query execution plans in databases. LPLM operates in three main steps:

---

## 1. Prepare Training Datasets

To train LPLM, you first need a dataset of SQL LIKE-patterns. This step generates the LIKE-patterns that will be used to train the model.

**Instructions:**
- Run the `prepare_training_data.py` script to generate the LIKE patterns training dataset.

---

## 2. Compute Ground Truth Probabilities

Once the LIKE patterns are generated, the next step is to compute the ground truth cardinalities for these patterns. These cardinalities represent the actual selectivity of the LIKE-queries when executed on a database.

**Instructions:**
- Generate a database for each dataset by 'running createdb.py'.
- Run the `compute_ground_truth.py` script to compute the ground truth cardinalities for the generated LIKE patterns.
  
  **Details:**
  - LPLM uses the SQLite library to interact with databases and compute these probabilities.
  - Computing ground truth cardinalities for LIKE-queries is time-consuming and resource-intensive. To mitigate this, the process is parallelized using multiple processes, each working with different database instances.
  - We use 1000 different SQLite databases for efficient and scalable computation.

---

## 3. Train the Model and Get Estimated Cardinalities

The core of LPLM is its neural language model, which is trained using the LIKE-patterns and their corresponding ground truth cardinalities. Once trained, the model can estimate cardinalities for unseen queries.

**Instructions:**
- Use the `main.py` script to train the model and obtain estimated cardinalities.
  
  **Details:**
  - The model can either be trained from scratch or reloaded from a previously saved model.
  - Once the model is trained, it can be used to estimate the cardinalities of test queries.
  
---

## Injecting Estimated Cardinalities into PostgreSQL

To use the estimated cardinalities for query optimization in PostgreSQL, follow these steps:

**Instructions:**
1. Apply the provided `benchmark.patch` to modify the PostgreSQL codebase to accept estimated cardinalities.
2. Follow the instructions from the [End-to-End-CardEst-Benchmark](https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark) to integrate and benchmark the selectivities in PostgreSQL.
   
   **Details:**
   - Our modifications support the injection of selectivities for both LIKE and equality patterns into PostgreSQL's query planner.

---

## Description of Hardware Needed
All our experiments were performed on a machine with two NVIDIA RTX 3090 24
GB GPUs, AMD Ryzen Threadripper 3970X CPU and 256 GB of RAM, Ubuntu 20.04 LTS. We used
PyTorch 1.10.1 for building the deep-learning models.


## Repository Structure

- `prepare_training_data.py`: Script to generate LIKE-patterns for training.
- `compute_ground_truth.py`: Script to compute ground truth cardinalities.
- `main.py`: Script to train the model and estimate cardinalities.
- `benchmark.patch`: Patch for PostgreSQL to inject selectivities.
  
---

## Software/Library Dependencies 

- **Python** (3.x)
- **SQLite** (for computing ground truth probabilities)
- **PyTorch** (for model training)
- **PostgreSQL** (for integration of estimated cardinalities)

