# LPLM: A Neural Language Model for Cardinality Estimation of LIKE-Queries

LPLM has three main steps.

    Prepare training datasets.
    --- To create LIKE patterns training data set, you have to run prepare_training_data.py.

    Compute ground truth probabilities.

    --- To compute the ground truth cardinalities, you have to run compute_ground_truth.py file. 
    --- As a database, we used SQLite library that provides a lightweight disk-based database from Python.
    --- Computing ground truth probabilities of LIKE-patterns is timely and expensive. Therefore, we use multi processes with 1000 different databases.

    Train the Model and get estimated cardinalities.

    --- LPLM learns from the LIKE-Patterns and ground truth probabilities. main.py has the commands to train the Model and get the estimated cardinalities. 
    --- The Model can be trained from scratch or reloaded from a previously saved model to estimate the cardinalities of test queries.
    
To inject the estimated cardinalities to PostgreSQL:

    --- First apply benchmark.patch
    --- Second, follow instructions given in 
        https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark
    --- Our modifications accept selectivities for LIKE and equality patterns.
