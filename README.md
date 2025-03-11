# HGAT-STS
Similarity Calculation Based on Heterogeneous Graph Attention Networks (HGAT)
# HGAT-STS

## Setup
1. Environment Setup:

Install Python 3.6 or higher.

The necessary libraries to be installed are already listed in the environment_requirements.txt document.

2. Data Preparation:

Place the dataset in the dataset/ directory.

Ensure the data format is correct (e.g., .txt). 

The data content must be in the form of syntactic tree structures, for example:
entailment	( ( Twenty ( five people ) ) ( ( are marching ) . ) )	( ( A ( big group ) ) ( ( marches around ) . ) )	(ROOT (S (NP (CD Twenty) (CD five) (NNS people)) (VP (VBP are) (VP (VBG marching))) (. .)))	(ROOT (S (NP (DT A) (JJ big) (NN group)) (VP (VBZ marches) (ADVP (RB around))) (. .)))	Twenty five people are marching.	A big group marches around.	4871633378.jpg#3	4871633378.jpg#3r1e	entailment	entailment	entailment	entailment	neutral
The above example is based on reading data from the SNLI corpus. If other datasets are required, the data loading program can be modified accordingly. However, the input sentences must be in syntactic tree structures.

To use a different dataset, modify the data loading script while ensuring the input format remains consistent.

For retraining models, delete cached files in the dataset/ directory.

In the dataset/ directory, we have split the training set into six parts to facilitate data preprocessing. If the dataset is small, it can be consolidated into a single file.

3. Hardware:

A GPU is recommended for faster training and inference. Ensure sufficient memory (128GB or more recommended).


4. Logging:

Log files are generated during runtime and stored in the lightning_logs/ directory. Check these files for debugging and progress tracking.

5. Pre-trained BERT Model

The pre-trained BERT model must be placed directly in the root directory. We are currently using bert-base-uncased. If changes are required, please ensure that the corresponding code references are updated accordingly.

Notes for Running the Code:

During the first run, the data needs to be loaded, which may take a significant amount of time if the dataset is large. Subsequent runs will be much faster as the data is already loaded and cached.




## Run code

### Training
```sh
python main.py
```

### Testing
```sh
python main.py --mode test --ckpt_path [your_ckpt_file_path]
```
