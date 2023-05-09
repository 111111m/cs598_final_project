# cs598_final_project

1	Citation to the original paper
https://github.com/mp2893/gram

2	Link to the original paper’s repo (if applicable)
https://github.com/mp2893/gram

3	Dependencies

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import argparse
import os
import time
import torch.nn.functional as F
from collections import OrderedDict

4	Data download instruction

We chose to use the same MIMIC-III dataset that the paper originally used. The model only used the admissions data and the Diagnosis ICD data from MIMIC-III. The MIMIC-III v1.4 dataset contains 58,976 admissions to various hospitals between June 2001 and October 2012. The important pieces of information from this admission dataset happen to be the Diagnosis data that can have any of the possible 15,693 different diagnosis codes. GRAM requires more information about the ancestors of these codes through a IC9 to multi-level CSS mapping and the provided re-mapping script to build out the trees of the DAG. The result of the data-prep is pickle files corresponding to the different levels of the tree. We obtained access to the data set through https://physionet.org/.

5	Preprocessing code + command (if applicable)

python process_mimic.py ADMISSIONS.csv DIAGNOSES_ICD.csv out1

python build_trees.py ccs_multi_dx_tool_2015.csv out2 out1.seqs out1.3digitICD9.seqs

6	Training code + command (if applicable)

python gram_original.py out2.seqs out1.3digitICD9.seqs mimic_out2

7	Evaluation code + command (if applicable)

python gram_original.py out2.seqs out1.3digitICD9.seqs mimic_out2

8	Pretrained model (if applicable)

python create_glove_compap.py out2 out1.3digitICD9.seqs
python gram_original.py out2.seqs out1.3digitICD9.seqs out2 --embed_file out32f.48.npz 128
•	Table of results (no need to include additional experiments, but main reproducibility result should be included)
)
In the original paper

the reproducibility scope is in the original paper too. Some of the files are from the original repo cause that's not our reprocibility scope. 
This repository only includes part of the weights, some other weights files are too larget to upload. 
