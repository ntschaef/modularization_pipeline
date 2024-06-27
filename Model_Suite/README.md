# Modular implementation for model building, training, and evaluation.

## Quickstart Guide

### Initial Setup
Clone the repo
```
git clone https://ncigitlab01.repd.ornlkdi.org/nci/model_suite_v2.git
```
Load the working branch and pull in the subrepos
```
cd model_suite_v2
git checkout dev
```

It is not required to setup a conda env to train a model, the slurm scripts call a shared environment on `/mnt/nci/scratch`. 
If you want your own local working envorinment, the following commands will create one.
Setup the conda environment (the default name for the environment is "nsconda39", this can be edited in the nsconda39.yml file)
```
conda env create --file nsconda39.yml
conda activate nsconda39
```

### Model Training
The model_args.yaml file controls the settings for model training. Edit this file as desired. The `"save_name"` entry controls the name used for model checkpoints and prediction outputs; if left empty, a datetimestamp will be used.

#### Model args

We will detail the args most commonly changed for various tasks:
- `save_name`: controls the directory and file names of saved models and output files
- `tasks`: list of tasks
- `data_path`: absolute path to data folder
- `subset_proportion`: if less than 1, will generate a random subset tfor rapid prototyping
- `reproducible`: sets random number generator seeds for torch, numpy, and python
- `random_seed`: positive int or blank (`None`) for random number generator, ignored if `reproducible = False`
- `max_epochs`: maximum number of epochs to train
- `patience`: number of epochs without improving val loss to stop training
- `class_weights`: either blank, dictioary, or path to pickle file with dictioary of weights. If a dictionary, keys are tasks and values are lists fof weights
- `abstain_flga`: True or False, are we training with the DAC or not?
- `ntask_flag`: are we training with Ntask enabled?
- `ntask_tasks`: list of tasks for Ntask to check, must be a subset of `tasks`
 
The data used must be generated from the existing `modularization_pipeline` codebase. Add the path to existing data in the the 
`data_path` argument. You will also need to set the `tasks` and `class_weights`, if using, arguments. The latter will take either
a dict with keys corresponding to the task and value with a corresponding list of weights for that task.  If `class_weights` is None (blank), no class weights will be used.  

The default slurm script `new_IE.sh` is set to the information extraction task, with settings that
will train a model. This command is
```
python train_model.py -m ie
```
If you're wanting a case-level context model, change the above line in the slurm script, `new_IE.sh`, to 
```
python train_model.py -m clc
```


The default setting, if the `-m ` argument is omitted, is information extraction, and which task is specified in the `model_args` file.


#### Data Requirements

You may create your own custom datasets for model training, provided that the necessary files:
- data_fold0.csv,
- id2label_fold0.json,
- id2word_fold0.json,
- metadata.json,
- query.txt,
- schema.json, and
- word_embeds_fold0.npy
are present in the data directory listed in the `model_args.yml` file. For example, suppose that `data_fold0.csv` was created through the usual pipeline with
6 different registries data. You want to look at a specific subset relating to only one registry. Extracting the relevant rows of the DataFrame
into a new DataFrame, saving the new DataFrame in a different directory, and copying over the other fiels from the original directory is 
sufficient to train a model on the subset of interest, without re-running the datageneration pipeline. 


### use_model.py
To use a pretrained model to predict on a nother dataset, add the following line to your slurm script
```
python use_model.py -mp path/to/saved/model.h5 -dp path/to/data/folder/
```
Note in the above call that the model path specifies an h5 file and the data path
specifies a directory (and must terminal with /).


## Model_suite.py
    
    Main File 

    0) World Initialization (GPUs)
    1) Load Model Args and Valid
    2) Load the Data 

    Option 1
    3a) Create a Model
    3b) Train the Model
    3c) Save the Model
    Option 2
    3) Load pretrained Model

    4) Evaluate and Score 
    5) Save the output files 

#### Notes Specific to Transformers:
 1) Argument Parser stuff into model args
 
 3a) Needs to include data parallel stuff 
 3c) Saving Model based on rank 
 
 3) Reloading need to figure out GPU to CPU compatibility 

### 0) World Initialization (GPUs)
   Configs specific to device
    Automaticatic figuring out

### 1) Load Model Args and Valid : validate_params.py 
    validating model_args, and updating if neccessary
    assertions are done here

### 2) Load the Data and Valid  : (equivalent of PathReports Class) don't use dataloaders as name
    Input : data path from model_args
    Add an index in Data sample in dataloaders which helps us join to metadata at later stage
    Output: train, test and val torch Dataloaders returned , metadata
    Model_args: data_path, fold_number, max_len, tasks (list of strings), batch_size, subset_proportion (sampling_rate), random_seed, add_noise_flag, add_noise_parameters, multilabel_flag

    Extra Dependent Classes: Mutual Information, Keywords, Abstention {add a class}, Multilabel{one hot encoded Ys}

### 3) Option 1

#### 3a) Create a Model : model directory 
    Input requirements : model_args from Step 1
    Note: Common and Unique args need to be identified 
                    MTCNN model file 
                    MTHisan model file
                    Transformers model file

                    Function List for Model File
                    i) Initialization
                    ii) Architecture Definition (inheritance from torch.nn.module)
                    iii) Forward Function (return additional returns based on requirements)
                    iv) Model Parallelization (will need info from step 0)
                        - DDP in Citadel
                        - DataParallel in Enclave (fallback for DDP)
                        - CPU
    Output: model object returned

    Model_args: model_flag, MTCNN args, MTHisan args, Transformers args, random seed, #epochs, patience, keyword_flag, class_weights_flag, multilabel(softmax/sigmoid)

    Extra Dependent Classes: Keywords, Class_Weights(number of classes, task), MTCNN , MTHisan , Transformers

#### 3b) Train the Model (Separate class)
    Input : Model from 3a, dataloaders from 2, training specific arguments from 1
    i) Loss function declaration 
    ii) Optimizer function declaration
    iii) For through batchs
        - Explicit call to Forward Function (not model(x)) , ability to add additional params & returns with forward
        - Loss calculation
        - Val Loss Calculation
        - Print diagnostics 
        - Based val loss, calculate patience
    iv) If data parallel, loss needs to all reduced
    v) Return metadata, all this can be seein with `cat job.<job#>.out`
            Metadata includes the following        
                - Epoch Count
                - Train Loss
                - Val Loss
                - Per Epoch , train and val loss
            Note: Try bringing in metrics class from torch or other to make this consistent 
    Output: metadata , model gets updated during the process 



#### 3c) Save the model (in main file)


### 3) Option 2

#### Load the model (in main file)






### 4) Evaluate and Score (Separate Class)
    Input : Model from 3, dataloaders from 2 
    Have this function the ability to call within epoch and for inference.

    Model is set as eval. 

    Metrics:
    1) F1 Micro and Macro
    2) Precision
    3) Recall
    4) Accuracy

    Output: 



### 5) Save the output files 
    - Train Val Test Scores 
    - Epoch level metrics (optional)
    - Softmax scores: Note that metadata needs to be joined actual data (optional) (True and Prediction Labels plus softmax scores)
    - True and Prediction Labels : Note that metadata needs to be joined actual data
    - Model_args from step 1
    - Carry over Data_args plus decribe.txt (data origin story)
    - Metadata from step 3b
    - Leaderboard
        Metrics
        Metadata ie git hash keys
        Location data and models



    Metadata needs recordId and registry. Make a note of the format of recordId to match it with ctc.
