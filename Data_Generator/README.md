# Data_Generator

This module will capture necessary lists and dictionaries needed for the model.  This is currently limited to the vocabulary, the tasks and expected predictions based on rule based alterations to the data.

# Input
## Environmental Variable

None

## Model Args
These are the variables passed through the model_args that are specific for this module.

* use_security (boolean): indication that the vocabulary will be restricted to only a restricted.
* use_w2v (boolean): indication of whether to use the word2vec module (the alternative is randomizing the input layer).
* concat_split (boolean): this will combine the records together based on how they were split (split_type) ##THIS IS BROKEN
* vocab_source (string): indication of where the w2v data is coming from.  Options are None in which case the train split will be used to generate data and randomize the input layer, "raw" indicating the cleaned (unfiltered) records, "train_val_test" indicating indicating that all splits will be used or "train" will utilize only the train split.
* min_label_count (int): the amount of occurances a label must appear within all the reports to be captured in the labels list. 1 means capture all of them.
* w2v_kwargs (dict): various arguments for w2v. These can include "size", "min_count", "window", "workers", "iter", and anything else that may be needed.
* np_seed (int): sets the random seed for numpy. -1 will keep it random.

## output
### cache
these are alll the files found in the generated cache

* deps_package (dict): the package that generates the cache
* describe.txt (str): description of what the module produces
* filled_files.md5 (class): the file used to test the cache for a completed build
* id2labels_foldx.json (dict): the label mapping for each task for fold x
* id2word_foldx.json (dict): the dictionary created for fold x
* word_embeds_foldx.npy (2d numpy array): the first layer of the model for fold x

## module calls

* build_from_raw:        needed to run the pipeline
* get_cached_deps:       get the pregenerated arguements
* check_args:            Tests the args provided against the restrictions provided
* build_vocab:           Will generate the X values for the model and the input map
* run_w2v:               Train a word2vec model which will generate an input layer and a vocab.
* randomize_layer:       Create the input layer through randomization.
* build_labels:          Will generate the Y values for the model and the output layer
* describe:              Creates a report of what the filter did.

## command line calls

There are no calls with this script

## test_suite

Within the testing directory there is a test suite that will check the following:
* pipeline  call
* expected file structure
* outputs
* model arg options

This will use the files in `testing/data/` directory which includes:
* fold:            pickled list of dataframes that included the IDs for each fold
* schema:          dictonary file to describe the fields of the test_df
* test_df:         pickled dataframe

### test suite call

after navigating to the 'testing' directory
`python test_suite.py`

