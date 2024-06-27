# Encoder

The produces the reports split into train, test, and val in a form that is readable by the model.

# Input
## Environmental Variable

None

## Model Args
These are the variables passed through the model_args that are specific for this module.

* metadata_fields (list): List of all the names of the metadata desired. They must be expected to match the names found in the in the sanitized database. Dates are are both listed as DATEFIELD_diff and DATEFIELD_KEYDATEFIELD_diff.
* remove_unks (boolean): This will remove the unknown tokens from the reports (as opposed to mapping it to an unk term).
* reverse_tokens (boolean): This will reverse the documents to read the last tokens first.
* sequence_length (int): the max length of the encoded reports.

## Output
### Cache
These are all the files found in the generated cache

* testMetadata_foldx (pd.DataFrame[pickle in chunks]): The additional data that is included in the final dataset for the test split
* trainMetadata_foldx (pd.DataFrame[pickle in chunks]): The additional data that is included in the final dataset for the train split
* valMetadata_foldx (pd.DataFrame[pickle in chunks]): The additional data that is included in the final dataset for the validation split
* testY_foldx (pd.DataFrame[pickle in chunks]): the tokenized Y values that will be used in the final dataset for the test split
* trainY_foldx (pd.DataFrame[pickle in chunks]): the tokenized Y values that will be used in the final dataset for the train split
* valY_foldx (pd.DataFrame[pickle in chunks]): the tokenized Y values that will be used in the final dataset for the validation split
* testX_foldx.npy (2d numpy array): the tokenized reports that will be used in the final dataset for the test split
* trainX_foldx.npy (2d numpy array): the tokenized reports that will be used in the final dataset for the train split
* valX_foldx.npy (2d numpy array): the tokenized reports that will be used in the final dataset for the validation split
* deps_package (dict): the package that generates the cache
* describe.txt (str): description of what the module produces
* filled_files.md5 (class): the file used to test the cache for a completed build

 sequence_length (int): Max token length that will be used for the records.

## module calls 

* build_from_raw:        needed to run the pipeline
* get_cached_deps:       get the pregenerated arguements
* check_args:            Tests the args provided against the restrictions provided
* generate_inputs:       Generate the tokenized inputs and metadata for each of the splits
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
* test_df:         pickled dataframe
* fold:            pickled list of dataframes that included the IDs for each fold

### test suite call

after navigating to the 'testing' directory
`python test_suite.py`
