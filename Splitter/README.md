# Splitter

Uses the filtered dataset to create the training sets, validation sets, and test set for the model that will be run/trained.

## Input
### Environmental Variables

None

### Model Args

* split_type (string):       at which level are the reports split?  The current options are "case" and "record" .
* by_registry (boolean):     dictates if the splits are done by registry or as a full set.
* test_split (list):         identification of how the test split will be withheld. It will take the form (how, value).
                              The "how" is currently restricted to "percent" (a flat percent with the value, v, such that 
                              0 < =v < 1), "year" (where the focus year will be greater than the value) and "cv" (which 
                              will divide the dataset by K sets >2 and rotate through each set).
* val_split (list):          identification of how the validation split will be designated. It will take the form (how, value).
                              The "how" is currently restricted to "percent" (a flat percent with the value, v, such that 
                              0 <= v < 1) and "cv" (where the value will be the amount of folds).
* sklearn_random_seed (int): The seed at which sklearn will genrate the splits.

## Output
### Cache
These are all the files found in the generated cache

* train# (pd.DataFrame[pickle in chunks]): list of ids for the train split in fold #
* val# (pd.DataFrame[pickle in chunks]): list of ids for the val split in fold #
* test# (pd.DataFrame[pickle in chunks]): list of ids for the test split in fold #
* deps_package (dict): the package that generates the cache
* describe.txt (str): description of what the module produces
* filled_files.md5 (class): the file used to test the cache for a completed build
* fold_num.txt (str): number of folds included in the build

## module calls

* build_from_raw:        needed to run the pipeline
* get_cached_deps:       get the pregenerated arguements
* check_args:            Tests the args provided against the restrictions provided
* cv_split:              Splits the dataset provided using cross validation
* test_year:             Splits the test set by year (more recent will be the test set)
* percentage_split:      Splits the dataset provided using a pecentage of the filtered data.
* split_data:            Groups the data and pass to the appropriate calls
* describe:              generates a description of what what built during the pipeline build

## command line calls

There are no calls with this script.

## test_suite

within the testing directory there is a test suite that will check the following:
* pipeline calls
* expected file structure
* outputs
* model arg options

This will use the database in the `testing/data/` directory which includes:
* test_df:          pickled dataframe
* test_scheam.json: dictonary file to describe the fields of the test_df

### test suite call

after navigating to the 'testing' directory
`python test_suite.py`
