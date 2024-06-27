# Filters

Uses the indexes from the sanitized data to filter based on the predetermined criteria.

## Input
### Enviornmental Variable

None

## Model Args
These are the variables passed through the model_args that are specific for this module.

* tasks (list): this will be the list of single or combined tasks that will be predicted.
* only_single (string): Whether to accept only records that are unique to the patient, case, 
                          or none. Respective options: "patient", "case", "none".
* include_only_scores (dict): restrict to specific scores in each field.
* window_days (list): the beginning and end date of the date filter. Single positive digit 
                          in list is +/-. [0] removes the filter.
* window_fields (list): the date fields included in the filter.
* min_year (int): the minimum year that is allowed for the reports.
* max_year (int): the maximum year that is allowed for the reports. If "0" or greater than
                      the present year, then the max year defaults to the current date.

## module calls

* build_from_raw:        needed to run the pipeline
* get_cached_deps:       get the pregenerated arguements
* check_args:            This will test the schema against the args fields.
* filter_by_single:      module filters out due to the single category
* filter_by_window:      module filters out based on the date window
* filter_by_field:       module filters out based on a custom fields restrictions
* filter_gold:           module filters out based on the what is needed for the gold values
* filter_by_early_date:  module filters out the records that occur before the given year
* filter_data:           main function used to filter the data
* describe:              generates a description of what what built during the pipeline build

## Output
### Cache
These are all the files found in the generated cache

* filtered_ids (pd.DataFrame[pickle in chunks]): the ids of reports that remain.
* missing_tasks (pd.DataFrame[pickle in chunks]): the tasks that are generated for each id.
* deps_package (dict): the package that generates the cache
* describe.txt (str): description of what the module produces
* filled_files.md5 (class): the file used to test the cache for a completed build

## module calls

There are no calls with this script

## test_suite

within the testing directory there is a test suite that will check the following:
* pipeline calls
* expected file structure
* model arg options

This will use the database in the `testing/data/` directory which includes:
* test_df:          pickled dataframe
* test_scheam.json: dictonary file to describe the fields of the test_df

### test suite call

after navigating to the 'testing' directory
`python test_suite.py`
