# Cleaner

Ensuring that no PHI is presented in the final database (for exporting datasets outside of a secure environment).

## Input
### Environmental Variable
These will override all the other variables passed.

* TOKEN_SAVE (string): the location where the token database is saved

### Model Args
These are the vairables passed through the model_args that are specific to this module.

* deidentify (boolean):  identifies if the module will be used or just skipped.
* min_doc_length (int):  indictates the smallest amount of characters that will be allowed after
                            whitespace is removed.
* remove_dups (boolean): should duplicate ids be removed? "False" will stop the run if found. 

## Output
### Cache
These are alll the files found in the generated cache

* df_tokens (pd.DataFrame[pickle in chunks]): the tokenized data.
* deps_package (dict): the package that generates the cache
* describe.txt (str): description of what the module produces
* filled_files.md5 (class): the file used to test the cache for a completed build
* id_map.json (dict): the map for all the ids
* token_dump.txt (str): database dump
* word_map.json (dict): the map for all the words

## module calls

* build_from_raw:   needed to run the pipeline
* get_cached_deps:  get the pregenerated arguements
* tokenize_text:    This function will convert words to intergers with sentences and keep a running map from words to intergers.
* tokenize_ids:     This function will convert identifing keys to intergers with sentences and keep a running map from words to intergers.
* make_token_dump:  This function will create a tokenized database dump text.
* describe:         generates a description of what what built during the pipeline build

## command line calls

using `python cleaner.py <token> <category> <cache_path>` you will be able to convert a token into the appropriate value

### options
These are the same as the `applicalble model_args` listed above and can be generated using `python cleaner.py -h`

```
positional arguments:
  token       integer expecting to be identified
  category    the category the token comes from. "id" is a case identifier, "word" is an element of sentences
  cache_path  the location from which the maps will be pulled.
```

## test_suite

within the testing directory there is a test suite that will check the following:
* pipeline calls
* expected file structure
* model arg options

This will use the database in the `testing/data/` directory which includes:
* dfclean: pickled database

### test suite call

after navigating to the testing directory
`python test_suite.py`
