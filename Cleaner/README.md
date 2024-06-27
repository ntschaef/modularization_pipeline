# Cleaner

This class will clean the text and store the results for other modules.

## Input
### Environmental Variable

None for the module

### Model Agrs

* cc (class):                      the CachedClass will be passed through all the modules
                                         and keep track of the pipeline arguements.
* remove_whitespace (boolean):     combine groups of spaces into one space?
  * EXAMPLE - remove_whitespace: True
* remove_longword (boolean):       remove words over 25 characters in length?
  * EXAMPLE - remove_longword: True
* remove_breaktoken (string):      indicates which punctuation will be removed: none, dups, or all.
  * EXAMPLE - remove_breaktoken: "dups"
* remove_punc (string):            indicates which punctuation will be removed: none, dups, most, or all.
  * EXAMPLE - remove_punc: "most"
* lowercase (boolean):             make all alpha characters lowercase?
  * EXAMPLE - lowercase: True
* convert_breaktoken (boolean):    convert all \n \r and \t to "breaktokens"
  * EXAMPLE - convert_breaktoken: True
* convert_escapecode (boolean):    convert "\x??" tokens to "\n" or " " (instead of being left alone)
  * EXAMPLE - convert_escapecode: True
* convert_general (boolean):       convert numbers with decimals to "floattoken" and large number (>=100) to "largeinttoken"
  * EXAMPLE - convert_general: True
* stem (boolean):                  regularize terms into their "root" category (aka use stemmer package)
  * EXAMPLE - stem: False
* fix_clocks (boolean):            regularize the "time" terminology.
  * EXAMPLE - fix_clocks: False

## Output
### Cache
These are all the files found in the generated cache

* df_cleaned (pd.DataFrame[pickle in chunks]): the cleaned data.
* deps_package (dict): the package that generates the cache
* describe.txt (str): descriptiBon of what the module produces
* filled_files.md5 (class): the file used to test the cache for a completed build

## module calls

* build_from_raw:   needed to run the pipeline
* get_cached_deps:  get the pregenerated arguements
* clean_text:       clean the text from a string
* clean_data:       clean the text from a pandas DataFrame
* describe:         generates a description of what was built during the pipeline build

## command line calls

using `python cleaner.py <text> --<option>` you will be able to clean the text that you input with the options you list

### options
These are the same as the `applicalble model_args` listed above and can be generated using `python cleaner.py -h`

```
  --convert_breaktoken CONVERT_BREAKTOKEN, -cb CONVERT_BREAKTOKEN
                        convert all \n \r and \t to "breaktokens" (default: False)
  --convert_escapecode CONVERT_ESCAPECODE, -ce CONVERT_ESCAPECODE
                        convert "\x??" tokens to "\n" or " ". (default: False)
  --convert_general CONVERT_GENERAL, -cg CONVERT_GENERAL
                        convert numbers with decimals to "floattoken" and large number (>=100) to "largeinttoken". (default: False)
  --fix_clocks FIX_CLOCKS, -fc FIX_CLOCKS
                        regularize the "time" terminology. (default: False)
  --lowercase LOWERCASE, -lc LOWERCASE
                        make all alpha characters lowercase? (default: False)
  --remove_breaktoken REMOVE_BREAKTOKEN, -rb REMOVE_BREAKTOKEN
                        indicates which punctuation will be removed: none, dups, or all. (default: none)
  --remove_longword REMOVE_LONGWORD, -rl REMOVE_LONGWORD
                        remove words over 25 characters in length? (default: False)
  --remove_punc REMOVE_PUNC, -rp REMOVE_PUNC
                        indicates which punctuation will be removed: none, dups, most, or all. (default: none)
  --remove_whitespace REMOVE_WHITESPACE, -rw REMOVE_WHITESPACE
                        combine groups of spaces into one space? (default: False)
  --stem STEM, -s STEM  regularize terms into their "root" category. (default: False)
  --logging LOGGING, -l LOGGING
                        this is the logging level that you will see. Debug is 10 (default: 20)
```

## test_suite

within the testing directory there is a test suite that will check the following:
* pipeline calls
* expected file structure
* model arg options

This will use the database in the `testing/data/` directory which includes:
* rawdf:    pickled database
* schema:   dictionary of what fields are expected

### test suite call

after navication to the 'testing' directory
`python testing_suite.py`

