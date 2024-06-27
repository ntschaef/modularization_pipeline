class Temp_mod():
    '''
    this is a stand in script for modules.
    '''
    def __init__(self, cc, temp_arg=False, **kwargs):
        '''
        stand in initializer.

        Parameters:
            cc (class): the CachedClass will be passed through all the modules and keep track of the pipeline arguements.
            temp_arg (boolean): stand in for argument.
        '''
        # check to make sure the args match the deps.json in the parent folder. 
        assert cc.check_args('Temp_mod',locals()), 'mismatch between expected args and what is being used'
        # save all the mods to the cache class for later use and documentation
        cc.saved_mod_args['temp_arg'] = (temp_arg, 'Temp_mod', 'stand in for argument.')
        # initiate the files that will be cached
        self.pickle_files = {}
        self.text_files={}
        # unnecessary but convinient call reference to the args (both in this and previous modules)

    def build_dataset(self, cc):
        '''
        this is the function used to continue the build of the model within the pipeline

        Parameters:
            cc (class): the CachedClass will be passed through all the modules
                        and keep track of the pipeline arguements.
        Modules:
            temp_script (string): this will produce text for each fold.
        '''
        # check to see if cache is already generated. If not run the scripts.
        mod_args = cc.all_mod_args()
        mod_name = self.__module__.split('.')[-2]
        if not cc.test_cache(mod_name=mod_name, del_dirty=cc.remove_dirty, mod_args=mod_args):
            # use timing to print and show progress
            with cc.timing(f"extracting data for {mod_name}"):
                # test args to ensure they won't throw errors in the script
                self.check_args(mod_args['temp_arg'])
                # grab previous data for use in module
                folds, describe = self.get_cached_deps(cc)
                # set up variable
                temp_text = []
                for i in range(folds):
                    temp_text.append(self.temp_script(i))
            # generate descibe for this module  
            self.describe(describe, temp_text)
            # save to the cache
            cc.build_cache(mod_name=mod_name,
                               pick_files=self.pickle_files,
                               text_files=self.text_files,
                               mod_args=mod_args)

    def predict_single(self, data, stored, args, filters):
        '''
        what to do in the json run. The return is expected to be a dict.

        Parameters:
            data (dict): the raw data
            stored (dict): data that is needed to predict the record
            args (dict): arguments passed through
            filters (boolean): includes additional filter fields, Default: False
        Return:
            dict: the sanitized data
        '''
        return data

    def predict_multi(self, data, stored, args, filters):
        '''
        what to do in the dataframe run. The return is expected to be a dataframe.

        Parameters:
            data (dict): the raw data
            stored (dict): data that is needed to predict the record
            args (dict): arguments passed through
            filters (boolean): includes additional filter fields, Default: False
        Return:
            dict: the sanitized data as a pd.Dataframe
        '''
        return data

    def get_cached_deps(self, cc):
        '''
        This function pulls the prior cached data.

        Parameters:
            cc (class): the CachedClass will identify the previous caches.

        Return:
            data (tuple): collection of the different cached data that was recieved:
                          fold (int), describe (string)
        '''
        data = (
             int(cc.get_cache_file('Splitter', 'fold_num')),
             cc.get_cache_file('Encoder', 'describe')
        )

        return data

    def check_args(self, temp_arg):
        '''
        This will verify the arguments that have been provided to make sure are expected.

        Parameters:
            logging (function): will return the print statement
            temp_arg (dict): the schema that has the text field names
            mod_args (dict): the saved model args to identify how to filter the text as well as everything else
        '''
        # ensure that the user inputs are acceptable according to the designated inputs
        assert isinstance(temp_arg, bool), "the temp_arg is not a boolean value"

    def temp_script(self, fold):
        '''
        a test script that will produce a text string for each fold.

        Parameters:
            fold (int): the number of folds produced
        Return:
            string: fold produced as a string.
        '''
        
        self.text_files[f'fold_{fold}_text'] = str(fold)
        return str(fold)

    def describe(self, describe, temp_text):

        '''
        generates a description of what what built during the pipeline build

        Parameters:
            describe (text): description from previous modules
            temp_text (list): number of folds create in the module
        '''
        lines = []
        lines.append(describe)
        lines.append('\nTemp_mod:')
        lines.append(f'This was a temp mod and the description was not changed. It was supposed to create text stored enumerating each of the {len(temp_text)} folds')
        self.text_files['describe'] = '\n'.join(lines) 
