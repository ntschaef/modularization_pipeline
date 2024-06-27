import sklearn
class Splitter():
    '''
    Uses the indexes from the sanitized data to filter based on the predetermined criteria.
    '''
    def __init__(self, cc, split_type='case', by_registry=True, test_split=['percent',.15], val_split=['percent',.15], 
                           sklearn_random_seed=0, **kwargs):
        '''
        Initializer of the Splitter class.  Defines the "attributes".

        Parameters:
            cc (class): the CachedClass will be passed through all the modules and keep track of the pipeline arguements.
            split_type (string): at which level are the reports split?  The current options are "patient", "case", and "record".
            by_registry (boolean): will the splits be created on the registry level (instead of reating all the data as one set)?
            test_split (list): identification of how the test split will be withheld. It will take the form (how, value). The "how" is currently restricted to "percent" (a flat percent with the value, v, such that 0 < =v < 1), "year" (where the focus year will be greater than the value) and "cv" (which will divide the dataset by K sets >2 and rotate through each set).
            val_split (list): identification of how the validation split will be designated. It will take the form (how, value). The "how" is currently restricted to "percent" (a flat percent with the value, v, such that 0 <= v < 1) and "cv" (where the value will be the amount of folds).
            sklearn_random_seed (int): The seed at which sklearn will genrate the splits.
        '''
        assert cc.check_args('Splitter', locals()), 'mismatch between expected args and what is being used'
        self.pickle_files = {}
        self.text_files = {}
        cc.saved_mod_args['split_type'] = (split_type, 'Splitter', 'at which level are the reports split?  The current options are "patient" (which will look at the patient keys only), "case" (which will only look at the case keys) and "record" (which will treat every case as individual')
        cc.saved_mod_args['by_registry'] = (by_registry, 'Splitter', 'Will the splits be created on the registry level (instead of creating all the data as one set)? Eg if reg1 has 200 records and reg2 has 100, should the split force 10% to be 20 from reg1 and 10 from reg2 (vs something like 23 from reg 1 and 7 from reg2)?')
        cc.saved_mod_args['test_split'] = (test_split, 'Splitter', 'identification of how the test split will be withheld. It will take the form (how, value). The "how" is currently restricted to "percent" (a flat percent with the value, v, such that 0 <= v < 1), "recent" (THE FIRST percent with the value, v, such that 0 <= v < 1 of reports when ordered by report date and index), "year" (where the focus year will be greater than the value), and "cv" (which will divide the dataset by K sets >2 and rotate through each set).')
        cc.saved_mod_args['val_split'] =(val_split, 'Splitter', 'identification of how the test split will be withheld. It will take the form (how, value). The "how" is currently restricted to "percent" (a flat percent with the value - which cannot be called with the "test_split": cv - , v, such that 0 < =v < 1) and "cv" (which will divide the dataset by K sets >2 and rotate through each set).')
        cc.saved_mod_args['sklearn_random_seed'] = (sklearn_random_seed, 'Splitter', 'The seed at which sklearn will genrate the splits.')

    def build_dataset(self, cc):
        '''
        this is the function used to continue the build of the model within the pipeline

        Parameters:
            cc (class): the CachedClass will be passed through all the modules
                        and keep track of the pipeline arguements.

        Modules:
            check_args: will verify the arguements are appropriate for the options chosen.
            split_data: the main run which will split the data provided. Returns the number of records in each split.
            describe: creates a report of what the filter did.
        '''
        mod_name = self.__module__.split('.')[-2]
        mod_args = cc.all_mod_args()
        if not cc.test_cache(mod_name=mod_name, del_dirty=cc.remove_dirty, mod_args=mod_args):
            with cc.timing("extracting data for splitter"):
                schema, df_tokens, filtered_ids, miss_tasks, describe = self.get_cached_deps(cc)
                mt = [v for v in miss_tasks.columns if v not in df_tokens.columns]
                for c in miss_tasks.columns:
                    df_tokens[c] = miss_tasks[c]
                self.check_args(cc.logging.debug, schema, mod_args)
                ordered_ids = sorted(set([v.split(' ')[-1] for v in schema[schema['tables'][0]]['order_id_fields'] + schema[schema['tables'][1]]['order_id_fields']]))
                # join (inner) the token data and the filtered ids
                df_filtered = df_tokens.set_index(ordered_ids).join(filtered_ids.set_index(ordered_ids),how='inner').reset_index()
            with cc.timing("creating splits"):
                # use filtered data to make the splits
                counts, rids = self.split_data(cc.logging.warning, cc.logging.debug, df_filtered, schema, mod_args)
                self.describe(describe, schema, len(df_filtered), rids, counts, mod_args)
            cc.build_cache(mod_name=mod_name,
                               pick_files=self.pickle_files,
                               text_files=self.text_files,
                               mod_args=mod_args)

    def predict_single(self, data, stored, args, filters):
        '''
        no filtering is to be done with the json run

        Parameters:
            data (dict): the raw data
            stored (dict): data that is needed to predict the record
            args (dict): arguments passed through
            filters (boolean): includes additional filter fields, Default: False
        Return:
            dict: the sanitized data
        '''
        return data

    def predict_multi(self, data, stored, args, filters, logging):
        '''
        will create a dataframe based on the schema of the args

        Parameters:
            data (dict): path for the database needing to be converted
            stored (dict): data that is needed to predict the record
            args (dict): arguments passed through
            filters(bool): will the data be filtered according to the args?
            logging (function): print fuction
        Return:
            pd.DataFrame: the converted data
        '''
        return data

    def get_cached_deps(self, cc):
        '''
        This function pulls the prior cached data.

        Parameters:
            cc (class): the CachedClass will identify the previous caches.

        Return:
            data (tuple): collection of the different cached data that was recieved:
                          describe (string), schema (dict), filtered_ids (pd.DataFrame), df_tokens (pd.DataFrame)
        '''
        data = (
             cc.get_cache_file('Parser', 'db_schema'),
             cc.get_cache_file('Sanitizer', 'df_tokens'),
             cc.get_cache_file('Filter', 'filtered_ids'),
             cc.get_cache_file('Filter', 'missing_tasks'),
             cc.get_cache_file('Filter', 'describe'))
        return data

    def check_args(self, logging, schema, mod_args):
        '''
        This will verify the arguments that have been provided to make sure are expected.

        Parameters:
            logging (function): will return the print statement
            schema (dict): the schema that has the text field names
            mod_args (dict): the saved model args to identify how to filter the text as well as everything else
        '''
        # ensure that the user inputs are acceptable according to the designated inputs
        id_fields = sorted(set([v.split(' ')[-1] for v in schema[schema['tables'][0]]['id_fields']+schema[schema['tables'][0]]['id_fields']]))
        order_id_fields = sorted(set([v.split(' ')[-1] for v in schema[schema['tables'][0]]['order_id_fields']+schema[schema['tables'][1]]['order_id_fields']]))
        ## first split_type
        assert mod_args['split_type'] in ['patient','case','record'], f'"split_type" {mod_args["split_type"]} is unknown. "patient", "case" or "record" was expected.'
        if mod_args['split_type'] == 'patient':
            assert 'tumorId' in id_fields, f'tumorId is not located in the id_fields.'
        ## test by_registry
        if mod_args['by_registry']:
            assert 'registryId' in id_fields+order_id_fields, f'registryId is not located in the id_fields.'

        ## test test_split
        assert mod_args['test_split'][0] in ['cv','percent','recent','year'], f'the first entry in "test_split" {mod_args["test_split"][0]} is unknown. "cv", "percent", "recent", or "year" was expected.'

        if mod_args['test_split'][0] == 'cv':
            assert isinstance(mod_args['test_split'][1],int) and (mod_args['test_split'][1]>2), f'the second entry in "test_split" was expected to be a percent (>=0 and <1)'
        if mod_args['test_split'][0] in ['percent','recent']:
            assert (mod_args['test_split'][1] >=0) and (mod_args['test_split'][1]<1), f'the second entry in "test_split" was expected to be a percent (>=0 and <1)'
        if mod_args['test_split'][0] == 'year':
            assert isinstance(mod_args['test_split'][1],int), f'the second entry in "test_split" was expected to be an integer'
        ## test val_split
        assert mod_args['val_split'][0] in ['percent','cv'], f'the first entry in "test_split" {mod_args["val_split"][0]} is unknown. "percent" or "cv" was expected.'
        if mod_args['val_split'][0] == 'percent':
            assert mod_args['test_split'] != 'cv', 'test split cannot be set to "cv" while the val split is set to "pecentage".'
            assert (mod_args['val_split'][1] >=0) and (mod_args['val_split'][1]<1), f'the second entry in "val_split" was expected to be a percent (>=0 and <1)'
        if mod_args['val_split'][0] == 'cv':
            assert isinstance(mod_args['val_split'][1],int) and (mod_args['val_split'][1]>1), f'the second entry in "val_split" was expected to be an integer greater than 1'
        ## test to make sure the percentages combined are not greater than or equal to 1
        assert (mod_args['test_split'][0] != 'percent') or (mod_args['val_split'][0] != 'percent') or ((mod_args['test_split'][1]+mod_args['val_split'][1])<1), f'both percentages cannot sum to be 1 or higher'
        
    def cv_split(self, K, seed, data, test=None):
        '''
        the split if the test set is cv.

        Parameters:
            K (int): the number of folds that will be used for the split
            seed (int): the seed for sklearn to initialize the randomization
            data (pd.DataFrame): the data that will be split
            test (pd.DataFrame): the data that has been created for the test split. If none then it will account for both
        Return:
            list: the list of K folds. The entires will be tuples: (train, val, test)
        '''
        from sklearn.model_selection import KFold
        Kf = KFold(n_splits=K, random_state=seed, shuffle=True)
        foldidx = []
        folds = []
        for train_index, test_index in Kf.split(data):
            foldidx.append(test_index)
        testflag = test is None 
        for i in range(K):
            nottrain=list(foldidx[i])
            val = data.iloc[foldidx[i]]
            if testflag:
                if i == 0:
                    nottrain = nottrain + list(foldidx[-1])
                    test = data.iloc[foldidx[-1]]
                else:
                    nottrain = nottrain + list(foldidx[i-1])
                    test = data.iloc[foldidx[i-1]]
            train = data.iloc[[v for v in range(len(data)) if v not in nottrain]]
            folds.append((train, val, test))
        return folds

    def test_year(self, value, data, rids):
        '''
        split the data based on the year or percentage provided.
        If a year, everything below is reserved for train and validation.  Everything equal and above is test.
        If a percentage the ordered data will be split with the most recent being test, everything else train and val.

        Parameters:
            value (int): either the min year or percentage the test set will be generated by
            data (pd.DataFrame): the data that will be split
        Return:
            pd.DataFrame: the remaining data to be split into train and val
            pd.DataFrame: the test split
        '''
        # if value is less than 1 it is a percentage
        if value < 1:
            testsize = int(len(data)*value)
            return data[testsize:],data[:testsize][rids]
        else:
            # values greater than 1 are years
            d_field = [v for v in data.columns if v[-5:] == '_year'][0]
            return data[data[d_field] < value][rids], data[data[d_field] >= value][rids]

    def percent_split(self, split, seed, data, test=None):
        '''
        split the data into a test set with the percentage given. Everything else will be split later.

        Parameters:
            split (long): the decimal multiplier (%) that will be used for the test set
            seed (int): the seed for sklearn to initialize the randomization
            data (pd.DataFrame): the data that will be split
        Return:
            pd.DataFrame: the remaining data to be split into train and val
            pd.DataFrame: the test split
        '''
        from sklearn.model_selection import train_test_split
        if test is None:
            if split > 0:
                return train_test_split(data, test_size=split, random_state=seed)
            else:
                return data, data[0:0]
        else:
            if split > 0:
                new_split = ((len(data)+len(test))*split)/len(data)
                assert new_split<1, 'the requested split for the dataset exceeds the amount of records remaining after the test was removed.'
                return [train_test_split(data, test_size=new_split, random_state=seed) + [test]]
            else:
                return [[data,data[0:0],test]]

    def split_data(self, warning, logging, data, schema, mod_args):
        '''
        the main run which will split the data provided.

        Parameters:
            logging (function): the will return the print statements
            data (pd.DataFrame): the dataframe that will be used to split the records.
            schema (dict): the schema of the original database
            mod_args (dict): all the saved arguements thus far.

        Return:
            tuple: the number of records split into train, test and val for each fold.
            list: the ids that were grouped to make the splits
        '''
        # getting the ids that will be used to split the data
        rids = sorted(set([v.split(' ')[-1] for v in schema[schema['tables'][0]]['id_fields']+schema[schema['tables'][1]]['id_fields']+['registryId']]))
        if mod_args['split_type'] == 'patient':
            rids = [v for v in rids if v != 'tumorId']+['registryId']
        elif mod_args['split_type'] == 'record':
            rids = sorted(set([v.split(' ')[-1] for v in schema[schema['tables'][0]]['order_id_fields']+schema[schema['tables'][1]]['order_id_fields']+['registryId']]))

        # dividing the data so each part will be split seperately
        data_parts = []
        if mod_args['by_registry']:
            for reg in data['registryId'].unique():
                data_parts.append(data[data['registryId'] == reg])
        else:
            data_parts = [data]

        folds = []
        for i,data_part in enumerate(data_parts):       
            folds_part = None
            # split the test set
            if mod_args['test_split'][0] == 'cv':
                df_ids = data_part[rids].drop_duplcates()
                warning('test_split will override the val_split option. CV will be used with both.')
                folds_part = self.cv_split(mod_args['test_split'][1], mod_args['sklearn_random_seed'], df_ids)
            elif mod_args['test_split'][0] in ['year','recent']:
                date_field = schema[schema['tables'][0]]['key_date'][0]+'_year'
                df_ids = data_part.groupby(rids)[date_field].max().reset_index().sort_values([date_field]+rids, ascending=False)
                trainval, test = self.test_year(mod_args['test_split'][1], df_ids, rids)
            else:
                df_ids = data_part[rids].drop_duplicates()
                trainval, test = self.percent_split(mod_args['test_split'][1], mod_args['sklearn_random_seed'], df_ids)
            # split the train and val
            if folds_part == None:
                if mod_args['val_split'][0] == 'cv':
                    folds_part = self.cv_split(mod_args['val_split'][1], mod_args['sklearn_random_seed'], trainval, test)
                else:
                    folds_part = self.percent_split(mod_args['val_split'][1], mod_args['sklearn_random_seed'], trainval, test)

#            print(sum([len(fp) for fp in folds_part[1]]))
            for j,fold_part in enumerate(folds_part):
                if i == 0:
                    folds.append(fold_part)
                else:
                    for k,split_part in enumerate(fold_part):
                        folds[j][k] = folds[j][k].append(split_part, ignore_index=True)

        logging('folds: ' + str(folds))
        logging('folds count: ' + str(len(folds)))
        counts = []
        self.text_files['fold_num'] = str(len(folds))
        for i,(train, val, test) in enumerate(folds):
            logging(rids)
            logging(train)
            train_data = data.set_index(rids).join(train[rids].set_index(rids), how='inner')
            val_data = data.set_index(rids).join(val[rids].set_index(rids), how='inner')
            test_data = data.set_index(rids).join(test[rids].set_index(rids), how='inner')
            recids = [v.split(' ')[-1] for v in schema[schema['tables'][0]]['order_id_fields']]
            self.pickle_files['train_fold'+str(i)] = train_data[recids]
            self.pickle_files['val_fold'+str(i)] = val_data[recids]
            self.pickle_files['test_fold'+str(i)] = test_data[recids]
            counts.append((len(train),len(val),len(test)))
        return counts, rids

    def describe(self, describe, schema, o_num, rids, counts, mod_args):
        '''
        generates a description of what what built during the pipeline build

        Parameters:
            describe (text): description from previous modules
            schema (dict): identifies the fields that are being used in the dataframe
            o_num (int): number of records after filtering
            rids (list): the ids fields that were used to group the splits
            counts (tuple):  lists the amount of records ids in each split
            mod_args (dict): identify the arguments passed
        '''
        lines = []
        lines.append(describe)
        lines.append('\nSPLITTER:')
        lines.append(f'The remaining {o_num} records were grouped using {", ".join(rids)} resulting in {counts[0][0]+counts[0][1]+counts[0][2]} unique indexes{" split evenly by registries" if mod_args["by_registry"] else ""}.')
        if mod_args['test_split'][0] in ['year', 'recent']:
            lines.append(f'These indexes were then seperated by {schema[schema["tables"][0]]["key_date"][0]}_year with the max date of the indexes that lay above {mod_args["test_split"][1] if mod_args["test_split"][1]>=1 else str(mod_args["test_split"][1]*100)+" percent"} being withheld for the test set')
        elif mod_args['test_split'][0] == 'percent':
                lines.append(f'{mod_args["test_split"][1] * 100}% were then withheld as the test set.')
        if mod_args['val_split'][0] == 'percent':
            lines.append(f'{mod_args["val_split"][1] * 100}% was withheld as the validation set.')
        lines.append(f'These counts of the indexes accross {len(counts)} folds are as follows:')
        for i, (train, val, test) in enumerate(counts):
            lines.append(f'   fold {i}: {train} in train, {val} in validation, and {test} in test.')
        self.text_files['describe'] = '\n'.join(lines) 
