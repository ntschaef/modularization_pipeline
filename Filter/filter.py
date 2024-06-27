from datetime import datetime
import pandas as pd
import numpy as np
import itertools

class Filter():
    '''
    Uses the indexes from the sanitized data to filter based on the predetermined criteria.
    '''
    def __init__(self, cc, tasks=['behavior','histology','laterality','site','subsite', 'recode'], only_single='none', include_only_scores={}, window_days=[10], window_fields=['dateOfDiagnosis', 'rxDateMostDefinSurg', 'rxDateSurgery'], min_year=2004, max_year=0, **kwargs):
        '''
        Initializer of the Filter class.  Defines the "attributes".

        Parameters:
            cc (class): the CachedClass will be passed through all the modules and keep track of the pipeline arguements.
            tasks (list): this will be the list of single or combined tasks that will be predicted.
            only_single (string): Whether to accept only records that are unique to the patient, case, or none. Respective options: "patient", "case", "none".
            include_only_scores (dict): restrict to specific scores in each field.
            window_days (list): the beginning and end date of the date filter. Single positive digit in list is +/-. [0] removes the filter.
            window_fields (list): the date fields included in the filter.
            min_year (int): the minimum year that is allowed for the reports.
            max_year (int): the maximum year that is allowed for the reports. If "0" or greater than the present year, then the max year defaults to the date.
        '''
        assert cc.check_args('Filter', locals()), 'mismatch between expected args and what is being used'
        self.pickle_files = {}
        self.text_files = {}
        cc.saved_mod_args['tasks'] = (tasks, 'Filter', 'The list of tasks for which to generate labels. If desired, multiple tasks can be combined into a single task using the syntax "task1+task2".  For example, if task1 has labels (1,2) and task2 has labels (a,b), "task1+task2" will result in the label set (1a,1b,2a,2b).\n"behavior","grade","histology","laterality","site","subsite","biomarkers_er","biomarkers_pr","biomarkers_her2","biomarkers_kras","biomarkers_msi","recode","ICCC": availible for all schemas.\n"reportability","report_category": availible for ONLY the man_coded schema.\n"recurrence": availible in the recurrence schema.')
        cc.saved_mod_args['only_single'] = (only_single, 'Filter', 'Each case or patient can be associated with multiple path reports. If set to "case" or "patient", the resulting dataset will only include path reports from cases/patients associated with a single report. Set to "None" will include reports from all cases/patients.')
        cc.saved_mod_args['include_only_scores'] = (include_only_scores, 'Filter', 'restrict to specific scores in each field. Example usage: {"csMetsAtDx":["<1","2","3",">=5"]} will only filter out "1" and "4" from being used in the metastatic field. If there is a numeric values a range filtered can be used by declaring ">", "<", ">=", or "<=" BEFORE the number. Default: empty dictionary includes all values.')
        cc.saved_mod_args['window_days'] = (window_days, 'Filter', 'Only keep reports where at least one of the specified window_fields is within X days of "pathDateSpecCollect1". Examples: "[10]" will keep reports within 10 days before or after SpecCollect, "[-5,10]" will keep reports within 5 days before or 10 days after SpecCollect, and "[0]" will ignore this filter and keep all reports')
        cc.saved_mod_args['window_fields'] = (window_fields, 'Filter', f'The date fields used with the "window_days" filter.  Default is {str(window_fields)}')
        cc.saved_mod_args['min_year'] = (min_year, 'Filter', 'The minimum year that is allowed for the reports.')
        cc.saved_mod_args['max_year'] = (max_year, 'Filter', 'The maximum year that is allowed for the reports. If "0" or greater than the present year, then the max year defaults to the date.')

    def build_dataset(self, cc):
        '''
        this is the function used to continue the build of the model within the pipeline

        Parameters:
            cc (class): the CachedClass will be passed through all the modules
                        and keep track of the pipeline arguements.
        Modules:
            check_args: will verify the arguements are appropriate for the options chosen, will return the key date if no errors are found.
            filter_data: the main run which will filter the data provided. Returns the number of records removed from each filter.
            describe: creates a report of what the filter did.
        '''
        mod_name = self.__module__.split('.')[-2]
        mod_args = cc.all_mod_args()
        if not cc.test_cache(mod_name=mod_name, del_dirty=cc.remove_dirty,mod_args=mod_args):
            with cc.timing("extracting data for filter"):
                s_name, schema, dftokens, describe = self.get_cached_deps(cc)
                miss_tasks = [v for v in mod_args['tasks'] if v not in dftokens.columns]
                key_date = self.check_args(cc.logging.debug, s_name, schema, mod_args, miss_tasks)
                dftokens = self.biomarkers_creation(dftokens, miss_tasks)
                dftokens = self.recode_creation(dftokens, miss_tasks)
                dftokens = self.ICCC_creation(dftokens, miss_tasks)
            with cc.timing("filtering out unwanted data"):
                counts,dftokens = self.filter_data(cc.logging.debug, dftokens, schema, mod_args, key_date, miss_tasks)
                self.pickle_files['missing_tasks'] = dftokens[miss_tasks]
                self.describe(describe, counts, mod_args, key_date)
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
        import json
        if filters:
            key_date = stored['schema'][stored['schema']['tables'][0]]['key_date'][0]
            args['window_days'] = args['window_days']
            args['window_fields'] = args['window_fields']
            args['include_only_scores'] = args['include_only_scores']
            args['tasks'] = args['tasks']
            self.filter_data(logging, data, stored['schema'], args, key_date)
            cols = list(self.pickle_files['filtered_ids'].columns)
            data = data.set_index(cols).join(self.pickle_files['filtered_ids'].set_index(cols)).reset_index()
        return data


    def get_cached_deps(self,cc):
        '''
        This function pulls the prior cached data.

        Parameters:
            cc (class): the CachedClass will identify the previous caches.
        Return:
            data (tuple): collection of the different cached data that was recieved:
                          s_name (string), schema (dict), dftokens (pd.DataFrame), describe (string)
        '''
        data = (
             cc.get_cache_file('Parser', 'schema_name'),
             cc.get_cache_file('Parser', 'db_schema'),
             cc.get_cache_file('Sanitizer', 'df_tokens'),
             cc.get_cache_file('Sanitizer', 'describe'))
        return data

    def check_args(self, logging, s_name, schema, mod_args, missing_tasks):
        '''
        This will test the schema against the args fields.
        
        Parameters:
            logging (fuction): will return the print statements
            s_name (string): name of the schema being used
            schema (dict): the schema that has the text field names
            mod_args (dict): the saved model args to identify how to filter the text as well as everything else

        Return:
            string: key date field for the schema
        '''
        # Create a list of all the fields
        field_list = []
        for tbl in schema['tables']:
            for l in schema[tbl].values():
                logging(f'tabel list is: {l}')
                field_list += [v.split(' ')[-1] for v in l]
        filed_list = sorted(set(field_list))

        # test min date is smaller than max date
        logging(f'min_date: {mod_args["min_year"]}')
        logging(f'max_date: {mod_args["max_year"]}')
        if mod_args['max_year']>0:
            assert mod_args['min_year'] < mod_args['max_year'], f'the min_date given is not smaller than the max_date given'
        
        # test date window choices
        logging(f'schema: {schema}')
        if mod_args['window_days']!=[0]:
            mismatch_list = [f for f in mod_args['window_fields'] if f not in schema[schema['tables'][1]]['date_fields']]
            assert len(mismatch_list) == 0, f'the following fields are not dates in the {s_name} schema: {", ".join(mismatch_list)}. Remove them from window_fields or turn off the filter with window_days=[0]'
        # test included fields
        mismatch_list = [f for f in mod_args['include_only_scores'].keys() if f not in field_list]
        assert len(mismatch_list) == 0, f'the following fields are not in the {s_name} schema: {", ".join(mismatch_list)}. Remove them from include_only_scores.'

        # test tasks
        task_list = sorted(set([f1 for f in mod_args['tasks'] for f1  in f.split('+')]))

        add_fields = [f for fs in mod_args['add_fields'].values() for f in fs] 
        mismatch_list = [v for v in task_list if v not in [v.split(' ')[-1] for v in schema[schema['tables'][1]]['baseline_fields']+add_fields]]
        mismatch_list = [v for v in mismatch_list if v not in missing_tasks]
        assert len(mismatch_list) == 0, f'the following fields are not tasks in the {s_name} schema: {", ".join(mismatch_list)}'

        # get the key date field for the schema.  It comes up a few times
        kdl = schema[schema['tables'][0]]['key_date']
        assert len(kdl)==1, 'the key_date for the schema should only have one value'
        
        return kdl[0]

    def filter_by_single(self, logging, data, arg, schema):
        '''
        module filters out due to the single category

        Parameters:
            logging (fuction): will return the print statements
            data (pd.DataFrame): records being processed
            arg (string): the model arg used in this filter
            schema (dict): the schema that has the text field names

        Return:
            pd.DataFrame: filtered data
            int:          the number of records filtered out
        '''
        n = len(data)
        ids = data.columns 
        if arg == 'patient':
            logging(f'data fields: {", ".join(data.columns)}')
            ids = [v for v in ['patientId', 'registryId'] if v not in ids]
            assert len(ids) == 0, f'the following ids cannot be found: {", ".join(ids)}'
            dup_patients = data[data[['patientId','registryId']].duplicated()][['patientId','registryId']]
            logging(f'these are the duplicate ids: {dup_patients}')
            data = data[~(data['patientId']+data['registryId']).isin(dup_patients['patientId']+dup_patients['registryId'])]
        elif arg == 'case':
            ids = [v for v in ['patientId', 'registryId','tumorId'] if v not in ids]
            assert len(ids) == 0, f'the following ids cannot be found: {", ".join(ids)}'
            dup_patients = data[data[['patientId','registryId','tumorId']].duplicated()][['patientId','registryId','tumorId']]
            data = data[~(data['patientId']+data['tumorId']+data['registryId']).isin(dup_patients['patientId']+dup_patients['tumorId']+dup_patients['registryId'])]
        else:
            assert arg == 'none', f'the "only_single" value was expected to be "patient", "case" or "none"; not "{arg}"' 
        logging(f'filterd single data: {data}') 
        return data, n-len(data)

    def filter_by_window(self, logging, data, arg):
        '''
        module filters out based on the date window

        Parameters:
            logging (fuction): will return the print statements
            data (pd.DataFrame): records being processed
            arg (tuple): the model arg used in this filter

        Return:
            pd.DataFrame: filtered data
            int:          the number of records filtered out
        '''
        n = len(data)
        # make an easy parsing of the number of days
        assert isinstance(arg[0],list), f'the "window_days" is expected to be a list'
        logging(f'window argument: {arg[0]}, {arg[1]}')
        assert len(arg[0]) in [1,2], f'the "window_days" is expected to be a list of form [#] or [#1, #2] not {arg[0]}'
        if len(arg[0])==1:
           assert arg[0][0]>=0, f'the "window_days" (if one value) is expected to be >= 0, not {arg[0][0]}'
           if arg[0][0] > 0:
               days = [-1*arg[0][0],arg[0][0]]
           else:
               # this filter is skipped if window_days is [0]
               return data, 0
        else:
            days = arg[0]
        # now that window days is of the form [min,max] loop through the fields provided
        datefields = [v for v in data.columns if v[-5:] == "_diff" for d_field in arg[1] if d_field in v]
        kept_fields = data[[]].copy(deep=True)
        kept_fields['keep'] = False
        for f in datefields:
            kept_fields['temp'] = True
            kept_fields['temp'] = kept_fields['temp'] & (data[f] >= days[0])
            kept_fields['temp'] = kept_fields['temp'] & (data[f] <= days[1])
            kept_fields['keep'] = kept_fields['keep'] | kept_fields['temp']
        data = data[kept_fields['keep']]
        logging(f'filterd custom window data: {data}') 
        return data, n-len(data)

    def filter_by_field(self, logging, data, arg):
        '''
        module filters out based on a custom fields restrictions

        Parameters:
            logging (fuction): will return the print statements
            data (pd.DataFrame): records being processed
            arg (dict): the model arg used in this filter

        Return:
            pd.DataFrame: filtered data
            int:          the number of records filtered out
        '''
        if arg == {}:
            return data, 0
        n = len(data)
        kept_fields = data[[]].copy(deep=True)
        # loop through fields and split the filter into distinct values and ranges
        kept_fields['keep'] = True
        for f,l in arg.items():
            kept_fields['temp'] = True
            r = [v for v in l if v[:1] in ['<','>']]
            dv = [v for v in l if v[:1] not in ['<','>']]
            #loop through ranges
            for text in r:
                if text[:2] == '>=':
                    kept_fields['temp'] = kept_fields['temp'] & (data[f].apply(lambda v: int(v) >= int(text[2:]) if v != '' else False))
                elif text[:2] == '<=':
                    kept_fields['temp'] = kept_fields['temp'] & (data[f].apply(lambda v: int(v) <= int(text[2:]) if v != '' else False))
                elif text[:1] == '>':
                    kept_fields['temp'] = kept_fields['temp'] & (data[f].apply(lambda v: int(v) > int(text[1:]) if v != '' else False))
                elif text[:1] == '<':
                    kept_fields['temp'] = kept_fields['temp'] & (data[f].apply(lambda v: int(v) < int(text[1:]) if v != '' else False)) 
            # check to see if you keep the entries
            if dv != []:
                if r == []:
                    kept_fields['temp'] = False
                kept_fields['temp'] = kept_fields['temp'] | (data[f].map(str).isin(dv))
            kept_fields['keep'] = kept_fields['keep'] & kept_fields['temp']
        data = data[kept_fields.keep]
        logging(f'filterd custom field data: {data}') 
        return data, n-len(data)

    def filter_gold(self, logging, data, arg):
        '''
        module filters out based on the what is needed for the gold values

        Parameters:
            logging (fuction): will return the print statements
            data (pd.DataFrame): records being processed
            arg (list): the model arg used in this filter

        Return:
            pd.DataFrame: filtered data
            int:          the number of records filtered out
        '''
        n = len(data)
        #parse the argument to make sure that combined fields are seperated
        golds = sorted(set([v for v1 in arg for v in v1.split('+')]))
        kept_fields = data[[]].copy(deep=True) 
        kept_fields['keep'] = True
        # loop through tasks and remove the null values
        for t in golds:
            kept_fields['keep'] = kept_fields['keep'] & (data[t]!="") & (~data[t].isna())
        data = data[kept_fields.keep]
        assert len(data) > 0, f'there are no records after filtering out the gold standard'
        logging(f'filterd gold data: {data}') 
        return data, n-len(data)

    def filter_by_date_range(self, logging, data, min_year, max_year, key_date):
        '''
        module filters out the records that occur before the given year or after current year

        Parameters:
            logging (fuction): will return the print statements
            data (pd.DataFrame): records being processed
            min_year (int): the minimum year that will be considered acceptable
            max_year (int): the maximum year that will be considered acceptable. 0 will use the current date.
            key_date (string): the key date that that filter will focus on

        Return:
            pd.DataFrame: filtered data
            int:          the number of records filtered out
        '''
        start_count = len(data)
        data = data[data[key_date+'_year'] >= int(min_year)]
        mid_count = len(data)
        max_year = int(max_year)
        if max_year==0:
            max_year=datetime.now().year
        data = data[data[key_date+'_year'] <= max_year]
        logging(f'filterd year range data: {data}') 
        return data, start_count-mid_count, mid_count-len(data)

    def filter_data(self, logging, data, schema, mod_args, key_date, miss_tasks):
        '''
        Main function used to filter the data

        Parameters:
            logging (fuction): will return the print statements
            data (pd.DataFrame): records being processed
            schema (dict): the schema that has the text field names
            mod_args (dict): the saved model args to identify how to filter the text as well as everything else
            key_date (string): the record date that will be used as the primary
            miss_tasks (list): missing tasks list 

        Modules:
            filter_gold: removes records with unexpected gold entries
            filter_by_single: refine records to only those that are unique in thier category (case or patient)
            filter_by_field: keeps only records with only wanted scores
            filter_by_date_range: removes all records before selected date and after the current year
            filter_by_window: removes all records outside of a chosen time window

        Return:
            r_single (int): number of records filtered out due to the single category
            r_window (int): number of records filtered out due to the date window
            r_fields (int): number of records filtered out due to the custom field refinement
            r_gold (int): number of records with incorrect gold scores
            r_early (int): number of records that are too old
            r_old (int): number of records that are too new
        '''
        
        logging(f"prefiltered data: {data}")
        data, r_single = self.filter_by_single(logging, data, mod_args['only_single'],schema)
        data, r_window = self.filter_by_window(logging, data, (mod_args['window_days'], mod_args['window_fields']))
        data, r_field = self.filter_by_field(logging, data, mod_args['include_only_scores'])
        data, r_gold = self.filter_gold(logging, data, mod_args['tasks'])
        data, r_early, r_old = self.filter_by_date_range(logging, data, mod_args['min_year'], mod_args['max_year'], key_date)
        logging(f"single drops: {r_single}, window drops: {r_window}, custom field drops: {r_field}, null gold drops: {r_gold}, old record drops: {r_early}")
        f_id_fields = sorted(set([v.split(' ')[-1] for v in 
                                 schema[schema['tables'][0]]['order_id_fields']+
                                 schema[schema['tables'][1]]['order_id_fields']]))
        filtered_ids = data[f_id_fields]

        logging(f'filtered_ids: {filtered_ids}')
        logging(f'filtered_ids_columns: {filtered_ids.columns}')
        self.pickle_files['filtered_ids'] = filtered_ids
        return (r_single, r_window, r_field, r_gold, r_early, r_old), data

    def describe(self, describe, counts, mod_args, key_date):
        '''
        generates a description of what what built during the pipeline build

        Parameters:
            deps (dict):     the dependencies expected for the naming of the package information
            describe (text): description from previous modules
            counts (tuple):  lists the amount of records that were removed with each filter
            mod_args (dict): identify the arguements passed through.
        '''
        lines = []
        lines.append(describe)
        lines.append('\nFILTER:')
        lines.append('The following filters were applied in the following order:')
        if mod_args['only_single'] != 'none':
            lines.append(f'  {counts[0]} had duplicate {mod_args["only_single"]}')
        if mod_args['window_days'][0]>0:
            days = mod_args['window_days']
            if len(days) == 1:
                days = [-1*days[0],days[0]]
            lines.append(f'  {counts[1]} had none of the following dates surrounding "{key_date}" in the range from {days[0]} to {days[1]}: {", ".join(mod_args["window_fields"])}')
        if len(mod_args['include_only_scores'].keys()) > 0:
            lines.append(f'  {counts[2]} had scores that did not fall into the following criteria:')
            for k,v in mod_args['include_only_scores'].items():
                lines.append(f'     the field "{k}" with scores {", ".join(v)}')
        q="\"" # backslashes throw errors in python "f-strings"
        lines.append(f'  {counts[3]} had null entries in the required tasks "{(q+","+q).join(mod_args["tasks"])}"')
        lines.append(f'  {counts[4]} had "{key_date}" that fell below the year {mod_args["min_year"]}')
        if mod_args['max_year']==0:
            lines.append(f'  {counts[5]} had "{key_date}" that is in the future')
        else:
            lines.append(f'  {counts[5]} had "{key_date}" that is above the year {mod_args["max_year"]}')
        self.text_files['describe'] = '\n'.join(lines)

    def biomarkers_creation(self, dftokens, miss_tasks):
        
        '''
        Function to filter columns for the biomarkers task.
            
        Parameters:
           dftokens (pd.DataFrame): records being processed
           miss_tasks (list): new tasks to add in original data
        '''

        cols_till_2017 = {'biomarkers_er': 'csSiteSpecificFactor1', 
                          'biomarkers_pr': 'csSiteSpecificFactor2', 
                          'biomarkers_her2': 'csSiteSpecificFactor15', 
                          'biomarkers_kras': 'csSiteSpecificFactor9',
                          'biomarkers_msi': ''}

        cols_after_2017 = {'biomarkers_er': 'estrogenReceptorSummary', 
                           'biomarkers_pr': 'progesteroneRecepSummary', 
                           'biomarkers_her2': 'her2OverallSummary', 
                           'biomarkers_kras': 'kras',
                           'biomarkers_msi': 'microsatelliteInstability'}

        for miss_task in miss_tasks:
            if 'biomarkers' in miss_task:
                col_till_2017 = cols_till_2017[miss_task]
                col_after_2017 = cols_after_2017[miss_task]
                dftokens[miss_task] = ''

            if miss_task in ["biomarkers_er", "biomarkers_pr", "biomarkers_her2"]:
            
                if miss_task != 'biomarkers_her2':
                    # For dataset collected in or before 2017
                    dftokens.loc[(dftokens.site == 'C50') & (dftokens.pathDateSpecCollect1_year >= 2004) & (dftokens.pathDateSpecCollect1_year <= 2017) & (dftokens[col_till_2017].isin(['010', '030'])), miss_task] = '1'
                    dftokens.loc[(dftokens.site == 'C50') & (dftokens.pathDateSpecCollect1_year >= 2004) & (dftokens.pathDateSpecCollect1_year <= 2017) & (dftokens[col_till_2017]== '020'), miss_task] = '0'
                    SSF1_filter_list = ['997', '988', '998', '999']
                    dftokens.loc[(dftokens.site == 'C50') & (dftokens.pathDateSpecCollect1_year >= 2004) & (dftokens.pathDateSpecCollect1_year <= 2017) & (dftokens[col_till_2017].isin(SSF1_filter_list)),miss_task] = '9'
                    # For dataset collected after 2017
                    dftokens.loc[(dftokens.site == 'C50') & (dftokens.pathDateSpecCollect1_year > 2017) & (dftokens[col_after_2017] == '0'), miss_task] = '0'
                    dftokens.loc[(dftokens.site == 'C50') & (dftokens.pathDateSpecCollect1_year > 2017) & (dftokens[col_after_2017] == '1'), miss_task] = '1'
                    dftokens.loc[(dftokens.site == 'C50') & (dftokens.pathDateSpecCollect1_year > 2017) & (dftokens[col_after_2017].isin(['7','9'])), miss_task] = '9'
                else:
                    # For dataset collected in or before 2017
                    dftokens.loc[(dftokens.site == 'C50') & (dftokens.behavior == '3') & (dftokens.pathDateSpecCollect1_year >= 2010) & (dftokens.pathDateSpecCollect1_year <= 2017) & (dftokens[col_till_2017].isin(['020','030'])), miss_task] = '0'
                    SSF1_filter_list = ['988', '998', '999', '997']
                    dftokens.loc[(dftokens.site == 'C50') & (dftokens.behavior == '3') & (dftokens.pathDateSpecCollect1_year >= 2010) & (dftokens.pathDateSpecCollect1_year <= 2017) & (dftokens[col_till_2017].isin(SSF1_filter_list)),miss_task] = '9'
                    dftokens.loc[(dftokens.site == 'C50') & (dftokens.behavior == '3') & (dftokens.pathDateSpecCollect1_year >= 2010) & (dftokens.pathDateSpecCollect1_year <= 2017) & (dftokens[col_till_2017] =="010"), miss_task] = '1'
                    # For dataset collected after 2017
                    dftokens.loc[(dftokens.site == 'C50') & (dftokens.behavior == '3') & (dftokens.pathDateSpecCollect1_year > 2017) & (dftokens[col_after_2017] == '0'), miss_task] = '0'
                    dftokens.loc[(dftokens.site == 'C50') & (dftokens.behavior == '3') & (dftokens.pathDateSpecCollect1_year > 2017) & (dftokens[col_after_2017] == '1'), miss_task] = '1'
                    dftokens.loc[(dftokens.site == 'C50') & (dftokens.behavior == '3') & (dftokens.pathDateSpecCollect1_year > 2017) & (dftokens[col_after_2017].isin(['7','9'])), miss_task] = '9'

            if miss_task == "biomarkers_kras":

                site_list = ['C18', 'C19', 'C20']
                SSF9_filter_list = ['988', '997', '998', '999']
                # For dataset collected in or before 2017
                dftokens.loc[(dftokens.site.isin(site_list)) & (dftokens.pathDateSpecCollect1_year >= 2010) & (dftokens.pathDateSpecCollect1_year <= 2017) & (dftokens[col_till_2017] == '020'),miss_task] = '0'
                dftokens.loc[(dftokens.site.isin(site_list)) & (dftokens.pathDateSpecCollect1_year >= 2010) & (dftokens.pathDateSpecCollect1_year <= 2017) & (dftokens[col_till_2017] == '010'),miss_task] = '1'
                dftokens.loc[(dftokens.site.isin(site_list)) & (dftokens.pathDateSpecCollect1_year >= 2010) & (dftokens.pathDateSpecCollect1_year <= 2017) & (dftokens[col_till_2017].isin(SSF9_filter_list)),miss_task] = '9'

                # For dataset collected after 2017
                dftokens.loc[(dftokens.site.isin(site_list)) & (dftokens.pathDateSpecCollect1_year > 2017) & (dftokens[col_after_2017] == '0'),miss_task] = '0'
                dftokens.loc[(dftokens.site.isin(site_list)) & (dftokens.pathDateSpecCollect1_year > 2017) & (dftokens[col_after_2017].isin(['1', '2', '3', '4'])),miss_task] = '1'
                dftokens.loc[(dftokens.site.isin(site_list)) & (dftokens.pathDateSpecCollect1_year > 2017) & (dftokens[col_after_2017].isin(['7', '8', '9'])),miss_task] = '9'

            if miss_task == "biomarkers_msi":

                site_list = ['C18', 'C19', 'C20']

                dftokens.loc[(dftokens.site.isin(site_list)) & (dftokens.pathDateSpecCollect1_year > 2017) & (dftokens[col_after_2017].isin(['0','1'])),miss_task] = '0'
                dftokens.loc[(dftokens.site.isin(site_list)) & (dftokens.pathDateSpecCollect1_year > 2017) & (dftokens[col_after_2017] == '2'),miss_task] = '1'
                dftokens.loc[(dftokens.site.isin(site_list)) & (dftokens.pathDateSpecCollect1_year > 2017) & (dftokens[col_after_2017].isin(['8','9'])),miss_task] = '9'

        return dftokens

    def recode_creation(self, dftokens, miss_tasks):

        '''
        Function to filter columns for the biomarkers task.
            
        Parameters:
           dftokens (pd.DataFrame): records being processed
           miss_tasks (list): new tasks to add in original data
        '''

        if "recode" in miss_tasks:
            miss_task = "recode"
            dftokens[miss_task] = ''
            histology_list = list(dftokens.histology.unique())
            subsite_list = list(dftokens.subsite.unique())
    
            #Oral
            dftokens.loc[dftokens.subsite.isin(["C00"+str(i) for i in range(0,10)]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '20010'        
            dftokens.loc[dftokens.subsite.isin(["C0"+str(i) for i in range(19,29)]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '20020'
            dftokens.loc[dftokens.subsite.isin(["C0"+str(i) for i in range(79,90)]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '20030'
            dftokens.loc[dftokens.subsite.isin(["C0"+str(i) for i in range(40,50)]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '20040'
            dftokens.loc[dftokens.subsite.isin(["C0"+str(i) for i in range(30,40)]+["C0"+str(i) for i in range(50,60)]+["C0"+str(i) for i in range(60,70)]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '20050'
            dftokens.loc[dftokens.subsite.isin(["C0"+str(i) for i in range(110,120)]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '20060'
            dftokens.loc[dftokens.subsite.isin(["C0"+str(i) for i in range(90,100)]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '20070'
            dftokens.loc[dftokens.subsite.isin(["C"+str(i) for i in range(100,110)]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '20080'
            dftokens.loc[dftokens.subsite.isin(["C129"]+["C"+str(i) for i in range(130,140)]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '20090'
            dftokens.loc[dftokens.subsite.isin(["C140","C142","C148"]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '20100'
    
            #Digestive
            dftokens.loc[dftokens.subsite.isin(["C"+str(i) for i in range(150,160)]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '21010'
            dftokens.loc[dftokens.subsite.isin(["C"+str(i) for i in range(160,170)]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '21020'
            dftokens.loc[dftokens.subsite.isin(["C"+str(i) for i in range(170,180)]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '21030'
            dftokens.loc[dftokens.subsite.isin(["C180"]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '21041'
            dftokens.loc[dftokens.subsite.isin(["C181"]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '21042'
            dftokens.loc[dftokens.subsite.isin(["C182"]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '21043'
            dftokens.loc[dftokens.subsite.isin(["C183"]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '21044'
            dftokens.loc[dftokens.subsite.isin(["C184"]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '21045'
            dftokens.loc[dftokens.subsite.isin(["C185"]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '21046'
            dftokens.loc[dftokens.subsite.isin(["C186"]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '21047'
            dftokens.loc[dftokens.subsite.isin(["C187"]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '21048'
            dftokens.loc[dftokens.subsite.isin(["C"+str(i) for i in range(188,190)]+["C260"]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '21049'
            dftokens.loc[dftokens.subsite.isin(["C199"]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '21051'
            dftokens.loc[dftokens.subsite.isin(["C209"]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '21052'
            dftokens.loc[dftokens.subsite.isin(["C"+str(i) for i in range(210,213)]+["C218"]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '21060'
            dftokens.loc[dftokens.subsite.isin(["C220"]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '21071'
            dftokens.loc[dftokens.subsite.isin(["C221"]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '21072'
            dftokens.loc[dftokens.subsite.isin(["C239"]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '21080'
            dftokens.loc[dftokens.subsite.isin(["C"+str(i) for i in range(240,250)]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '21090'
            dftokens.loc[dftokens.subsite.isin(["C"+str(i) for i in range(250,260)]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '21100'
            dftokens.loc[dftokens.subsite.isin(["C480"]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '21110'
            dftokens.loc[dftokens.subsite.isin(["C"+str(i) for i in range(481,483)]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '21120'
            dftokens.loc[dftokens.subsite.isin(["C268","C269","C488"]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '21130'
    
            #Respiratory
            dftokens.loc[dftokens.subsite.isin(["C300","C301"]+["C"+str(i) for i in range(310,320)]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '22010'
            dftokens.loc[dftokens.subsite.isin(["C"+str(i) for i in range(320,330)]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '22020'
            dftokens.loc[dftokens.subsite.isin(["C"+str(i) for i in range(340,350)]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '22030'
            dftokens.loc[dftokens.subsite.isin(["C384"]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '22050'
            dftokens.loc[dftokens.subsite.isin(["C"+str(i) for i in range(381,383)]+["C339","C388","C390","C398","C399"]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '22060'
    
            #Bones
            dftokens.loc[dftokens.subsite.isin(["C"+str(i) for i in range(400,420)]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '23000'
    
            #Soft tissue
            dftokens.loc[dftokens.subsite.isin(["C380"]+["C"+str(i) for i in range(470,480)]+["C"+str(i) for i in range(490,500)]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '24000'
    
            #Skin
            dftokens.loc[dftokens.subsite.isin(["C"+str(i) for i in range(440,450)]) & dftokens.histology.isin([v for v in histology_list if v not in list(range(8720,8791))]),miss_task] = '25010'
            dftokens.loc[dftokens.subsite.isin(["C"+str(i) for i in range(440,450)]) & dftokens.histology.isin([v for v in histology_list if v not in list(range(8000,8006))+list(range(8010,8047))+list(range(8050,8085))+list(range(8090,8111))+list(range(8720,8791))+list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '25050'
    
            #Breast
            dftokens.loc[dftokens.subsite.isin(["C"+str(i) for i in range(500,510)]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '26000'
    
            #Female Genital
            dftokens.loc[dftokens.subsite.isin(["C"+str(i) for i in range(530,540)]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '27010'
            dftokens.loc[dftokens.subsite.isin(["C"+str(i) for i in range(540,550)]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '27020'
            dftokens.loc[dftokens.subsite.isin(["C559"]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '27030'
            dftokens.loc[dftokens.subsite.isin(["C569"]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '27040'
            dftokens.loc[dftokens.subsite.isin(["C529"]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '27050'
            dftokens.loc[dftokens.subsite.isin(["C"+str(i) for i in range(510,520)]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '27060'
            dftokens.loc[dftokens.subsite.isin(["C"+str(i) for i in range(570,580)]+["C589"]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '27070'
    
            #Male genital
            dftokens.loc[dftokens.subsite.isin(["C619"]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '28010'
            dftokens.loc[dftokens.subsite.isin(["C"+str(i) for i in range(620,630)]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '28020'
            dftokens.loc[dftokens.subsite.isin(["C"+str(i) for i in range(600,610)]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '28030'
            dftokens.loc[dftokens.subsite.isin(["C"+str(i) for i in range(630,640)]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '28040'
    
            #Urinary
            dftokens.loc[dftokens.subsite.isin(["C"+str(i) for i in range(670,680)]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '29010'
            dftokens.loc[dftokens.subsite.isin(["C649", "C659"]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '29020'
            dftokens.loc[dftokens.subsite.isin(["C669"]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '29030'
            dftokens.loc[dftokens.subsite.isin(["C"+str(i) for i in range(680,690)]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '29040'
    
            #Eye
            dftokens.loc[dftokens.subsite.isin(["C"+str(i) for i in range(690,700)]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '30000'
    
            #Brain
            dftokens.loc[dftokens.subsite.isin(["C"+str(i) for i in range(710,720)]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(range(9530,9540))+list(map(str, range(9590,9993)))]),miss_task] = '31010'
            dftokens.loc[dftokens.subsite.isin(["C"+str(i) for i in range(710,720)]) & dftokens.histology.isin([v for v in histology_list if v not in list(range(9530,9540))]),miss_task] = '31040'
            dftokens.loc[dftokens.subsite.isin(["C"+str(i) for i in range(700,710)]+["C"+str(i) for i in range(720,730)]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '31040'
    
            #Endocrine
            dftokens.loc[dftokens.subsite.isin(["C379"]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '32010'
            dftokens.loc[dftokens.subsite.isin(["C379"]+["C"+str(i) for i in range(740,750)]+["C"+str(i) for i in range(750,760)]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '32020'
    
            #Lymphoma
            dftokens.loc[dftokens.subsite.isin(["C024", "C111", "C142", "C379", "C422", "C098", "C099"]+["C"+str(i) for i in range(770,780)]) & dftokens.histology.isin([str(i) for i in range(9650,9668)]),miss_task] = '33011'
            dftokens.loc[dftokens.subsite.isin([v for v in subsite_list if v not in ["C024", "C111", "C142", "C379", "C422", "C098", "C099"]+["C"+str(i) for i in range(770,780)]]) & dftokens.histology.isin([str(i) for i in range(9650,9668)]),miss_task] = '33012'
            dftokens.loc[dftokens.subsite.isin(["C024", "C111", "C142", "C379", "C422", "C098", "C099"]+["C"+str(i) for i in range(770,780)]) & dftokens.histology.isin([str(i) for i in range(9590,9598)]+["9670","9671", "9673", "9675", "9684", "9695", "9705", "9708", "9709", "9712", "9735", "9737", "9738", "9823", "9827", "9837"]+[str(i) for i in range(9678, 9681)]+[str(i) for i in range(9687, 9692)]+[str(i) for i in range(9698, 9703)]+[str(i) for i in range(9714, 9720)]+[str(i) for i in range(9724, 9729)]+[str(i) for i in range(9811, 9819)]),miss_task] = '33041'
            dftokens.loc[dftokens.subsite.isin([v for v in subsite_list if v not in ["C024", "C111", "C142", "C379", "C422", "C098", "C099"]+["C"+str(i) for i in range(770,780)]]) & dftokens.histology.isin([str(i) for i in range(9590,9598)]+["9670","9671", "9673", "9675", "9684", "9695", "9705", "9708", "9709", "9712", "9735", "9737", "9738"]+[str(i) for i in range(9678, 9681)]+[str(i) for i in range(9687, 9692)]+[str(i) for i in range(9698, 9703)]+[str(i) for i in range(9714, 9720)]+[str(i) for i in range(9724, 9729)]),miss_task] = '33042'
            dftokens.loc[dftokens.subsite.isin([v for v in subsite_list if v not in ["C024", "C111", "C142", "C379", "C422", "C098", "C099"]+["C"+str(i) for i in range(770,780)]]) & dftokens.histology.isin(["9823", "9827", "9837"]+[str(i) for i in range(9811, 9819)]),miss_task] = '33042'
    
            #Myeloma
            dftokens.loc[dftokens.subsite.isin(subsite_list) & dftokens.histology.isin(list(map(str, range(9731, 9733)))+["9734"]),miss_task] = '34000'
    
            #Leukemia
            dftokens.loc[dftokens.subsite.isin(subsite_list) & dftokens.histology.isin(["9826", "9835", "9836"]),miss_task] = '35011'
            dftokens.loc[dftokens.subsite.isin(["C420","C421","C424"]) & dftokens.histology.isin(list(map(str, range(9811, 9819)))+["9837"]),miss_task] = '35011'
            dftokens.loc[dftokens.subsite.isin(["C420","C421","C424"]) & dftokens.histology.isin(["9823"]),miss_task] = '35012'
            dftokens.loc[dftokens.subsite.isin(subsite_list) & dftokens.histology.isin(list(map(str, range(9832, 9835)))+["9820", "9940"]),miss_task] = '35013'
            dftokens.loc[dftokens.subsite.isin(subsite_list) & dftokens.histology.isin(list(map(str, range(9871, 9874)))+["9840", "9861", "9865", "9866", "9867", "9869", "9895", "9896", "9897", "9898", "9910", "9911", "9920"]),miss_task] = '35021'
            dftokens.loc[dftokens.subsite.isin(subsite_list) & dftokens.histology.isin(["9891"]),miss_task] = '35031'
            dftokens.loc[dftokens.subsite.isin(subsite_list) & dftokens.histology.isin(["9863", "9875", "9876", "9945", "9946"]),miss_task] = '35022'
            dftokens.loc[dftokens.subsite.isin(subsite_list) & dftokens.histology.isin(["9860", "9930"]),miss_task] = '35023'
            dftokens.loc[dftokens.subsite.isin(subsite_list) & dftokens.histology.isin(list(map(str, range(9805, 9810)))+["9801", "9931"]),miss_task] = '35041'
            dftokens.loc[dftokens.subsite.isin(subsite_list) & dftokens.histology.isin(["9733", "9742", "9800", "9831", "9870", "9948", "9963", "9964"]),miss_task] = '35043'
            dftokens.loc[dftokens.subsite.isin(["C420","C421","C424"]) & dftokens.histology.isin(["9827"]),miss_task] = '35043'
    
            #Mesothelioma
            dftokens.loc[dftokens.subsite.isin(subsite_list) & dftokens.histology.isin(list(map(str, range(9050,9056)))),miss_task] = '36010'
    
            #Kaposi
            dftokens.loc[dftokens.subsite.isin(subsite_list) & dftokens.histology.isin(["9140"]),miss_task] = '36020'
    
            #Miscellaneous
            dftokens.loc[dftokens.subsite.isin(subsite_list) & dftokens.histology.isin(["9740", "9741", "9950", "9960", "9961", "9962", "9965", "9966", "9967", "9970", "9971", "9975", "9980", "9989", "9991", "9992"]+list(map(str, range(9750,9770)))+list(map(str, range(9982,9988)))),miss_task] = '37000'
            dftokens.loc[dftokens.subsite.isin(["C809"]+["C"+str(i) for i in range(760,769)]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '37000'
            dftokens.loc[dftokens.subsite.isin(["C"+str(i) for i in range(420,425)]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '37000'
            dftokens.loc[dftokens.subsite.isin(["C"+str(i) for i in range(770,780)]) & dftokens.histology.isin([v for v in histology_list if v not in list(map(str, range(9050,9056)))+["9140"]+list(map(str, range(9590,9993)))]),miss_task] = '37000'
    
            #Invalid
            dftokens.loc[dftokens.recode=="",miss_task]="99999"

        return dftokens

    def ICCC_creation(self, dftokens, miss_tasks):

        '''
        Func.tion to filter columns for the biomarkers task.
            
        Parameters:
           dftokens (pd.DataFrame): records being processed
           miss_tasks (list): new tasks to add in original data
        '''

        if "ICCC" in miss_tasks:
            miss_task = "ICCC"
            dftokens[miss_task] = ''
            
            #filter age<40 and set up minor filters
            dftokens = dftokens[(pd.to_datetime(dftokens['dateOfDiagnosis'], format = '%Y%m%d', errors='coerce')-pd.to_datetime(dftokens['dateOfBirth'], format='%Y%m', errors='coerce')).astype('<m8[Y]')<40]
            minor = (pd.to_datetime(dftokens['dateOfDiagnosis'], format = '%Y%m%d', errors='coerce')-pd.to_datetime(dftokens['dateOfBirth'], format='%Y%m', errors='coerce')).astype('<m8[Y]')<19

            dftokens.loc[dftokens.histology.isin([str(i) for i in range(9811,9819)]+["9823","9827","9837", "9591"]) & dftokens.subsite.isin(["C420","C421","C423", "C424", "C809"]) & (dftokens.behavior == '3') & minor,miss_task] = '011'
            dftokens.loc[dftokens.histology.isin(["9826" ,"9831" ,"9832" ,"9833", "9834","9835","9836", "9940", "9948", "9820"]) & (dftokens.behavior == '3') & minor,miss_task] = '011'
            dftokens.loc[dftokens.histology.isin(["9840" ,"9861" ,"9865" ,"9866", "9867","9891","9895", "9896", "9897","9898", "9910", "9911", "9920", "9930", "9931"]+[str(i) for i in range(9869,9875)]) & (dftokens.behavior == '3') & minor,miss_task] = '012'
            dftokens.loc[dftokens.histology.isin(["9863" ,"9875" ,"9876" ,"9950"]+[str(i) for i in range(9960,9965)]) & (dftokens.behavior == '3') & minor,miss_task] = '013'
            dftokens.loc[dftokens.histology.isin(["9945" ,"9946" ,"9975" ,"9980", "9989", "9991", "9992"]+[str(i) for i in range(9982,9988)]) & (dftokens.behavior == '3') & minor,miss_task] = '014'
            dftokens.loc[dftokens.histology.isin(["9800" ,"9801" ,"9860" ,"9965", "9966", "9967"]+[str(i) for i in range(9805,9810)]) & (dftokens.behavior == '3') & minor,miss_task] = '015'
            dftokens.loc[dftokens.histology.isin(["9659" ,"9667"]+[str(i) for i in range(9650,9656)]+[str(i) for i in range(9661,9666)]) & (dftokens.behavior == '3') & minor,miss_task] = '021'
            dftokens.loc[dftokens.histology.isin(["9700", "9701", "9702", "9705","9708", "9709", "9714", "9716", "9717", "9718", "9719", "9724", "9725", "9726", "9727", "9728", "9729", "9597","9670","9671","9673","9675","9678","9679", "9680", "9684", "9688", "9689", "9690", "9691", "9695", "9698", "9699", "9712", "9737", "9738", "9760", "9761", "9762", "9970", "9971"]+[str(i) for i in range(9731,9736)]+[str(i) for i in range(9764,9770)]) & (dftokens.behavior == '3') & minor,miss_task] = '022'
            dftokens.loc[dftokens.histology.isin(["9837" ,"9823" ,"9827" ,"9591"]+[str(i) for i in range(9811,9819)]) & dftokens.subsite.isin(["C422"]+["C"+str(i).zfill(3) for i in range(000,420)]+["C"+str(i) for i in range(440,780)]) & (dftokens.behavior == '3') & minor,miss_task] = '022'
            dftokens.loc[dftokens.histology.isin(["9687"]) & (dftokens.behavior == '3') & minor,miss_task] = '023'
            dftokens.loc[dftokens.histology.isin(["9740" ,"9741" ,"9742" ,"9750", "9751"]+[str(i) for i in range(9754,9760)]) & (dftokens.behavior == '3') & minor,miss_task] = '024'
            dftokens.loc[dftokens.histology.isin(["9590" ,"9596"]) & (dftokens.behavior == '3') & minor,miss_task] = '025'
    
            dftokens.loc[dftokens.histology.isin(["9383" ,"9396"]+[str(i) for i in range(9391,9395)]) & dftokens.behavior.isin(['0','1','3']) & minor,miss_task] = '031'
            dftokens.loc[dftokens.histology.isin(["9390"]) & dftokens.behavior.isin(['0','1','3']) & minor,miss_task] = '031'
            dftokens.loc[dftokens.histology.isin(["9380"]) & dftokens.subsite.isin(["C723"]) & dftokens.behavior.isin(['0','1','3']) & minor,miss_task] = '032'
            dftokens.loc[dftokens.histology.isin(["9384" ,"9425", "9440", "9441", "9442"]+[str(i) for i in range(9400,9412)]+[str(i) for i in range(9420,9425)]) & dftokens.behavior.isin(['0','1','3']) & minor,miss_task] = '032'
            dftokens.loc[dftokens.histology.isin(["9470", "9471", "9472", "9473", "9480", "9508"]+[str(i) for i in range(9474,9479)]) & dftokens.behavior.isin(['0','1','3']) & minor,miss_task] = '033'
            dftokens.loc[dftokens.histology.isin(["9501", "9502", "9503", "9504"]) & dftokens.subsite.isin(["C"+str(i) for i in range(700,730)]) & dftokens.behavior.isin(['0','1','3']) & minor,miss_task] = '033'
            dftokens.loc[dftokens.histology.isin(["9450", "9451", "9460", "9382", "9385", "9381", "9430", "9431", "9444", "9445"]) & dftokens.behavior.isin(['0','1','3']) & minor,miss_task] = '034'
            dftokens.loc[dftokens.histology.isin(["9380"]) & dftokens.subsite.isin(["C751", "C753"]+["C"+str(i) for i in range(700,723)]+["C"+str(i) for i in range(724,730)]) & dftokens.behavior.isin(['0','1','3']) & minor,miss_task] = '034'
            dftokens.loc[dftokens.histology.isin(["8158", "8290"]) & dftokens.subsite.isin(["C751"]) & dftokens.behavior.isin(['0','1','3']) & minor,miss_task] = '035'
            dftokens.loc[dftokens.histology.isin(["8300" ,"9350", "9351", "9352", "9432", "9582", "9360", "9361", "9362", "9395", "9412", "9413", "9492", "9493", "9505", "9506", "9507", "9509"]+[str(i) for i in range(8270,8282)]+[str(i) for i in range(9530,9540)]) & dftokens.behavior.isin(['0','1','3']) & minor,miss_task] = '035'
            dftokens.loc[dftokens.histology.isin([str(i) for i in range(8000,8006)]) & dftokens.subsite.isin(["C751", "C752", "C753"]+["C"+str(i) for i in range(700,730)]) & dftokens.behavior.isin(['0','1','3']) & minor,miss_task] = '036'
            dftokens.loc[dftokens.histology.isin(["9490", "9500"]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '041'
            dftokens.loc[dftokens.histology.isin(["8680", "8681", "8682", "8683", "8690", "8691", "8692", "8693", "8700", "9520", "9521", "9522", "9523"]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '042'
            dftokens.loc[dftokens.histology.isin([str(i) for i in range(9501,9505)]) & dftokens.subsite.isin(["C809"]+["C"+str(i).zfill(3) for i in range(000,700)]+["C"+str(i) for i in range(739,769)]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '042'
            dftokens.loc[dftokens.histology.isin([str(i) for i in range(9510,9515)]) & dftokens.behavior.isin(['3']),miss_task] = '050'
            dftokens.loc[dftokens.histology.isin(["8959", "8960"]+[str(i) for i in range(8964,8968)]) & dftokens.behavior.isin(['3']),miss_task] = '061'
            dftokens.loc[dftokens.histology.isin(["8963"]) & dftokens.subsite.isin(["C649"]) & dftokens.behavior.isin(['3']),miss_task] = '061'
            dftokens.loc[dftokens.histology.isin(["8082", "8120", "8121", "8122", "8143", "8155", "8210", "8211", "8240", "8241", "8244", "8245", "8246", "8260", "8261", "8262", "8263", "8290", "8310", "8320" ,"8323", "8325", "8401", "8430", "8440", "8504", "8510", "8550", "8560", "8561", "8562", "8570", "8571", "8572", "8573", "9013"]+[str(i) for i in range(8010,8042)]+[str(i) for i in range(8050,8076)]+[str(i) for i in range(8130,8142)]+[str(i) for i in range(8190,8202)]+[str(i) for i in range(8221,8232)]+[str(i) for i in range(8480,8491)]) & dftokens.subsite.isin(["C649"]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '062'
            dftokens.loc[dftokens.histology.isin(["8311", "8312", "8316", "8317", "8318", "8319", "8361"]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '062'
            dftokens.loc[dftokens.histology.isin([str(i) for i in range(8000,8006)]) & dftokens.subsite.isin(["C649"]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '063'
            dftokens.loc[dftokens.histology.isin(["8970", "8975"]) & dftokens.behavior.isin(['3']),miss_task] = '071'
            dftokens.loc[dftokens.histology.isin(["8963", "8991"]) & dftokens.subsite.isin(["C220", "C221"]) & dftokens.behavior.isin(['3']),miss_task] = '071'
            dftokens.loc[dftokens.histology.isin(["8082", "8120", "8121", "8122", "8140", "8141", "8145", "8148", "8155", "8158", "8202", "8210", "8211", "8230", "8231", "8240", "8241", "8244", "8245", "8246" ,"8310", "8320", "8323", "8401", "8430", "8440", "8470", "8503", "8504", "8510", "8550", "8560", "8561", "8562", "8570", "8571", "8572", "8573", "9013"]+[str(i) for i in range(8010,8042)]+[str(i) for i in range(8050,8076)]+[str(i) for i in range(8190,8202)]+[str(i) for i in range(8221,8232)]+[str(i) for i in range(8260,8265)]+[str(i) for i in range(8480,8491)]) & dftokens.subsite.isin(["C220", "C221"]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '072'
            dftokens.loc[dftokens.histology.isin(["8160", "8161", "8162"]+[str(i) for i in range(8170,8181)]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '072'
            dftokens.loc[dftokens.histology.isin([str(i) for i in range(8000,8006)]) & dftokens.subsite.isin(["C220", "C221"]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '073'
            dftokens.loc[dftokens.histology.isin(["9200"]+[str(i) for i in range(9180,9188)]+[str(i) for i in range(9191,9196)]) & dftokens.subsite.isin(["C809"]+["C"+str(i) for i in range(400,420)]+["C"+str(i) for i in range(760,769)]) & dftokens.behavior.isin(['3']),miss_task] = '081'
            dftokens.loc[dftokens.histology.isin(["9210", "9220", "9240"]) & dftokens.subsite.isin(["C809"]+["C"+str(i) for i in range(400,420)]+["C"+str(i) for i in range(760,769)]) & dftokens.behavior.isin(['3']),miss_task] = '082'
            dftokens.loc[dftokens.histology.isin(["9221", "9222", "9230"]+[str(i) for i in range(9211,9214)]+[str(i) for i in range(9241,9244)]) & dftokens.behavior.isin(['3']),miss_task] = '082'
            dftokens.loc[dftokens.histology.isin(["9231"]) & dftokens.subsite.isin(["C"+str(i) for i in range(400,420)]) & dftokens.behavior.isin(['3']),miss_task] = '082'
            dftokens.loc[dftokens.histology.isin(["9260"]) & dftokens.subsite.isin(["C809"]+["C"+str(i) for i in range(400,420)]+["C"+str(i) for i in range(760,769)]) & dftokens.behavior.isin(['3']),miss_task] = '083'
            dftokens.loc[dftokens.histology.isin(["9365", "9364"]) & dftokens.subsite.isin(["C"+str(i) for i in range(400,420)]) & dftokens.behavior.isin(['3']),miss_task] = '083'
            dftokens.loc[dftokens.histology.isin(["8810", "8811", "8818", "8823", "8830"]) & dftokens.subsite.isin(["C"+str(i) for i in range(400,420)]) & dftokens.behavior.isin(['3']),miss_task] = '084'
            dftokens.loc[dftokens.histology.isin(["8812", "9262", "9370", "9371", "9372", "9280", "9281", "9282", "9290", "9300", "9301", "9302", "9310", "9311", "9312", "9320", "9321", "9322", "9330", "9250", "9261"]+[str(i) for i in range(9270,9276)]+[str(i) for i in range(9340,9343)]) & dftokens.behavior.isin(['3']),miss_task] = '084'
            dftokens.loc[dftokens.histology.isin(["8800", "8801", "8803", "8804", "8805"]+[str(i) for i in range(8000,8006)]) & dftokens.subsite.isin(["C"+str(i) for i in range(400,420)]) & dftokens.behavior.isin(['3']),miss_task] = '085'
            dftokens.loc[dftokens.histology.isin(["8910", "8912", "8920"]+[str(i) for i in range(8900,8906)]) & dftokens.behavior.isin(['3']),miss_task] = '091'
            dftokens.loc[dftokens.histology.isin(["8991"]) & dftokens.subsite.isin(["C"+str(i).zfill(3) for i in range(000,219)]+["C"+str(i) for i in range(239,810)]) & dftokens.behavior.isin(['3']),miss_task] = '091'
            dftokens.loc[dftokens.histology.isin(["8810", "8811", "8821", "8823", "8834", "8835"]+[str(i) for i in range(8813,8818)]) & dftokens.subsite.isin(["C809"]+["C"+str(i).zfill(3) for i in range(000,400)]+["C"+str(i) for i in range(440,769)]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '092'
            dftokens.loc[dftokens.histology.isin(["8820", "8822", "9150", "9160", "9491", "9580"]+[str(i) for i in range(8824,8829)]+[str(i) for i in range(9540,9572)]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '092'
            dftokens.loc[dftokens.histology.isin(["9140"]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '093'
            dftokens.loc[dftokens.histology.isin(["9260"]) & dftokens.subsite.isin(["C"+str(i).zfill(3) for i in range(000,400)]+["C"+str(i) for i in range(440,450)]+["C"+str(i) for i in range(470,760)]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '094'
            dftokens.loc[dftokens.histology.isin(["9365"]) & dftokens.subsite.isin(["C809"]+["C"+str(i).zfill(3) for i in range(000,400)]+["C"+str(i) for i in range(470,640)]+["C"+str(i) for i in range(659,769)]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '094'
            dftokens.loc[dftokens.histology.isin(["9364"]) & dftokens.subsite.isin(["C809"]+["C"+str(i).zfill(3) for i in range(000,400)]+["C"+str(i) for i in range(440,450)]+["C"+str(i) for i in range(470,700)]+["C"+str(i) for i in range(739,769)]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '094'
            dftokens.loc[dftokens.histology.isin(["8963"]) & dftokens.subsite.isin(["C809"]+["C"+str(i).zfill(3) for i in range(000,219)]+["C"+str(i) for i in range(239,640)]+["C"+str(i) for i in range(659,700)]+["C"+str(i) for i in range(739,769)]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '094'
            dftokens.loc[dftokens.histology.isin(["8860", "8861", "8862", "8870", "8880", "8881", "8831", "8832", "8833", "8836", "9251", "9252", "9141", "9142", "9161", "9581", "8587", "8806", "8840", "8841", "8842", "8921", "8990", "8992", "9045", "9373"]+[str(i) for i in range(8710,8715)]+[str(i) for i in range(9170,9176)]+[str(i) for i in range(8850,8859)]+[str(i) for i in range(8890,8899)]+[str(i) for i in range(9040,9045)]+[str(i) for i in range(9120,9126)]+[str(i) for i in range(9130,9134)]+[str(i) for i in range(9135,9139)]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '094'
            dftokens.loc[dftokens.histology.isin(["8830", "9231"]) & dftokens.subsite.isin(["C809"]+["C"+str(i).zfill(3) for i in range(000,400)]+["C"+str(i) for i in range(440,769)]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '094'
            dftokens.loc[dftokens.histology.isin(["9200", "9210", "9220","9240"]+[str(i) for i in range(9180,9188)]+[str(i) for i in range(9191,9196)]) & dftokens.subsite.isin(["C220", "C649", "C696"]+["C"+str(i) for i in range(300,389)]+["C"+str(i) for i in range(470,510)]+["C"+str(i) for i in range(600,620)]+["C"+str(i) for i in range(670,680)]+["C"+str(i) for i in range(700,730)]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '094'
            dftokens.loc[dftokens.histology.isin([str(i) for i in range(8800,8806)]) & dftokens.subsite.isin(["C809"]+["C"+str(i).zfill(3) for i in range(000,400)]+["C"+str(i) for i in range(440,769)]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '095'
            dftokens.loc[dftokens.histology.isin(["9070", "9071", "9072", "9100", "9101", "9085"]+[str(i) for i in range(9060,9066)]+[str(i) for i in range(9080,9085)]) & dftokens.subsite.isin(["C"+str(i) for i in range(700,730)]+["C"+str(i) for i in range(751,754)]) & dftokens.behavior.isin(['0','1','3']) & minor,miss_task] = '101'
            dftokens.loc[dftokens.histology.isin(["9070", "9071", "9072", "9100", "9101", "9103", "9104", "9105", "9085", "9086"]+[str(i) for i in range(9060,9066)]+[str(i) for i in range(9080,9085)]) & dftokens.subsite.isin(["C809"]+["C"+str(i).zfill(3) for i in range(000,560)]+["C"+str(i) for i in range(570,620)]+["C"+str(i) for i in range(630,700)]+["C"+str(i) for i in range(739,751)]+["C"+str(i) for i in range(754,769)]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '102'
            dftokens.loc[dftokens.histology.isin(["9070", "9071", "9072", "9073", "9090", "9091", "9085"]+[str(i) for i in range(9060,9066)]+[str(i) for i in range(9080,9085)]+[str(i) for i in range(9100,9106)]) & dftokens.subsite.isin(["C569"]+["C"+str(i) for i in range(620,630)]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '103'
            dftokens.loc[dftokens.histology.isin(["8044", "8082", "8143", "8153", "8210", "8211", "8213", "8244", "8245", "8246", "8290", "8310", "8320", "8323", "8430", "8440", "8441", "8442", "8450", "8470", "8504", "8510", "8550", "8560", "8561", "8562", "9000", "9014"]+[str(i) for i in range(8010,8042)]+[str(i) for i in range(8050,8076)]+[str(i) for i in range(8120,8124)]+[str(i) for i in range(8130,8142)]+[str(i) for i in range(8190,8202)]+[str(i) for i in range(8221,8242)]+[str(i) for i in range(8260,8264)]+[str(i) for i in range(8380,8385)]+[str(i) for i in range(8480,8491)]+[str(i) for i in range(8570,8574)]) & dftokens.subsite.isin(["C569"]+["C"+str(i) for i in range(620,630)]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '104'
            dftokens.loc[dftokens.histology.isin(["8313", "8443", "8444", "8451", "8474", "9015"]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '104'
            dftokens.loc[dftokens.histology.isin(["8158", "8243", "8410", "8452", "8460", "8461", "8462", "8463", "8471", "8472", "8473", "8576"]) & dftokens.subsite.isin(["C569"]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '104'
            dftokens.loc[dftokens.histology.isin([str(i) for i in range(8590,8672)]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '105'
            dftokens.loc[dftokens.histology.isin(["8000", "8001", "8002", "8003", "8004", "8005"]) & dftokens.subsite.isin(["C569"]+["C"+str(i) for i in range(620,630)]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '105'
            dftokens.loc[dftokens.histology.isin(["8158"]) & dftokens.subsite.isin(["C740"]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '111'
            dftokens.loc[dftokens.histology.isin([str(i) for i in range(8370,8376)]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '111'
            dftokens.loc[dftokens.histology.isin([ "8082", "8158", "8190", "8200", "8201", "8211", "8230", "8231", "8244", "8245", "8246", "8260", "8261", "8262", "8263", "8290", "8310", "8320", "8323", "8324", "8333", "8430", "8440", "8480", "8481", "8510", "8589"]+[str(i) for i in range(8010,8042)]+[str(i) for i in range(8050,8076)]+[str(i) for i in range(8120,8123)]+[str(i) for i in range(8130,8142)]+[str(i) for i in range(8190,8202)]+[str(i) for i in range(8221,8242)]+[str(i) for i in range(8260,8263)]+[str(i) for i in range(8570,8574)]) & dftokens.subsite.isin(["C739"]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '112'
            dftokens.loc[dftokens.histology.isin(["8350", "8588"]+[str(i) for i in range(8330,8333)]+[str(i) for i in range(8334,8338)]+[str(i) for i in range(8340,8348)]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '112'
            dftokens.loc[dftokens.histology.isin([ "8082", "8083", "8190", "8200", "8201", "8211", "8230", "8231", "8244", "8245", "8246", "8260", "8261", "8262", "8263", "8290", "8310", "8320", "8323", "8430", "8440", "8480", "8481"]+[str(i) for i in range(8010,8042)]+[str(i) for i in range(8050,8076)]+[str(i) for i in range(8120,8123)]+[str(i) for i in range(8130,8142)]+[str(i) for i in range(8500,8552)]+[str(i) for i in range(8560,8563)]+[str(i) for i in range(8570,8577)]) & dftokens.subsite.isin(["C"+str(i) for i in range(110,120)]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '113'
            dftokens.loc[dftokens.histology.isin(["8790"]+[str(i) for i in range(8720,8781)]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '114'
            dftokens.loc[dftokens.histology.isin([ "8078", "8081", "8082", "8140", "8143", "8147", "8190", "8200", "8211", "8240", "8246", "8247", "8260", "8263", "8310", "8320", "8323", "8430", "8480", "8540", "8542", "8560"]+[str(i) for i in range(8010,8042)]+[str(i) for i in range(8050,8076)]+[str(i) for i in range(8090,8099)]+[str(i) for i in range(8100,8111)]+[str(i) for i in range(8390,8421)]+[str(i) for i in range(8570,8574)]) & dftokens.subsite.isin(["C519", "C609", "C632"]+["C"+str(i) for i in range(440,450)]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '115'
            dftokens.loc[dftokens.histology.isin([ "8098", "8163", "8290", "8310", "8314", "8315", "8333", "8360", "8401", "8402", "8405", "8406", "8410", "8450", "8460", "8461", "8470", "8589", "8941", "8983", "9000", "9016", "9020", "9030"]+[str(i) for i in range(8010,8078)]+[str(i) for i in range(8080,8087)]+[str(i) for i in range(8120,8159)]+[str(i) for i in range(8190,8266)]+[str(i) for i in range(8320,8325)]+[str(i) for i in range(8380,8385)]+[str(i) for i in range(8430,8443)]+[str(i) for i in range(8452,8455)]+[str(i) for i in range(8480,8587)]+[str(i) for i in range(9010,9015)]) & dftokens.subsite.isin(["C"+str(i).zfill(3) for i in range(79,90)]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '116'
            dftokens.loc[dftokens.histology.isin([ "8098", "8163", "8290", "8310", "8314", "8315", "8333", "8360", "8401", "8402", "8405", "8406", "8410", "8450", "8460", "8461", "8470", "8589", "8941", "8983", "9000", "9016", "9020", "9030"]+[str(i) for i in range(8010,8078)]+[str(i) for i in range(8080,8087)]+[str(i) for i in range(8120,8159)]+[str(i) for i in range(8190,8266)]+[str(i) for i in range(8320,8325)]+[str(i) for i in range(8380,8385)]+[str(i) for i in range(8430,8443)]+[str(i) for i in range(8452,8455)]+[str(i) for i in range(8480,8587)]+[str(i) for i in range(9010,9015)]) & dftokens.subsite.isin(["C181", "C379", "C809"]+["C"+str(i) for i in range(760,769)]+["C"+str(i) for i in range(690,700)]+["C"+str(i) for i in range(670,680)]+["C"+str(i) for i in range(530,540)]+["C"+str(i) for i in range(500,510)]+["C"+str(i).zfill(3) for i in range(79,90)]+["C"+str(i) for i in range(340,350)]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '116'
            dftokens.loc[dftokens.histology.isin([ "8098", "8163", "8290", "8310", "8314", "8315", "8333", "8360", "8401", "8402", "8405", "8406", "8410", "8450", "8460", "8461", "8470", "8589", "8941", "8983", "9000", "9016", "9020", "9030"]+[str(i) for i in range(8010,8078)]+[str(i) for i in range(8080,8087)]+[str(i) for i in range(8120,8159)]+[str(i) for i in range(8190,8266)]+[str(i) for i in range(8320,8325)]+[str(i) for i in range(8380,8385)]+[str(i) for i in range(8430,8443)]+[str(i) for i in range(8452,8455)]+[str(i) for i in range(8480,8587)]+[str(i) for i in range(9010,9015)]) & dftokens.subsite.isin(["C180", "C199", "C209"]+["C"+str(i) for i in range(182,190)]+["C"+str(i) for i in range(210,219)]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '116'
            dftokens.loc[dftokens.histology.isin([ "8098", "8163", "8310", "8314", "8315", "8333", "8360", "8401", "8402", "8405", "8406", "8410", "8450", "8460", "8461", "8470", "8589", "8941", "8983", "9000", "9016", "9020", "9030"]+[str(i) for i in range(8010,8078)]+[str(i) for i in range(8080,8087)]+[str(i) for i in range(8120,8159)]+[str(i) for i in range(8190,8266)]+[str(i) for i in range(8320,8325)]+[str(i) for i in range(8380,8385)]+[str(i) for i in range(8430,8443)]+[str(i) for i in range(8452,8455)]+[str(i) for i in range(8480,8587)]+[str(i) for i in range(9010,9015)]) & dftokens.subsite.isin(["C529", "C619", "C630", "C631"]+["C"+str(i).zfill(3) for i in range(0,70)]+["C"+str(i).zfill(3) for i in range(90,110)]+["C"+str(i) for i in range(129,180)]+["C"+str(i) for i in range(239,340)]+["C"+str(i) for i in range(380,400)]+["C"+str(i) for i in range(480,489)]+["C"+str(i) for i in range(510,519)]+["C"+str(i) for i in range(540,560)]+["C"+str(i) for i in range(570,609)]+["C"+str(i) for i in range(637,640)]+["C"+str(i) for i in range(659,670)]+["C"+str(i) for i in range(680,690)]+["C"+str(i) for i in range(700,730)]+["C"+str(i) for i in range(750,760)]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '116'
            dftokens.loc[dftokens.histology.isin([ "8158", "8290"]) & dftokens.subsite.isin(["C750"]+["C"+str(i).zfill(3) for i in range(0,70)]+["C"+str(i).zfill(3) for i in range(90,110)]+["C"+str(i) for i in range(129,180)]+["C"+str(i) for i in range(239,340)]+["C"+str(i) for i in range(380,400)]+["C"+str(i) for i in range(480,489)]+["C"+str(i) for i in range(510,530)]+["C"+str(i) for i in range(540,560)]+["C"+str(i) for i in range(570,620)]+["C"+str(i) for i in range(630,640)]+["C"+str(i) for i in range(659,670)]+["C"+str(i) for i in range(680,690)]+["C"+str(i) for i in range(752,760)]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '116'
            dftokens.loc[dftokens.histology.isin(["8936", "8971", "8972", "8973", "8940", "8950", "8951", "8974", "9110", "9363"]+[str(i) for i in range(8930,8936)]+[str(i) for i in range(8980,8983)]+[str(i) for i in range(9050,9056)]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '121'
            dftokens.loc[dftokens.histology.isin([str(i) for i in range(8000,8006)]) & dftokens.subsite.isin(["C"+str(i).zfill(3) for i in range(0,218)]+["C"+str(i) for i in range(239,400)]+["C"+str(i) for i in range(420,560)]+["C"+str(i) for i in range(570,620)]+["C"+str(i) for i in range(630,640)]+["C"+str(i) for i in range(659,700)]+["C"+str(i) for i in range(739,751)]+["C"+str(i) for i in range(754,810)]) & dftokens.behavior.isin(['3']) & minor,miss_task] = '122'
    
            #invalid
            dftokens.loc[(dftokens.ICCC=="") & minor,miss_task]="999"

        return dftokens
