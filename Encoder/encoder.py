class Encoder():
    '''
    This class will prepare the reports for model usage which includes use in the model. 
    '''
    def __init__(self, cc, 
                    metadata_fields=['registryId','recordDocumentId','patientId','tumorId','pathDateSpecCollect1_year',
                    'rxDateMostDefinSurg_pathDateSpecCollect1_diff','rxDateSurgery_pathDateSpecCollect1_diff',
                    'dateOfDiagnosis_pathDateSpecCollect1_diff'],
                    reverse_tokens = False, remove_unks = False, sequence_length = 3000, **kwargs):
        '''
        Initializer of the Encoder class.  Defines the "attributes".

        Parameters:
            cc (class): the CachedClass will be passed through all the modules and keep track of the pipeline arguements
            metadata_fields (list): list of all the names of the metadata desired
            reverse_tokens (boolean): this will reverse the documents to read the last tokens first.
            remove_unks (boolean): this will remove the unknown tokens from the reports (as opposed to mapping it to an unk term.
            sequence_length (int): max token length that will be used for the records AFTER it has been ordered (or reversed). 
        '''
        assert cc.check_args('Encoder', locals()), 'mismatch between expected args and what is being used'
        self.pickle_files = {}
        self.text_files = {}
        cc.saved_mod_args['metadata_fields'] = (metadata_fields, 'Encoder', 'List of all the names of the metadata desired. These will be checked against the schema used. Data fields and baseline fields will be excluded. dates will only be availible in xxx_year and xxx_yyyy_diff format.')
        cc.saved_mod_args['reverse_tokens'] = (reverse_tokens, 'Encoder', 'This will reverse the documents to read the last tokens first.')
        cc.saved_mod_args['remove_unks'] = (remove_unks, 'Encoder', 'This will remove the unknown tokens from the reports (as opposed to mapping it to an unk term).')
        cc.saved_mod_args['sequence_length'] = (sequence_length, 'Encoder', 'Max token length that will be used for the records AFTER it has been ordered (or reversed).')

    def build_dataset(self, cc):
        '''
        this is the function used to continue the build of the model within the pipeline

        Parameters:
            cc (class): the CachedClass will be passed through all the modules

        Modules:
            check_args: will verify the arguements are appropriate for the options chosen.
            build_vocab: will generate the X values for the model and the input map
            build_labels: will generate the Y values for the model and the output layer
            describe: creates a report of what the filter did.
        '''
        mod_name = self.__module__.split('.')[-2]
        mod_args = cc.all_mod_args()
        if not cc.test_cache(mod_name=mod_name, del_dirty=cc.remove_dirty, mod_args=mod_args):
            with cc.timing("extracting data for Encoder"):
                describe, schema, records, miss_tasks, folds, vocab = self.get_cached_deps(cc)
                mt = [v for v in miss_tasks.columns if v not in records.columns]
                for c in mt:
                    records[c] = miss_tasks[c]
                records.dropna(subset = mt, inplace = True)
                self.check_args(cc.logging.debug, mod_args, schema)
                data_fields = schema[schema['tables'][0]]['data_fields']
                index_fields = sorted(set([f.split(' ')[-1] for t in schema['tables'] for f in schema[t]['order_id_fields']]))
            for i, fold in enumerate(folds):
                with cc.timing(f'tokenizing records to fit model inputs for fold {i}'):
                    self.generate_data(cc.logging.debug, records, index_fields, mod_args['tasks'], mod_args['metadata_fields'], mod_args['remove_breaktoken']!='all', 
                           data_fields, mod_args['concat_split'], fold, i, vocab[i], mod_args['reverse_tokens'], mod_args['remove_unks'], mod_args['sequence_length'])
            self.describe(describe, mod_args)
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
        data_fields = stored['schema'][stored['schema']['tables'][0]]['data_fields']
        record_break = args['remove_breaktoken'] != 'all'
        text = self.concat_text(data, data_fields, record_break)
        data = self.sequence_text(text, bool(args['reverse_tokens']), bool(args['remove_unks']), 
                                    stored['word2tok'], stored['vocab'], int(args['sequence_length']))
        return [data]

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
        import numpy as np
        data_fields = stored['schema'][stored['schema']['tables'][0]]['data_fields']
        word2tok = stored['word2tok']
        vocab = stored['vocab']
        reverse_tokens = bool(args['reverse_tokens'])
        remove_unks = bool(args['remove_unks'])
        sequence_length = int(args['sequence_length'])
        record_break = args['remove_breaktoken'] != 'all'
        text = self.concat_text(data, data_fields, record_break)

        def applyfield(d, df):
            d['seq'] = df.apply(lambda v: self.sequence_text(v, reverse_tokens, remove_unks, word2tok, vocab, sequence_length))
            logging(str(list(d['seq'])))
        text = text.rename('concat_data_field').to_frame()
        if text.shape[0] > 10:
            # use multiprocessing for incresed speed
            from multiprocessing import Process, Manager
            seqtext = []
            for j in range(0, len(text), 100000):

                manager = Manager()
                seqtext_temp = manager.dict()
                p = Process(target=applyfield, args=(seqtext_temp, text['concat_data_field'][j:j+100000]))
                p.start()
                p.join()
                seqtext += list(seqtext_temp['seq'])
        else:
            seqtext = text['concat_data_field'].apply(lambda v: self.sequence_text(v, reverse_tokens, remove_unks, word2tok, vocab, sequence_length))
        return np.array(seqtext, dtype=object)

    def get_cached_deps(self, cc):
        '''
        This function pulls the prior cached data

        Parameters:
            cc (class): the CachedClass will identify the previous caches.

        Returns:
            data (tuple): collection of the different cached data that was recieved:
                          
        '''
        fold_num = int(cc.get_cache_file('Splitter', 'fold_num'))

        data = (
             cc.get_cache_file('Data_Generator', 'describe'),
             cc.get_cache_file('Parser', 'db_schema'),
             cc.get_cache_file('Sanitizer', 'df_tokens'),
             cc.get_cache_file('Filter', 'missing_tasks'), 
             [[
                 cc.get_cache_file('Splitter', f'train_fold{i}'),
                 cc.get_cache_file('Splitter', f'val_fold{i}'),
                 cc.get_cache_file('Splitter', f'test_fold{i}')] for i in range(fold_num)],
             [cc.get_cache_file('Data_Generator', f'id2word_fold{i}') for i in range(fold_num)])
        return data

    def check_args(self, logging, mod_args, schema):
        '''
        This will verify the arguments that have been provided to make sure are expected.

        Parameters:
            logging (function): will return the print statement
            mod_args (dict): the saved model args to identify how to filter the text as well as everything else
        '''
        assert isinstance(mod_args['metadata_fields'],list), f'metadata_fields is expected to be a list.  Got {type(mod_args["metadata_fields"])}.'
        key_date_field = schema[schema['tables'][0]]['key_date'][0]
        date_fields = [f'{d}_{key_date_field}_diff' for d in schema[schema['tables'][1]]['date_fields']] + \
                      [f'{d}_year' for d in schema[schema['tables'][1]]['date_fields']] + [f'{key_date_field}_year']
        fields = date_fields + \
                     [v.split(' ')[-1] for table in [t_group for t_name,t_group in schema.items() if t_name!="tables"] 
                                     for group in [g_fields for g_name,g_fields in table.items() 
                                     if 'date' not in g_name and 'baseline' not in g_name] 
                                     for v in group]
        logging('possible metadata fields: ' + str(fields))
        bad_metadata = [v for v in mod_args['metadata_fields'] if v not in fields]
        assert bad_metadata == [], f'was not expecting the following metadata fields: {", ".join(bad_metadata)}.'
        assert isinstance(mod_args['reverse_tokens'], bool), f'reverse_tokens was expected to be a boolean value, not {type(mod_args["reverse_tokens"])}.'
        assert isinstance(mod_args['remove_unks'], bool), f'remove_unks is expected to be a boolean value, not {type(mod_args["remove_unks"])}'
        assert isinstance(mod_args['sequence_length'], int), f'sequence_length was expected to be an integer, not {type(mod_args["sequence_length"])}.'
        assert mod_args['sequence_length'] > 0, f'sequence_length should be > 0, not {mod_args["sequence_length"]}.'

    def concat_text(self, df, data_fields, recordbreak):
        '''
        concatenates the relevant fields of the database

        Parameters:
            df (pd.DataFrame): data that will be concatenated
            data_fields (list): fields to be concatenated
            recordbreak (boolean): will the concatenated fields use a token to signify the split?
        Return:
            pd.Series: list of concatenated fields
        '''
        concat_text = df[data_fields[0]]
        #### this next if was included due to an error where nulls would cause an incorrect token
        #### this doesn't happen for single reports so the case of a single string is tested for
        if not isinstance(concat_text, str):
            concat_text = concat_text.fillna('')
        # concat the fields
        for f in data_fields[1:]:
            if recordbreak:
                concat_text = concat_text + ' recordbreaktoken '
            else:
                concat_text = concat_text + ' '
            if type(df) == dict:
                concat_text = (concat_text + df[f]).strip()
            else:
                concat_text = (concat_text + df[f].fillna('')).apply(lambda v: str(v).strip())

        return concat_text

    def concat_reports(self, df, data_fields, concat_split):
        '''
        concatenates the relevant fields of the database

        Parameters:
            df (pd.DataFrame): data that will be concatenated
            data_fields (list): fields to be concatenated
            concat_split (boolean): will the records be concatenated by the split type?
        Return:
            pd.Series: list of concatenated fields
        '''
        data_fields = ['textPathClinicalHistory','textStagingParams','textPathMicroscopicDesc',
                        'textPathNatureOfSpecimens','textPathSuppReportsAddenda','textPathFormalDx',
                        'textPathFullText','textPathComments','textPathGrossPathology']

        if concat_split and len(df)>0:
            import pandas as pd
            groups = df.groupby(df.index)
            cases = pd.DataFrame(groups['concat_data_field'].apply(lambda x: ' '.join(x)))
            for col in df.columns:
                if col not in data_fields and col != 'concat_data_field':
                    cases[col] = groups[col].apply(lambda x: x[0])
            cases.index = pd.MultiIndex.from_arrays((
                        [v[0] for v in cases.index.values],
                        [v[1] for v in cases.index.values],
                        [v[2] for v in cases.index.values]),
                        names=('patientId','registryId','tumorId'))
            return cases
        return df

    def sequence_text(self,s,reverse_tokens, remove_unks, word2tok, vocab, sequence_length):
        '''
        will translate a sentence to a list of indexes

        Parameters:
            s (text): a string of characters to be parsed
            reverse_tokens (boolean): will the tokens be reversed?
            remove_unks (boolean): will the unk_terms be removed completely?
            word2tok (dict): map from the words to the indexes
            vocab (list): expected terms
            sequence_length (int): length of the largest list
        '''
        unk_int = len(vocab) - 1
        # reverse the tokens if requested
        if reverse_tokens:
            s = list(reversed(s.split(' ')))
        else:
            s = s.split(' ')
        # remove unk terms
        if remove_unks:
            s = [word2tok[w] for w in s if w in word2tok.keys()]
        else:
            s = [word2tok[w] if w in word2tok.keys() else unk_int for w in s]
        # reverse tokens if needed and limit the length
        return s[:sequence_length]

    def generate_data(self,logging, records, index, truth_fields, metadata_fields, recordbreak, data_fields, concat_split, fold, fold_num, vocab, reverse_tokens, remove_unks, sequence_length):
        '''
        generate the tokenized inputs, metadata, and truth values for each record of the splits

        Parameters:
            logging (function): will return the print statement
            records (pd.DataFrame): lists of words representing each record
            index (list): list of columns that will be used as a unique index
            truth_fields (list): list of all the tasks associated with the records
            metadata_fields (list): list of all the names of the metadata desired
            data_fields (list): list of all the names of the data fields
            fold (list): list of dataframes that include the train, val, and test split ids
            fold_num (int): the fold number being processed
            vocab (list): list of words that creates the vocab
            reverse_tokens (boolean): will the tokens be read in a reverse order?
            remove_unks (boolean): will the unknown tokens be removed?
            sequence_length (int): the limit of tokens included for each report

        Modules:
            sanitize_record: sequences a string of text
        '''
        
        word2tok = {v:k for k,v in vocab.items()}
        import pandas as pd, numpy as np
        def applyfield(d, df):
            d['seq'] = df.apply(lambda v: self.sequence_text(v, reverse_tokens, remove_unks, word2tok, vocab, sequence_length))
            logging(str(list(d['seq'])))

        splits = ['train','val','test']
        idx = list(fold[0].columns)
        
        # loop through the train,val,test splits
        for i in range(3):
            rec_split = records.set_index(idx).join(fold[i].set_index(idx), how='inner').sort_values(index)
            # convert each model in chunks
            rec_split['concat_data_field'] = self.concat_text(rec_split, data_fields, recordbreak)
            rec_split = self.concat_reports(rec_split, data_fields, concat_split)
            if rec_split.shape[0] > 10:
                # use multiprocessing for incresed speed
                from multiprocessing import Process, Manager
                seqtext = []
                for j in range(0, len(rec_split), 100000):
                    manager = Manager()
                    seqtext_temp = manager.dict()
                    p = Process(target=applyfield, args=(seqtext_temp, rec_split['concat_data_field'][j:j+100000]))
                    p.start()
                    p.join()
                    seqtext += list(seqtext_temp['seq'])
            else:
                seqtext = rec_split['concat_data_field'].apply(lambda v: self.sequence_text(v, reverse_tokens, remove_unks, word2tok, vocab, sequence_length))
            logging('name: ' + splits[i]+'X_fold'+str(fold_num))
            # save metadata and truth fields
            self.pickle_files[f'{splits[i]}Y_fold{fold_num}'] = rec_split.reset_index()[sorted(set(truth_fields+index))].set_index(index)
            self.pickle_files[f'{splits[i]}Metadata_fold{fold_num}'] = rec_split.reset_index()[sorted(set(metadata_fields+index))].set_index(index)
            self.pickle_files[f'{splits[i]}X_fold{fold_num}'] = np.array(seqtext, dtype=object)

    def describe(self,describe, mod_args):
        '''
        a description of what what built during the pipeline build

        Parameters:
            describe (text): description from previous modules
            mod_args (dict): identify the arguments passed
        '''
        lines = []
        lines.append(describe)
        lines.append('\nEncoder:')
        lines.append(f'the {"last" if mod_args["reverse_tokens"] else "first"} {mod_args["sequence_length"]} tokens will be evaluated.{ "Words unkown to the vocabulary were ignored" if mod_args["remove_unks"] else ""}')
        self.text_files['describe'] = '\n'.join(lines)
