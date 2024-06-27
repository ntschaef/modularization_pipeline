import numpy as np

class Data_Generator():
    '''
    This module will capture necessary lists and dictionaries needed for the model.  This is currently 
    limited to the vocabulary, the tasks and expected predictions based on rule based alterations to the data.
    '''
    def __init__(self, cc, vocab_source="train", concat_split=False, use_security=False, use_w2v=True, min_label_count=1, w2v_kwargs={"window":5, "workers":5, "iter": 5, "min_count": 10, "size": 300}, np_seed=-1, add_tasks=[], add_labels={}, **kwargs):
        '''
        Initializer of the Data_Generator class.  Defines the "attributes".

        Parameters:
            cc (class): the CachedClass will be passed through all the modules and keep track of the pipeline arguements
            vocab_source (string): indication of where the w2v data is coming from.  Options are None in which case the train split will be used to generate data and randomize the input layer, "raw" indicating the cleaned (unfiltered) records, "train_val_test" indicating indicating that all splits will be used or "train" will utilize only the train split.
            concat_split (boolean): this will combined the records together based on how they were split (split_type) //THIS IS BROKEN CURRENTLY
            use_security (boolean): indication that the vocabulary will be restricted to a publically availible set of words. 
            use_w2v (boolean): indication of whether to use the word2vec module (the alternative is randomizing the input layer).
            min_label_count (int): the amount of occurances a label must appear within all the reports to be captured in the labels list. 1 means capture all of them.
            w2v_kwargs (dict): various arguments for w2v. These can include "size", "min_count", "window", "workers", "iter", and anything else that may be needed. Note: the "size" and "min_count" will be used regardless of if Word2Vec is used.
            np_seed (int): sets the random seed for numpy. -1 will keep it random
            add_tasks (list): list of additional tasks to include. Additional labels are expected if nonempty
            add_labels (dict): additional labels that will be added onto the generated tasks.
        '''
        assert cc.check_args('Data_Generator', locals()), 'mismatch between expected args and what is being used'
        self.pickle_files = {}
        self.text_files = {}
        cc.saved_mod_args['vocab_source'] = (vocab_source, 'Data_Generator', 'indication of where the w2v data is coming from.  Options are None in which case the train split will be used to generate data and randomize the input layer, "full" indicating the cleaned (unfiltered) records, "all" indicating indicating that all splits will be used, "test" will utilize the train + test splits, "val" is the train + validation splits, and "train" will utilize only the train split.')
        cc.saved_mod_args['concat_split'] = (concat_split, 'Data_Generator', 'This will combined the records together based on how they were split (split_type). //THIS IS BROKEN CURRENTLY')
        cc.saved_mod_args['use_security'] = (use_security, 'Data_Generator', 'indication that the vocabulary will be restricted to a publically availible set of words.')
        cc.saved_mod_args['use_w2v'] = (use_w2v, 'Data_Generator', 'indication of whether to use the word2vec module (the alternative is randomizing the input layer).')
        cc.saved_mod_args['min_label_count'] = (min_label_count, 'Data_Generator', 'the amount of occurances a label must must appear within the train split to be captured in the vocab list. 1 will capture all of them')
        cc.saved_mod_args['w2v_kwargs'] = (w2v_kwargs, 'Data_Generator', 'various arguments for w2v. These can include "size", "min_count", "window", "workers", "iter", and anything else that may be needed. NOTE: "size" and "min_count" will be used regardless of if Word2Vec is used.')
        cc.saved_mod_args['np_seed'] = (np_seed, 'Data_Generator', 'sets the random seed for numpy. -1 will keep it random')
        cc.saved_mod_args['add_tasks'] = (add_tasks, 'Data_Generator', 'list of additional tasks to include. Additional labels are expected if nonempty')
        cc.saved_mod_args['add_labels'] = (add_labels, 'Data_Generator', 'additional labels that will be added onto the generated tasks.')
        if int(np_seed)>=0:
            np.random.seed(int(np_seed))

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
            with cc.timing("extracting data for data_generator"): 
                describe, schema, records, record_iter, miss_tasks, folds = self.get_cached_deps(cc)
                mt = [v for v in miss_tasks.columns if v not in records.columns]
                for c in mt:
                    records[c] = miss_tasks[c]
                self.check_args(cc.logging.debug, mod_args)
                data_fields = schema[schema['tables'][0]]['data_fields']
                vocab_size = []
                labels = []
            for i, fold in enumerate(folds):
                with cc.timing(f'building vocab for fold {i}'):
                    vocab_size.append(self.build_vocab(cc.logging.debug, record_iter, mod_args['vocab_source'], fold, i, data_fields, mod_args['remove_breaktoken']!='all', mod_args['concat_split'], 
                                      mod_args['use_w2v'], mod_args['use_security'], mod_args['np_seed'], mod_args['w2v_kwargs']))
                with cc.timing(f'building labels for fold {i}'):
                    labels.append(self.build_labels(cc.logging.debug, records, fold[0], i, mod_args['tasks'], mod_args['min_label_count'], mod_args['add_tasks'], mod_args['add_labels'], miss_tasks))
            self.describe(describe, data_fields, mod_args, vocab_size, labels)
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

        Returns:
            data (tuple): collection of the different cached data that was recieved:
                          describe (string), schema (dict), records (pd.DataFrame), folds (list)
        '''
        fold_num = int(cc.get_cache_file('Splitter', 'fold_num'))
        
        data = (
             cc.get_cache_file('Splitter', 'describe'),
             cc.get_cache_file('Parser', 'db_schema'),
             cc.get_cache_file('Sanitizer', 'df_tokens'),
             cc.ittr_dataset(cc, 'Sanitizer', 'df_tokens'),
             cc.get_cache_file('Filter', 'missing_tasks'), 
             [[
                 cc.get_cache_file('Splitter',f'train_fold{i}'),
                 cc.get_cache_file('Splitter',f'val_fold{i}'),
                 cc.get_cache_file('Splitter',f'test_fold{i}')] for i in range(fold_num)])
        return data

    def check_args(self, logging, mod_args):
        '''
        This will verify the arguments that have been provided to make sure are expected.

        Parameters:
            logging (function): will return the print statement
            mod_args (dict): the saved model args to identify how to filter the text as well as everything else
        '''
        assert (isinstance(mod_args['np_seed'], int) and (mod_args['np_seed'] >= -1)), f'"np_seed was expected to be an int >=-1.  Got {mod_args["np_seed"]}'
        assert mod_args['vocab_source'] in ['raw','train_val_test', 'train'], f'{mod_args["vocab_source"]} is unexpected as a "vocab_source". Expected entries are "raw", "train_val_test", and "train".'
        assert mod_args['vocab_source'] == 'train' or mod_args['use_w2v'], f'"use_w2v" cannot be false while "vocab_source" is not "train".  Got {mod_args["vocab_source"]} instead.'
        assert (isinstance(mod_args['w2v_kwargs']['min_count'], int) and (mod_args['w2v_kwargs']['min_count'] >= 1)), f'min_count is expected to be a non-negative integer, not {mod_args["w2v_kwargs"]["min_count"]}'
        assert (isinstance(mod_args['min_label_count'], int) and (mod_args['min_label_count'] >= 1)), f'min_label_count is expected to be a non-negative integer, not {mod_args["min_label_count"]}'
        if len(mod_args['add_tasks']) > 0:
            assert len(sorted(set(mod_args['add_tasks']+mod_args['tasks']))) == len(mod_args['add_tasks'] + mod_args['tasks']), 'duplicate tasks were created with "add_tasks"'
            new_tasks = sorted(set([v for v in mod_args['add_tasks'] if v in mod_args['add_labels'].keys()]))
            assert len(new_tasks) == len(mod_args['add_tasks']), 'no new labels were found for the new tasks'
            for k in new_tasks:
                 assert len(mod_args['add_labels'][k]) > 0, 'no labels were found for the new tasks'

    def run_w2v(self, logging, records, fold_num, np_seed, w2v_kwargs):
        '''
        train a word2vec model which will generate an input layer and a vocab.

        Parameters:
            logging (function): will return the print statement
            records (pd.DataFrame): lists of words representing each record
            fold_num (int): fold that is currnetly being run
            w2v_kwargs (dict): various arguments for w2v. These can include "size", "window", "min_count", "workers", "iter", and anything else that may be needed.

        Return:
            int: number of terms in the vocab
        '''
        
        assert np_seed==-1, "random seed cannot be set for w2v yet.  Please set np_seed=-1 if you are going to set use_w2v=True" #TODO: ensure random seed can be set until then assert that it is random
        from gensim.models import Word2Vec
        mod = Word2Vec(records, **w2v_kwargs)
        assert '<unk>' not in mod.wv.index2word, f'"<unk>" found as a word in the vocab for fold {fold_num}.  This will cause an error later'
        assert '<pad>' not in mod.wv.index2word, f'"<pad>" found as a word in the vocab for fold {fold_num}.  This will cause an error later'



        ordered_vectors = []
        ordered_words = sorted(mod.wv.index2word)
        for w in ordered_words:
            ordered_vectors.append(mod.wv.vectors[mod.wv.vocab[w].index])
 
        self.pickle_files[f'id2word_fold{fold_num}'] = {i:v for i,v in enumerate(['<pad>'] + ordered_words + ['<unk>'])}
        self.pickle_files[f'word_embeds_fold{fold_num}'] = np.append(np.append(np.zeros((1,w2v_kwargs['size'])),ordered_vectors, axis=0),(np.random.randn(1,w2v_kwargs['size'])*.01), axis=0)
 
        return len(mod.wv.index2word)

    def randomize_embeddings(self, logging, records, fold_num, min_count, input_size, np_seed):
        '''
        Create the input layer through randomization.

        Only the train split will be utilized an passed through.

        Parameters:
            logging (function): will return the print statement
            records (pd.DataFrame): lists of words representing each record
            fold_num (int): fold that is currnetly being run
            min_count (int): the minimum times a word can appear before being captured in the vocab
            input_size (int): the layer size that the vocab will feed into.

        Return:
            int: number of terms in the vocab
        '''
        word_count = {}
        for rec in records:
            for w in rec:
                if w in word_count.keys():
                    word_count[w] += 1
                else:
                    word_count[w] = 1
        vocab = [w for w,v in word_count.items() if v >= min_count]
        assert '<unk>' not in vocab, f'"<unk>" found as a word in the vocab for fold {fold_num}.  This will cause an error later'
        assert '<pad>' not in vocab, f'"<pad>" found as a word in the vocab for fold {fold_num}.  This will cause an error later'

        self.pickle_files[f'id2word_fold{fold_num}'] = {i:v for i,v in enumerate(['<pad>'] + sorted(vocab) + ['<unk>'])}

        vocab_len = len(vocab)
        if np_seed >= 0:
            np.random.seed(np_seed)

        self.pickle_files[f'word_embeds_fold{fold_num}'] = np.append(np.zeros((1,input_size)),np.random.randn(vocab_len + 1, input_size)*0.01, axis=0)
        return vocab_len
        

    def build_vocab(self, logging, records, vocab_source, fold, fold_num, data_fields, recordbreak, concat_split, use_w2v, use_security, np_seed, w2v_kwargs):
        '''
        the main call for generating the vocab list, vocab mappings, and input layer generation.

        Note: "<pad>" will be mapped to the first index associated with a 0 vector. "unk" (used for unknown words) will be mapped the last index with a random vector.

        Parameters:
            logging (function): will return the print statement
            records (pd.DataFrame): tokenized data
            vocab_source (string): indication of the portion of reports that the vocab comes from.
            fold (list): list of dataframes that include the train, val, and test split ids
            fold_num (int): the fold number being processed
            data_fields (list): the fields to be searched for the vocab
            recordbreak (Boolean): adds "recordbreaktoken" between the fields
            use_w2v (Boolean): indication of whether to use word2vec (True) or randomized initiation (False)
            use_security (Boolean): indiction of whether to reduce down vocabulary to a pregenerated dataset.
            np_seed(int): the initial seed for the generation of the inital layer.
            w2v_kwargs (dict): various arguments for w2v. These can include "size", "window", "min_count", "workers", "iter", and anything else that may be needed.

        Modules:
            run_w2v: will return the count of vocab terms after training a w2v model.
            randomize_embeddings: will return the count of vocab terms and randomize the input layer.

        Return:
            int: number of terms in the vocab list
        '''
        def inner_function(records, logging=logging, vocab_source=vocab_source, fold=fold, fold_num=fold_num, data_fields=data_fields, 
                           use_w2v=use_w2v, use_security=use_security, np_seed=np_seed, w2v_kwargs=w2v_kwargs):
            # restrict the records to only the subset considered
            if vocab_source != 'raw':
                if vocab_source == 'train':
                    split = fold[0]
                else:
                    split = fold[0].append(fold[1].append(fold[2]))
                logging('data set size: ' + str(len(split)))
                records = records.set_index(list(split.columns)).join(split.set_index(list(split.columns)), how='inner')
            logging('records columns: ' + str(records.columns))
            # create a list of words the reference each report
            import pandas as pd
            pd.options.mode.chained_assignment = None
            for i,c in enumerate(data_fields):
                if i == 0:
                    records['concat_data_field'] = records[c].fillna("")
                elif recordbreak:
                    records['concat_data_field'] = (records['concat_data_field'] + ' recordbreaktoken ' + records[c].fillna("")).apply(lambda v: str(v).strip())
                else:
                    records['concat_data_field'] = (records['concat_data_field'] + ' ' + records[c].fillna("")).apply(lambda v: str(v).strip())

            if concat_split: #THIS IS CURRENTLY BROKEN AND NEEDS TO BE IMPLEMENTED

                groups = records.groupby(records.index)
                cases = pd.DataFrame(groups['concat_data_field'].apply(lambda x: ' '.join(x)))
                for col in records.columns:
                    if col not in data_fields and col != 'concat_data_field':
                        cases[col] = groups[col].apply(lambda x: x[0])
                cases.index = pd.MultiIndex.from_arrays((
                        [v[0] for v in cases.index.values],
                        [v[1] for v in cases.index.values],
                        [v[2] for v in cases.index.values]),
                        names=('patientId','registryId','tumorId'))
                records = cases

            pd.options.mode.chained_assignment = 'warn'
            logging('record type: ' + str(type(records)))
            if use_security:
                import pickle
                with open('BioWordVec.pickle', 'rb') as bwv:
                    sec_list = pickle.load(bwv)
                records['concat_data_field'] = records['concat_data_field'].apply(lambda v: [v1 for v1 in v.split(' ') if v1 in sec_list])
            else:
                records['concat_data_field'] = records['concat_data_field'].apply(lambda v: v.split(' '))
            return records[['concat_data_field']]
        records.add_func(inner_function)
        if use_w2v:
            return self.run_w2v(logging, records, fold_num, np_seed, w2v_kwargs)
        else:
            return self.randomize_embeddings(logging, records, fold_num, w2v_kwargs['min_count'], w2v_kwargs['size'], np_seed)
        
    def build_labels(self, logging, records, train, fold_num, token_fields, min_label_count, add_tasks, add_labels, add_missing_tasks):
        '''
        the main call for generating the label list(s) and label mappings.

        Parameters:
            logging (function): will return the print statement
            records (pd.DataFrame): tokenized data
            folds (list): list of dataframes that include the train, val, and test split ids
            token_fields (list): single or combined fields in which the tokens will be captured.
            min_label_count (int): the minimum count that has to be achieved for the labels to be captured.
            add_tasks (list): list of additional tasks to include. Additional labels are expected if nonempty
            add_labels (dict): additional labels that will be added onto the generated tasks.
            add_missing_tasks (list): list of additional new tasks to include

        Return:
            dict: tasks and associated labels.
        '''
        # added missing tasks to data
        miss_tasks = [v for v in token_fields if v not in records.columns]
        records[miss_tasks] = add_missing_tasks[miss_tasks]
        # reduce records to only the train set
        records = records.set_index(list(train.columns)).join(train.set_index(list(train.columns)), how = 'inner')
        # get token lists
        uni_toks = sorted(set([t for tokens in token_fields for t in tokens.split('+')]))
        records = records[uni_toks]
        # combined tokens
        for ts in token_fields:
            if ts not in records.columns:
                for i,t in enumerate(ts.split('+')):
                    if i == 0:
                        records[ts] = records[t]
                    else:
                        records[ts] += ' ' + records[t]
        # reduce and order
        tasks = {t:{i:l for i,l in enumerate(
                  sorted([v for v in records[t].drop_duplicates() 
                  if len(records[records[t] == v]) >= min_label_count]))} 
                      for t in token_fields}
        # add in additional tasks and labels
        for t in add_tasks:
            tasks[t] = {}
        for t in tasks.keys():
            if t in add_labels.keys():
                for l in add_labels[t]:
                    tasks[t][len(tasks[t].keys())] = l 
        
        # record return expected data
        self.text_files[f'id2labels_fold{fold_num}'] = tasks
        return tasks 


    def describe(self, describe, data_fields, mod_args, vocab_size, labels):
        '''
        generates a description of what what built during the pipeline build

        Parameters:
            describe (text): description from previous modules
            schema (dict): identifies the fields that are being used in the dataframe
            mod_args (dict): identify the arguments passed
            vocab_size (int): total number of vocab terms
            labels (dict): tasks and associated labels
        '''
        lines = []
        lines.append(describe)
        lines.append('\nData_Generator:')
        n = len(vocab_size)
        lines.append(f'using the fields')
        for f in data_fields:
            lines.append(f'   {f}')
        lines.append(f'vocabularies were created for the {n} folds{" by capturing all terms that occurred more than " + str(mod_args["w2v_kwargs"]["min_count"]) + " times" if mod_args["w2v_kwargs"]["min_count"]>0 else ""}. This was obtained from the {"unfiltered" if mod_args["vocab_source"]=="raw" else "train split" if mod_args["vocab_source"] == "train" else "filtered"} dataset.  Counts for each fold are as follows:')
        for i in range(n):
            lines.append(f"    fold index {i}: {vocab_size[i]}")
        lines.append(f'An input layer was also created using a {"Word2Vec model" if mod_args["use_w2v"] else "randomization"} mapping to a layer size of {mod_args["w2v_kwargs"]["size"]}. Two additional tokens, "<pad>" and "<unk>" were added to the input layer with a zero mapping and a random mapping at the beginning and end respectively.{"  The seed for the randomization was " + str(mod_args["np_seed"]) if mod_args["np_seed"]>=0 else ""}\n')
        lines.append(f'tokens were also captured from the training dataset {"as long as they occurred at least " + str(mod_args["min_label_count"]) + " times " if mod_args["min_label_count"]>0 else ""}and had the following count for each fold:')
        for i in range(n):
            lines.append(f'    for fold of index {i}:')
            for k,v in labels[i].items():
                lines.append(f'        {k} has {len(v)} options of the form {v[0]}')
        if len(mod_args['add_tasks'])>0:
            lines.append(f'the task{"s" if len(mod_args["add_tasks"])>1 else ""} {", ".join(mod_args["add_tasks"])} were added manually.')
        for k,v in mod_args['add_labels'].items():
            lines.append(f'    the value{"s" if len(v)>0 else ""} {", ".join(v)} were added to task {k}')
        self.text_files['describe'] = '\n'.join(lines)
