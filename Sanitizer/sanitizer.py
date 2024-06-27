import sqlite3, pandas as pd

class IDMap:
    '''call the stored token, if it doesn't exist create it.'''
    def __init__(self):
        self.map = {}
        self.next_id = len(self.map)
    def get(self, in_id):
        try:
            return self.map[in_id]
        except:
            self.next_id += 1
            self.map[in_id] = self.next_id
            return self.next_id

class TokenMap:
    '''Create the key mapping for the tokenized data'''
    def __init__(self):
        self.id = IDMap()
        self.word = IDMap()
    def invert(self):
        def invert_dict(d):
            return dict((v,k) for k,v in d.items())
        return {'id':invert_dict(self.id),
                'word':invert_dict(self.word)}

class Sanitizer():
    '''
    Ensuring that no PHI is presented in the final database (for exporting datasets outside of a secure environment).

    Attributes:
        cc (class): the CachedClass will be passed through all the modules
                    and keep track of the pipeline arguements.

        deidentify (boolean): identifies if the module will be used or just skipped
    '''
    def __init__(self, cc, remove_dups=False, min_doc_length=20, deidentify=False, **kwargs):
        '''
        Initializer of the Sanatizer.  Defines the "attributes".

        Parameters:
            cc (class): the CachedClass will be passed through all the modules and keep track of the pipeline arguements.
            remove_dups (boolean): should duplicate ids be removed? "False" will stop the run if duplicate ids are found
            min_doc_length (int): indicates the smallest amount of characters that will be allowed after whitespece is removed. 
            deidentify (boolean): identifies if the module will be used or just skipped
        '''
        assert cc.check_args('Sanitizer',locals()), 'mismatch between expected args and what is being used'
        self.pickle_files = {}
        self.text_files={}
        cc.saved_mod_args['remove_dups'] = (remove_dups, 'Sanitizer', 'should duplicate unique record identifiers be removed? "False" will stop the run if duplicate ids are found')
        cc.saved_mod_args['min_doc_length'] = (min_doc_length, 'Sanitizer', 'indicates the smallest amount of characters that will be allowed after whitespece is removed.') 
        cc.saved_mod_args['deidentify'] = (deidentify, 'Sanitizer', 'set to true to deidentify tokens for summit export.  Otherwise set to false.')
        self.token_map = TokenMap()

    def build_dataset(self, cc):
        '''
        this is the function used to continue the build of the model within the pipeline

        Parameters:
            cc (class): the CachedClass will be passed through all the modules
                        and keep track of the pipeline arguements.
        Modules:
            build_query (string): this will construct the queries needed to refine the data to the necessary fields
            import_database (pd.DataFrame): this will dump the fields of the database into a dataframe
        '''
        mod_name = self.__module__.split('.')[-2]
        mod_args = cc.all_mod_args()
        if not cc.test_cache(mod_name=mod_name, del_dirty=cc.remove_dirty, mod_args=mod_args):
            with cc.timing("extracting data for sanitizer"):
                data, schema, describe = self.get_cached_deps(cc)
                data, order_ids, small_count, dup_count = self.check_data(data,schema, mod_args['min_doc_length'], mod_args['remove_dups'])
            if mod_args['deidentify']:
                # continue mapping the deidentification
                with cc.timing("Creating tokens and refining database"):
                    self.make_token_dump(cc.logging.debug, cc.deps, cc.saved_packages, data.sort_values(by=order_ids), schema)
            else:
                # skip everything and save the cleaned data
                cc.logging.debug('tokens were not used')
                data.reset_index(drop=True, inplace=True)
                self.pickle_files['df_tokens'] = data
            self.describe(describe, schema, mod_args, small_count, dup_count)
            cc.build_cache(mod_name=mod_name,
                           pick_files=self.pickle_files,
                           text_files=self.text_files,
                           mod_args=mod_args)

    def predict_single(self, data, stored, args, filters):
        '''
        will sanitize the text of data passed through

        Parameters:
            data (dict): the raw data
            stored (dict): data that is needed to predict the record
            args (dict): arguments passed through
            filters (boolean): includes additional filter fields, Default: False
        Return:
            dict: the sanitized data
        '''
        if args['deidentify']=='True':
            t_fields = stored['schema'][stored['schema']['tables'][0]]['data_fields']
            for c in t_fields:
                if data[c] == "":
                    text = ""
                else:
                    text = ' '.join([str(stored['word_map'][v]) for v in data[c].split(' ') 
                                      if v in stored['word_map'].keys()])
                data[f] = text
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
        if args['deidentify']=='True':
            t_fields = stored['schema'][stored['schema']['tables'][0]]['data_fields']
            for c in t_fields:
                data[c] = self.tokenize_text(list(dfclean[c]))
        return data
    def get_cached_deps(self,cc):
        '''
        This function pulls the prior cached data.
        
        Parameters:
            cc (class): the CachedClass will identify the previous caches.
        Return:
            data (tuple): collection of the different cached data that was recieved:
                          df_cleaned (pd.DataFrame), schema (dict), describe (string)
        '''
        data = (
             cc.get_cache_file('Cleaner', 'df_cleaned'),
             cc.get_cache_file('Parser', 'db_schema'),
             cc.get_cache_file('Cleaner', 'describe'))
        return data

    def check_data(self, data, schema, min_doc_length, remove_dups):
        '''
        Will make sure that the ordering is assured

        Parameters:
            cc (class): the CachedClass will identify the previous caches.
            data (pd.DataFrame): data to be checked
            schema (dict): schema that is being used
            remove_dups (boolean): should the duplcate ordering be dropped? (the alternative is to stop the run)

        return:
            pd.DataFrame: data that fits the criteria
            list: order id fields that will be used to sort the files
            int: number of records dropped due to size
            int: number of records dropped due to duplicate ids
        '''
        text_fields = schema[schema['tables'][0]]['data_fields']
        small_count = len(data)
        concat_fields = data[text_fields[0]].fillna('')
        for c in text_fields[1:]:
            concat_fields = (concat_fields + data[c].fillna(''))
        data = data[concat_fields.apply(lambda v: len(''.join(v.split())) >= min_doc_length)]
        small_count -= len(data)
        order_ids = sorted(set([v.split(' ')[-1] for v in schema[schema['tables'][0]]['order_id_fields'] + schema[schema['tables'][0]]['order_id_fields']]))
        dup_count = len(data)
        if remove_dups:
            data = data.drop_duplicates(subset=order_ids)
        dup_count -= len(data)
        assert len(data[order_ids].drop_duplicates())==len(data), f'cannot sanitize this dataset because the following repos have duplicate order_ids: {",".join([": ".join([reg, str(len(data[data["registryId"] == reg]) - len(data[data["registryId"] == reg][order_ids].drop_duplicates()))]) for reg in data["registryId"].unique()])}'
        return data, order_ids, small_count, dup_count

    def tokenize_text(self, s_text):
        '''
        This function will convert words to intergers with sentences and keep a running map from words to intergers. 

        Parameters:
            s_text (list): the cleaned list of sentences

        Return:
            list: list of tokenized sentences  
        '''
        sids = []
        for sent in s_text:
            if sent is None:
                sent = ''
            wids = []
            for w in sent.split(' '):
                if len(w) == 0: 
                    wids.append('')
                else:
                    wids.append(str(self.token_map.word.get(w)))
            sids.append(' '.join(wids))
        return sids

    def tokenize_id(self, id_text):
        '''
        This function will convert identifing keys to intergers with sentences and keep a running map from words to intergers. 

        Parameters:
            id_text (list): the ids expecting to be tokenized

        Return:
            list: list of tokenized ids
        '''
        iids = []
        for ids in id_text:
            iids.append(self.token_map.id.get(str(ids)))
        return iids

    def make_token_dump(self, logging, deps, packs, dfclean, schema):
        '''
        This function will create a tokenized database dump text. 

        Parameters:
            logging (function): will return the print statements
            deps (dict): the dependencies expected for the naming of the package information
            packs (dict): the current saved state of the packages to indicate what is necessary to create the tokenization
            dfclean (pd.DataFrame): the cleaned dataframe expecting to be tokenized
            schema (dict): the schema of the data source, which has the field names

        Modules:
            tokenize_text: This function will convert words to intergers with sentences and keep a running map from words to intergers. 
            tokenize_id: This function will convert identifing keys to intergers with sentences and keep a running map from words to intergers. 

        Classes:
            TokenMap: an expanding list that will capture associations.
        '''
        t_fields = [v.split(' ')[-1] for v in schema[schema['tables'][0]]['data_fields']]
        id_fields = sorted(set([v.split(' ')[-1] for v in 
                             schema[schema['tables'][0]]['id_fields'] + 
                             schema[schema['tables'][0]]['order_id_fields'] +
                             schema[schema['tables'][1]]['id_fields'] +
                             schema[schema['tables'][1]]['order_id_fields']]))
        logging(f't_fileds: {t_fields}')
        logging(f'id_fields: {id_fields}')
        df_tok = dfclean.copy()

        for c in t_fields:
            df_tok[c] = self.tokenize_text(list(dfclean[c]))
        for c in id_fields:
            df_tok[c] = self.tokenize_id(list(dfclean[c]))
        self.pickle_files['df_tokens'] = df_tok
        self.text_files['word_map'] = self.token_map.word.map
        self.text_files['id_map'] = self.token_map.id.map
        
        # create metadata table
        prov_list = ['sub_mod','environ_vars','mod_args','packages']
        prov = {k:{} for k in prov_list}
        mod_list = [k for k,v in packs.items() if v!='']
        for m in mod_list:
            for i,k in enumerate(prov_list):
                new_prov = {k1:v1 for k1,v1 in zip(deps[m][i], packs[m][i]) if k1 not in prov[k]}
                prov[k] = {**prov[k],**new_prov}
        prov['commit'] = {k: packs[k][-1] for k in mod_list}
        self.dfmetadata = pd.DataFrame({k:{'values':str(v)} for k,v in prov.items()})

        # create database dump
        import os
        if 'TOKEN_SAVE' in os.environ:
            assert os.path.exists(os.environ['TOKEN_SAVE']), f'the path {os.environ["TOKEN_SAVE"]} does not exist'
            conn = sqlite3.connect(os.path.join(os.environ['TOKEN_SAVE'], 'token_db.sqlite'))
        else:
            conn = sqlite3.connect(':memory:')
        self.dfmetadata.to_sql('metadata',conn)
        id_names = [v.split(' ')[-1] for v in schema[schema['tables'][0]]['id_fields']] 
        df_tok.set_index(id_names).to_sql(f'{"_".join(schema["tables"])}', conn)
        dumptext = ''
        for line in conn.iterdump():
            dumptext += f'{line}\n'
        self.text_files['token_dump'] = dumptext


    def describe(self, describe, schema, mod_args, small_count, dup_count):
        '''
        generates a description of what what built during the pipeline build

        Parameters:
            deps (dict):       the dependencies expected for the naming of the package information
            describe (text):   description from previous modules
            schema (dict):     the schema of the data source, which has the field names
            mod_args (dict):   identify the arguements passed through.
            small_count (int): the number of records dropped because they were too small
            dup_count (int):   the number of records dropped because there were duplicate records
        '''
        clean_run = True
        lines = []
        lines.append(describe)
        lines.append('\nSANITIZER:')
        if small_count > 0:
            lines.append(f'there were {small_count} dropped because they had fewer than {mod_args["min_doc_length"]} non-white space characters.')
        if dup_count > 0:
            lines.append(f'there were {dup_count} dropped because they had duplicated ordering ids')
            clean_run = False
        if mod_args['deidentify']:
            lines.append('The following fields of data was converted to numeric characters for deidentification:')
            lines.append(', '.join([v.split(' ')[-1] for v in schema[schema['tables'][0]]['data_fields']]))
            lines.append('The following fields of ids were serialized for deidentification:')
            lines.append(', '.join([v.split(' ')[-1] for v in 
                              schema[schema['tables'][0]]['id_fields'] +
                              schema[schema['tables'][0]]['order_id_fields'] +
                              schema[schema['tables'][1]]['id_fields'] +
                              schema[schema['tables'][1]]['order_id_fields']]))
            clean_run = False
        if clean_run:
            lines.append('cleaned data was unchanged')
        self.text_files['describe'] = '\n'.join(lines)
        self.text_files['describe'] = describe

if __name__=='__main__':
    import argparse, json, os

    # provide options
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # reqired text
    parser.add_argument('token', type=int, 
                        help='integer expecting to be identified')
    parser.add_argument('category', type=str, 
                        help='the category the token comes from. "id" is a case identifier, "word" is an element of sentences')
    parser.add_argument('cache_path', type=str, 
                        help='the location from which the maps will be pulled.')
    
    args = parser.parse_args()

    assert args.category in ['id','word'], 'expected either "id" or "word" as the category'

    mapper = f'{args.category}_map.json'
    print(os.path.join(args.cache_path,mapper))
    with open(os.path.join(args.cache_path,mapper),'r') as jd:
        token_map = json.load(jd)

    rev_tm = {str(v):k for k,v in token_map.items()}
    token = str(int(args.token))
    if token in rev_tm.keys():
        print(rev_tm[token])
    else:
        print('unk')
