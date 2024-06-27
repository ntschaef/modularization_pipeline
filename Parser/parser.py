import os,json, yaml
class Parser():
    '''
    This class will pull normalized data from the database of your choice and convert it into a refined dataframe.

    Attributes:
        db_path_dict (dict): the refernces to the pathways to the saved databases and which schema they will use.
    '''

    def __init__(self, cc, db_list = {"20220715":["KY","LA","NJ","SE","UT"]}, add_fields = {}, **kwargs):
        '''
        Initializer of the Parser. Defines the "attributes".

        Parameters:
            cc (class): the CachedClass will be passed through all the modules 
                        and keep track of the pipeline arguements.
            db_list (dict): identifies the original datasets that will be used for the pipeline. key: date of data, value: list of databases
            add_fields (dict): list of additional fields to add to the schema. key: table, value: list of fields
        '''
        assert cc.check_args("Parser",locals()), 'mismatch between expected args and what is being used'
        self.cur_path = os.path.dirname(os.path.abspath(__file__))

        self.get_paths()

        self.pickle_files = {}
        self.text_files = {}
        cc.saved_mod_args['db_list'] = (db_list, 'Parser', f'the list of all the datasets that will be joined to train the model. Current options for each date are {", ".join([k + ": " + "|".join([",".join(v1.keys()) + " with schema " + k1 for k1,v1 in v.items()]) for k,v in self.date_paths.items()])}. Anything with the "man_coded" schema is a report specific classifer (eg bucketing or reportability), the "info_extraction" schema is case level predictions, reportability is specific to reportability, and recurrence is specific to recurrence')
        cc.saved_mod_args['add_fields'] = (add_fields, 'Parser', 'dictionary of additional fields to add to the schema. The key of the dictionary is the table and the value is the field list that is desired - eg. {"man_coded": ["at_risk_any_flag"]}.')

    def get_paths(self):
        # pull in the parquet files
        with open('/mnt/nci/scratch/data/prod_data/data_warehouse/catalog.yaml', 'r') as y:
            cat_paths = yaml.safe_load(y)
        self.date_paths = {}
        # restructring the catelog
        for k,v in cat_paths.items():
            for t in v:
                for reg, v1 in v[t].items():
                    for p in v1:
                        d = p.split('/')[-2].split('_')[-1]
                        if d not in self.date_paths.keys():
                            self.date_paths[d] = {}
                        if k not in self.date_paths[d].keys():
                            self.date_paths[d][k] = {}
                        if reg not in self.date_paths[d][k].keys():
                           self.date_paths[d][k][reg] = {}
                        self.date_paths[d][k][reg][t] = p

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
            with cc.timing("extracting data for parser"):
                data = self.get_cached_deps() #including this step because it is expected and is test (TODO).
                db_list, schema = self.fill_path(mod_args['db_list'])
            with cc.timing("Building database query"):
                self.test_db(cc.logging.debug, db_list, schema, mod_args['add_fields'])
                self.build_query(cc.logging.debug, db_list, self.text_files['db_schema'])
            with cc.timing("Creating dataframe"):
                self.import_database(self.text_files['duckdb_queries'], self.text_files['db_schema'])
                self.adjust_schema()
                self.describe(cc,"", db_list)
            cc.build_cache(mod_name=mod_name, 
                           pick_files=self.pickle_files, 
                           text_files=self.text_files,
                           mod_args=mod_args)


    def predict_single(self, data, stored, args, logging):
        '''
        will create an expected schema from the data provided

        Parameters:
            data (dict): the raw data
            stored (dict): data that is needed to predict the record
            args (dict): arguments passed through
            logging (function): print fuction
        Return:
            dict: the converted data
        '''
        newline = "\n"
        e_schema = stored['schema'][stored['schema']['tables'][0]]
        data_fields = e_schema['data_fields']
        id_fields = [v.split(' ')[0] for v in e_schema['id_fields']] 
        assert [k for k in data_fields if k in data.keys() and data[k] != ""] != [], f'expected to find one of the folloiwng fields: \n    {(newline+"    ").join(data_fields)}'
        miss_id_fields = [k for k in id_fields if k not in data.keys()]
        assert miss_id_fields == [], f'missing necessary field(s):\n    {(newline+"    ").join(miss_id_fields)}'
        return {k:data[k] if k in data.keys() else "" for k in id_fields+data_fields}

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
#        import pandas as pd, duckdb
#        df_list = []
#        for db_path in stored['db_paths']:
#            df_list.append(duckdb.query(stored['query']).to_df())
#        return pd.concat(df_list)
        
    def get_cached_deps(self):
        '''
        This function pulls the prior cached data.

        For this module no prior data is needed.  So this function is a stand in for tests
        '''
        return None

    def fill_path(self, keys):
        '''
        complete the paths for the parquet files.

        Parameters:
            keys (list): the list of the keys needed for the databases

        Returns:
            list: the list of dictionaries for each parquet path
            schema: the key for the schema to be used
        '''
        path_list = []
        base_schema = ""
        for d,regs in keys.items():
            if d not in self.date_paths.keys():
                exit(f'the date {d} was not found in the data sets')
            if base_schema == "":
                base_schema = list(self.date_paths[d].keys())[0]
            elif base_schema != list(self.date_paths[d].keys())[0]:
                exit('schemas are not consistent')
            for r in regs:
                if r not in self.date_paths[d][base_schema].keys():
                    exit(f'{r} is not a registry in the {d} list')
                path_list.append({t: f"'{p}' as {t}" for t,p in self.date_paths[d][base_schema][r].items()})
        return path_list, base_schema


    def test_db(self, logger, dblist, schema, add_fields):
        '''
        tesing the database to ensure that user input is appropriate
        
        Parameters:
            logger (logging method): method of documentation from within the script.
            dblist (list): list of dictionaries for the table paths.
            schema (string): the key for the schema to be used.
            add_fields (dict): list of additional fields to add to the schema
        
        Variables:
            db_schemas (dict): contains the expected fields need for each schema.

        Returns:
            self.text_files: 
                schema (string): the fields needed later to build the query, also a variable to be cached.
                db_paths (list): the directory paths needed to build the query, also a variable to be cached.
        '''
        #pull the schema from the external json
        with open(os.path.join(self.cur_path,'schema.json'),'rb') as jd:
            db_schemas = json.load(jd)

        data_fields = {'epath':
                         ['text_path_clinical_history as textPathClinicalHistory',
                          'text_staging_params as textStagingParams',
                          'text_path_microscopic_desc as textPathMicroscopicDesc',
                          'text_path_nature_of_specimens as textPathNatureOfSpecimens',
                          'text_path_supp_reports_addenda as textPathSuppReportsAddenda',
                          'text_path_formal_dx as textPathFormalDx',
                          'text_path_full_text as textPathFullText',
                          'text_path_comments as textPathComments',
                          'text_path_gross_pathology as textPathGrossPathology']}

        data_schemas = {'reportability':{'epath':data_fields['epath']},
                        'info_extraction':{'epath':data_fields['epath']},
                        'recurrence':{'epath':data_fields['epath']}}

        assert sorted(data_schemas.keys()) == sorted(db_schemas.keys()), 'mismatch between data and the loaded schemas'
        db_schemas = {k:{
                        k1:v1 if k1 not in data_schemas[k].keys() else 
                        {**v1,'data_fields': data_schemas[k][k1]}
                              for k1,v1 in v.items()}
                                 for k,v in db_schemas.items()}

        #check to make sure that if the database is overwritten, the database exists and the schema is identified
        if 'DATA_DB' in os.environ:
            assert 'DATA_SCHEMA' in os.environ, f'with a custom database, you need to set a schema ({",".join(db_schemas.keys())}) with DATA_SCHEMA.'
            assert os.path.exists(os.environ['DATA_DB']), f'{os.environ["DATA_DB"]} does not exist.'
            self.text_files['db_paths'].append(os.environ['DATA_DB'])
            schema = os.environ['DATA_SCHEMA']
            # if not custom make sure the list of databases all share the same "schema"
# start here

        # make sure the schema is expected
        assert schema in db_schemas.keys(), f'unknown database schema {schema}'

        # add in the extra fields
        for table, fields in add_fields.items():
            if 'other_fields' not in db_schemas[schema][table].keys():
                db_schemas[schema][table]['other_fields'] = []
            db_schemas[schema][table]['other_fields'] += fields

        self.text_files['db_schema'] = db_schemas[schema]
        self.text_files['schema_name'] = schema
        logger('both schema and db_paths are saved and will be written to cache.\n')

    def build_query(self, logger, dblist, db_schema):
        '''
        This will construct the queries needed to refine the data to the necessary fields
        Parameters:
            logger (logging method): method of documentation from within the script.
            dblist (list): dictionaries of the path to parquet files

        Attributes:
            NOTE: These are all contained in the self.text_files dict
            db_paths (list): the databases intended to be concatenated together.
            

        Returns:
            string: the combination of queries needed to construct the dataframe.
        '''
        queries = []
        query_fields = []
        id_fields = []
        tables = []
        used_fields = []
        logger(f'Parser - schema: \n{db_schema}\n')
        def date_convert(self,date_field,tbl_name):
            tbl_date_field = tbl_name + "." + date_field.split()[0]
            date_name = date_field.split()[-1]
            data_tbl_name = db_schema['tables'][0]
            dat_str = f'CASE WHEN LENGTH({tbl_date_field})>=4 THEN TRY_CAST(SUBSTR({tbl_date_field},1,4) AS int) ELSE Null END AS {date_name}_year'
            if date_field in db_schema[data_tbl_name]['key_date']: # if this is the base date, it only needs the year
                return dat_str
            
            basedate_field = db_schema[data_tbl_name]['key_date'][0].split()[0]
            basedate_name = db_schema[data_tbl_name]['key_date'][0].split()[-1]
            tbl_basedate_name =  data_tbl_name+"."+basedate_field
            basedate = f"TRY_CAST('' || SUBSTR({tbl_basedate_name},1,4) || '-' || SUBSTR({tbl_basedate_name},5,2) || '-' || SUBSTR({tbl_basedate_name},7,2) || '' AS TIMESTAMP)"
            return dat_str+f", CASE WHEN LENGTH({tbl_date_field})>=8 THEN " \
                           f" datediff('days', {basedate}, TRY_CAST('' || SUBSTR({tbl_date_field},1,4) || '-' || SUBSTR({tbl_date_field},5,2) || '-' || SUBSTR({tbl_date_field},7,2) || '' AS TIMESTAMP)) " \
                           f'ELSE Null END AS {date_name}_{basedate_name}_diff' # if not then it needs both the year and the diff 
        for tbl_name in db_schema['tables']:
            tables.append(tbl_name)
            for names,field_list in db_schema[tbl_name].items():
                for field in field_list:
                    # append the table to to the field and account for substr() calls
                    if len(field.split(' '))==1:
                        field += " as " + field
                    split_field = field.split('(')
                    tbl_field = '('.join([tbl_name + "." + v if i==len(split_field)-1 else v for i,v in enumerate(split_field)])
                    f_name = field.split(' ')[-1]
                    if 'date' in names:
                        if f_name.lower() not in used_fields:
                            query_fields.append(date_convert(self,field,tbl_name))
                            used_fields.append(f_name.lower())
                    elif 'id' in names:
                        if f_name not in used_fields:
                            query_fields.append(tbl_name+ '.' +field)
                            used_fields.append(f_name)
                    elif 'baseline' in names:
                        query_fields.append('CAST(' + tbl_field.split(' ')[0] + ' as text) as ' + tbl_field.split(' ')[-1])
                    elif tbl_field in query_fields:
                        print(f'the field {tbl_field} is used more than once. Please review this, for now only one will be used.')
                    else:
                        query_fields.append(tbl_field)

        # create the join statement
        ## if the fields match use "using" otherwise use "on"
        logger(f'1st id_fields: {[v.split(" ")[0] for v in db_schema[tables[0]]["id_fields"]]}')
        join_first = [v.split(' ')[0].lower() for v in db_schema[tables[0]]['id_fields']]
        join_second = [v.split(' ')[0].lower() for v in db_schema[tables[1]]['id_fields']]

        join_str = ""
        if join_first == join_second:
            join_str = f'using({",".join(join_first)})'
        else:
            for f0, f1 in zip(join_first,join_second):
                if len(join_str) > 0:
                    join_str += " and"
                else:
                    join_str = " on"
                join_str += f" {tables[0]}.{f0} = {tables[1]}.{f1}"
        orderlist = [f for t in db_schema['tables'] for f in db_schema[t]['order_id_fields']]
        orderby_str = ','.join(sorted(set([v.split(' ')[-1] for v in orderlist])))
        for dbdict in dblist:
            queries.append(f'select {",".join(query_fields)} from {dbdict[tables[0]]} join {dbdict[tables[1]]} {join_str} order by {orderby_str};')
        self.text_files['duckdb_queries'] = queries

    def check_indexes(self, df, db_schema):
        '''
        will check the indexes for the database created
        check for uniqueness and non_null and that only one registryId occurs per registry

        Parameters:
            df (pd.DataFrame): data compiled for a single registry
        '''
        schema = db_schema
        truth_schema = schema[schema['tables'][1]]
        report_schema = schema[schema['tables'][0]]
        id_fields = [v.split(' ')[-1] for v in truth_schema['id_fields']]
        order_fields = sorted(set([v.split(' ')[-1] for v in truth_schema['order_id_fields']+report_schema['order_id_fields']]))
        full_id_fields = sorted(set(order_fields + id_fields))
        for c in id_fields:
            assert len(df[df[c].isna()]) == 0, f'{c} has empty fields in it'
        assert 'registryId' in full_id_fields, '"registryId" is not in the id_fields'
        assert 'registryId' in df.columns, '"registryId" is not in the database'
        assert len(df['registryId'].unique())==1, 'there are multiple registryIds per registry'
        
    def import_database(self, queries, db_schema):
        '''
        this will dump the fields of the database into a dataframe

        Parameters:
            queries (list): the list of combination of queries needed to construct the dataframe.

        Variables:
            df (pd.Dataframe): running concatenation of data pulled form different the databases listed
        '''
        import duckdb, pandas as pd
        df = [] # initialize data bases.  The array is a placeholder.
        for query in queries:
            df_temp = duckdb.query(query).to_df()
            self.check_indexes(df_temp, db_schema)
            # append data to bigger dataset
            df.append(df_temp)
        df = pd.concat(df)

        # duckdb has a glitch in which all "none" types are NaN (float)
        df = df.where(pd.notnull(df),None)
        self.pickle_files['df_raw'] = df

    def adjust_schema(self):
        '''
        this module will adjust the schema to only provide the column names that will be found in the DataFrame
        '''
        for t in self.text_files['db_schema']['tables']:
            self.text_files['db_schema'][t] = {k: [v1.split(' ')[-1] if len(v1)>0 else v1 for v1 in v] for k,v in self.text_files['db_schema'][t].items()}

    def describe(self, cc, describe, db_list):
        '''
        Capturing a summary of what has been done

        '''
        describe_list = []
        describe_list.append('Parser:\n')
        describe_list.append(f'The schema "{self.text_files["schema_name"]}" was used to combine the database packages {db_list}')
        describe_list.append(f'The following query was used:')
        describe_list.append(f'    {",".join(self.text_files["duckdb_queries"])}')
        describe_list.append(f'which total to {len(self.pickle_files["df_raw"])} records.')
        self.text_files['describe'] = describe + "\n".join(describe_list)

if __name__=='__main__':
    import argparse

    # provide options
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dbtime', '-dt', type=str, default='20220715',
                        help='string to test the date of the loaded paths. Default: 20220715')
    parser.add_argument('--dbreg', '-dr', type=str, default='LA',
                        help='string to test the registry of the loaded paths. Default: LA')
    parser.add_argument('--addfield_epath', '-afe', type=str, default=None,
                        help='additional ctc field to include in the query')
    parser.add_argument('--addfield_ctc', '-afc', type=str, default=None,
                        help='additional epath field to include in the query')
    parser.add_argument('--logging', '-l', type=int, default=20,
                        help='this is the logging level that you will see. Debug is 10')

    args = parser.parse_args()
    class tempClass():
        def __init__(self, db_list, add_fields, verb):
            import logging
            logging.root.setLevel(verb)
            self.logging = logging
            self.cur_path = os.path.dirname(os.path.abspath(__file__))
            Parser.get_paths(self)
            self.db_list = db_list
            self.add_fields = {}
            if len(add_fields[0]) == 1:
                self.add_fields['epath'] = [add_fields[0]]
            if len(add_fields[1]) == 1:
                self.add_fields['ctc'] = [add_fields[1]]
            self.pickle_files = {}
            self.text_files = {}

        def check_indexes(self, df, schema):
            Parser.check_indexes(self, df, schema)


    def null_print(*string):
        pass

    temp_self = tempClass(db_list = {args.dbtime:[args.dbreg]}, add_fields = [args.addfield_epath, args.addfield_ctc], verb = args.logging)

    db_list, schema = Parser.fill_path(temp_self, temp_self.db_list)
    Parser.test_db(temp_self, null_print, db_list, schema, temp_self.add_fields)
    Parser.build_query(temp_self, null_print, db_list, temp_self.text_files['db_schema'])
    Parser.import_database(temp_self, temp_self.text_files['duckdb_queries'], temp_self.text_files['db_schema'])

    temp_self.pickle_files['df_raw'].to_pickle(os.path.join(temp_self.cur_path,'documentation','df_raw.pkl'))
    print(f"database was saved to {os.path.join(temp_self.cur_path,'documentation','df_raw.pkl')}")
