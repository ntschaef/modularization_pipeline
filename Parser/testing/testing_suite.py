import unittest, sys, os, pandas as pd
os.chdir('..')
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from parser import Parser

class tempCacheClass():
    def __init__(self, db_list = ['test']):
        self.saved_mod_args = {}
        self.cur_path = os.path.dirname(os.path.abspath(__file__))
        self.db_path_dict = {'test': [os.path.join(self.cur_path,'testing','db','test.sqlite'),'ctc'],
                             'test2': [os.path.join(self.cur_path, 'testing','db','test_extend.sqlite'), 'ctc']}
        self.db_list = db_list

    def mod_args(self, key):
        return self.saved_mod_args[key][0]

    def null_print(*string):
        pass
    def check_args(self, mod, args):
        return True

def build_mod(db_list=['test']):
    '''
    a test run of all the fucntions to ensure they work. This is a sample to run the scripts.
    '''
    ################## manual setup of 'self'

    cc = tempCacheClass(db_list) 
    temp_self = Parser(cc)
    temp_self.db_path_dict = cc.db_path_dict

    ################## continue with script

    temp_self.test_db(cc.null_print, db_list, [])
    temp_self.build_query(cc.null_print)
    path_list = [temp_self.db_path_dict[v][0] for v in db_list]
    temp_self.import_database(temp_self.text_files['sql_query'],path_list,temp_self.text_files['db_schema'])
    temp_self.describe(cc,"")
    return temp_self

def testColumnDups(df):
    first = []
    second = []
    for c in df.columns:
        if c in first:
            second.append(c)
        else:
            first.append(c)
    return second

########## Test Calls (general runs)

class TestParserReq(unittest.TestCase):
    def test_reqmethods_builddataset(self):
        self.assertTrue('build_dataset' in Parser.__dict__.keys(), 'cannot find build_dataset in the script')
    def test_reqmethods_predsingle(self):
        self.assertTrue('predict_single' in Parser.__dict__.keys(), 'cannot find predict_single in the script')
    def test_reqmethods_predmulti(self):
        self.assertTrue('predict_multi' in Parser.__dict__.keys(), 'cannot find predict_multi in the script')
    def test_reqmethods_getcacheddeps(self):
        self.assertTrue('get_cached_deps' in Parser.__dict__.keys(), 'cannot find get_cached_deps in the script')
    def test_reqmethods_describe(self):
        self.assertTrue('describe' in Parser.__dict__.keys(), 'cannot find describe in the script')

########## Test Files (simi-class specific runs)

class TestCleanerFiles(unittest.TestCase):
    def test_onlyFiles(self):
        self.assertTrue(sorted([v for v in os.listdir() if '__' not in v and v[0]!='.']) == ['README.md','documentation','parser.py','schema.json','testing'], 
               'there are files that should not be included in the module')

########## Test expected outputs (class specific)

class TestOutput(unittest.TestCase):
    temp = build_mod()
    def test_outputs_pickles(self):
        self.assertTrue(sorted(list(self.temp.pickle_files.keys())) == ['df_raw'], f'expected pickle output is ["df_raw"]; got {sorted(list(self.temp.pickle_files.keys()))}')
    def test_outputs_text(self):
        self.assertTrue(sorted(list(self.temp.text_files.keys())) == ['db_paths', 'db_schema', 'describe', 'schema_name', 'sql_query'], f'expected text output is ["db_paths", "db_schema", "describe", "schema_name", "sql_query"]; got {sorted(list(self.temp.text_files.keys()))}')

########## Test Module (class specific)

class TestParserInternal(unittest.TestCase):
    temp_self1 = build_mod()
    temp_self2 = build_mod(['test','test2'])
    def test_genmethods_dblist(self):
        self.assertTrue(len(self.temp_self1.text_files['db_paths'])>0, f'the file path was not saved')
    def test_genmethods_query(self):
        self.assertTrue(self.temp_self1.text_files['sql_query']!="",'query was never built')
    def test_genmethods_importdatabase(self):
        self.assertTrue(len(self.temp_self1.pickle_files['df_raw'])>0,'dataframe was not created')
    def test_genmethods_dflength(self):
        self.assertTrue(len(self.temp_self1.pickle_files['df_raw']) == 5, f'database was not the correct length. Expected 5, got {len(self.temp_self1.pickle_files["df_raw"])}')
    def test_genmethods_dupfields(self):
        self.assertTrue(len(testColumnDups(self.temp_self1.pickle_files['df_raw']))==0, f'the dataframe build has duplicate column names: {testColumnDups(self.temp_self1.pickle_files["df_raw"])}')
    def test_genmethods_dblist2(self):
        self.assertTrue(len(self.temp_self2.text_files['db_paths'])>0, f'the file path was not saved')
    def test_genmethods_query2(self):
        self.assertTrue(self.temp_self2.text_files['sql_query']!="",'query was never built')
    def test_genmethods_importdatabase2(self):
        self.assertTrue(len(self.temp_self2.pickle_files['df_raw'])>0,'dataframe was not created')
    def test_genmethods_dflength2(self):
        self.assertTrue(len(self.temp_self2.pickle_files['df_raw']) == 9, f'database was not the correct length. Expected 9, got {len(self.temp_self2.pickle_files["df_raw"])}')
    def test_genmethods_dupfields2(self):
        self.assertTrue(len(testColumnDups(self.temp_self2.pickle_files['df_raw']))==0, f'the dataframe build has duplicate column names: {testColumnDups(self.temp_self2.pickle_files["df_raw"])}')

########### Output results
    
TestResult = unittest.main(exit=False)
fail_num = len(TestResult.result.failures)
error_num = len(TestResult.result.errors)
test_num = TestResult.result.testsRun
lines = ['successes: ' + str(test_num - (fail_num + error_num)), 'failures: ' + str(fail_num), 'errors: ' + str(error_num), 'Total tests: ' + str(test_num)]
with open(os.path.join('documentation','test_results.md'),'w') as doc:
    doc.write('\n'.join(lines))
    
