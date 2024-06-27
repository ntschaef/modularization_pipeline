import unittest, sys, os, json, pandas as pd, sqlite3
os.chdir('..')
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from sanitizer import Sanitizer

args = {'deidentify':False, 'remove_dups': True, 'min_doc_length': 20}

class mockCacheClass():
    def __init__(self):
        self.saved_mod_args = {}
        cur_path = os.path.dirname(os.path.abspath(__file__))
        self.df = pd.read_pickle(os.path.join(cur_path,'testing','data','dfclean'))
    def check_args(self, mod, args):
        return True
    def null_print(*string):
        pass

schema = {'tables':['tbl0','tbl1'], 'tbl0':{'data_fields':['text_1','text2'],'id_fields':['id1','id2'],'order_id_fields':['order0','registryId']}, 'tbl1': {'id_fields':[],'order_id_fields':[]}}

cc = mockCacheClass()
s = Sanitizer(cc)

alt_df = cc.df[:-2]

def run_sanitizer(df):
    def sd_run(newdf):
        def extract_saved(ft, fn):
            assert ft in ['pickle','text'], f'expected "ft" to be "pickle" or "text", not "{ft}"'
            if ft == 'pickle':
                saved = s.pickle_files
            elif ft == 'text':
                saved = s.text_files
            if fn in saved:
                value = saved[fn]
#                _ = saved.pop(fn)
            else:
                value = False
            return value
            
        deps = {'Sanitizer': [['TestModule'], ['TestEnvVar'], ['deidentify'], ['TestPythonPackage']]}
        pack = {'Sanitizer': [['dep_module_hash'],['environ_string'],[True],['0.0'],'currentHash']}
        s.make_token_dump(cc.null_print, deps, pack, newdf, schema) 
        s.describe("", schema, args, 1, 0)
        s.describe("", schema, args, 0, 1)
        toks = extract_saved('pickle', 'df_tokens')
        dump = extract_saved('text', 'token_dump')
        ids = extract_saved('text', 'id_map')
        words = extract_saved('text', 'word_map')
        return toks, (dump, ids, words)

    original, _ = sd_run(df)
    tokens, (dump, ids, words) = sd_run(df)
    original_test = (original==df).all().all()
    return original_test, tokens, words, ids, dump

def dump_convert(dump_text):
    conn = sqlite3.connect(':memory:')
    conn.executescript(dump_text)
    df = pd.read_sql(f'select * from {"_".join(schema["tables"])}', conn)
    return df 

def word_convert(df, w_dict, df_o):
    rev_w = {str(v):k for k,v in w_dict.items()}
    test = True
    for c in df.columns:
        if 'text' in c:
            col = [' '.join([rev_w[t] for t in sent.split(' ')]) for sent in df[c]]
            if (df_o[c] != col).any():
                test = False
    return test

def id_convert(df, i_dict, df_o):
    rev_i = {str(v):k for k,v in i_dict.items()}
    test = True
    for c in df.columns:
        if 'text' not in c:
            col = [rev_i[str(tok)] for tok in df[c]]
            if (df_o[c] != col).any():
                test = False
    return test

########## Test Calls (general runs)

class TestCleanerInternal(unittest.TestCase):
    def test_reqmethods_builddataset(self):
        self.assertTrue('build_dataset' in Sanitizer.__dict__.keys(), 'cannot find build_dataset in the script')
    def test_reqmethods_predsingle(self):
        self.assertTrue('predict_single' in Sanitizer.__dict__.keys(), 'cannot find predict_single in the script')
    def test_reqmethods_predmulti(self):
        self.assertTrue('predict_multi' in Sanitizer.__dict__.keys(), 'cannot find predict_mulit in the script')
    def test_reqmethods_getcacheddeps(self):
        self.assertTrue('get_cached_deps' in Sanitizer.__dict__.keys(), 'cannot find get_cached_deps in the script')
    def test_reqmethods_describe(self):
        self.assertTrue('describe' in Sanitizer.__dict__.keys(), 'cannot find describe in the script')

########## Test Files (simi-class specific runs)

class TestCleanerFiles(unittest.TestCase):
    def test_onlyFiles(self):
        self.assertTrue(sorted([v for v in os.listdir() if '__' not in v and v[0]!='.']) == ['README.md','documentation','sanitizer.py','testing'], 
               'there are files that should not be included in the module')

########## Test expected outputs (class specific)

class TestOutput(unittest.TestCase):
    def test_outputs_pickles(self):
        self.assertTrue(sorted(list(s.pickle_files.keys())) == ['df_tokens'], f'expected pickle output is ["df_cleaned"]; got {sorted(list(s.pickle_files.keys()))}')
    def test_outputs_text(self):
        self.assertTrue(sorted(list(s.text_files.keys())) == ['describe', 'id_map', 'token_dump', 'word_map'], f'expected text output is ["describe","id_map", "token_dump", "word_map"]; got {sorted(list(s.text_files.keys()))}')
 

########## Test Module (class specific runs)
    
class TestParserInternal(unittest.TestCase):
    ot, t, w, i, d = run_sanitizer(alt_df)
    ri,_,_,_ = s.check_data(cc.df, schema, 0, 'True') 
    dt = dump_convert(d)
    wt = word_convert(dt, w, alt_df)
    it = id_convert(dt, i, alt_df)
    def test_token_type(self):
        self.assertTrue(isinstance(self.t, pd.DataFrame), f'the tokenized return is not a dataframe {self.t}')
    def test_wordMap_type(self):
        self.assertTrue(isinstance(self.w, dict), 'the word mapping is not a dictionary')
    def test_idMap_type(self):
        self.assertTrue(isinstance(self.i, dict), 'the word mapping is not a dictionary')
    def test_dump_type(self):
        self.assertTrue(isinstance(self.d, str), 'the dump is not text')
    def test_dropdupids(self):
        self.assertTrue(len(self.ri)==3, 'the amount of records returned after dropping duplicates are incorrect')
    def test_dump2word(self):
        self.assertTrue(self.wt, 'the words converted incorrectly')
    def test_dump2id(self):
        self.assertTrue(self.it, 'the ids converted incorrectly')

TestResult = unittest.main(exit=False)
fail_num = len(TestResult.result.failures)
error_num = len(TestResult.result.errors)
test_num = TestResult.result.testsRun
lines = ['successes: ' + str(test_num - (fail_num + error_num)), 'failures: ' + str(fail_num), 'errors: ' + str(error_num), 'Total tests: ' + str(test_num)]
with open(os.path.join('documentation','test_results.md'),'w') as doc:
    doc.write('\n'.join(lines))
