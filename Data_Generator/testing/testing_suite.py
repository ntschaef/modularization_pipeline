import unittest, sys, os, json, pickle, pandas as pd, sqlite3, numpy as np, types
os.chdir('..')
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from data_generator import Data_Generator

vocab_args = {'vocab_source':'train', 'use_security': False, 'use_w2v': False, 'np_seed':-1}
w2v_args = {'size': 20, 'min_count':1, 'window':5, 'workers': 5, 'iter':10}
vocab_args['w2v_kwargs'] = w2v_args
cur_path = os.path.dirname(os.path.abspath(__file__))


class mockCacheClass():
    def __init__(self):
        self.saved_mod_args = {}
        self.fold = pd.read_pickle(os.path.join(cur_path, 'testing', 'data', 'fold'))
        self.df = pd.read_pickle(os.path.join(cur_path, 'testing', 'data', 'test_df'))
        with open(os.path.join(cur_path,'testing','data','schema'), 'rb') as jb:
            self.schema = pickle.load(jb)
    def check_args(self, mod, args):
        return True
    def null_print(*string):
        pass
     
class ittr_dataset():
    def __init__(s):
        s.filename = os.path.join(cur_path,'testing','data','test_df')
        s.func = []

    def __iter__(s,column_num=0):
        df = pd.read_pickle(s.filename)
        for f in s.func:
            df = f(df)
        for i in range(len(df)):
            yield df.iat[i, column_num]

    def add_func(s, func):
        s.func.append(func)

cc = mockCacheClass()

dg = Data_Generator(cc)
## functions specific to the repo
def test_vocab():
     count_bv_vsTrain = dg.build_vocab(cc.null_print, records = ittr_dataset(), fold = cc.fold, fold_num = 0, data_fields = cc.schema['data_fields'], **vocab_args)

     vocab_args['vocab_source']='train_val_test'
     count_bv_vsTVT = dg.build_vocab(cc.null_print, records = ittr_dataset(), fold = cc.fold, fold_num = 0, data_fields = cc.schema['data_fields'], **vocab_args)

     vocab_args['vocab_source']='raw'
     count_bv_base = dg.build_vocab(cc.null_print, records = ittr_dataset(), fold = cc.fold, fold_num = 0, data_fields = cc.schema['data_fields'], **vocab_args)

     count_bv_dfAlt = dg.build_vocab(cc.null_print, records = ittr_dataset(), fold = cc.fold, fold_num = 0, data_fields = cc.schema['data_fields1'], **vocab_args)

     vocab_args['use_security']=True
     count_bv_sec = dg.build_vocab(cc.null_print, records = ittr_dataset(), fold = cc.fold, fold_num = 0, data_fields = cc.schema['data_fields'], **vocab_args)

     vocab_args['use_security']=False
     vocab_args['use_w2v']=True
     count_bv_w2v = dg.build_vocab(cc.null_print, records = ittr_dataset(), fold = cc.fold, fold_num = 0, data_fields = cc.schema['data_fields'], **vocab_args)

     vocab_args['use_w2v']=False
     vocab_args['w2v_kwargs']['min_count']=5
     count_bv_mvc5 = dg.build_vocab(cc.null_print, records = ittr_dataset(), fold = cc.fold, fold_num = 0, data_fields = cc.schema['data_fields'], **vocab_args)
     vocab_args['w2v_kwargs']['min_count']=1
     return count_bv_vsTrain, count_bv_vsTVT, count_bv_base, count_bv_dfAlt, count_bv_sec, count_bv_w2v, count_bv_mvc5

records = np.array(cc.df['text1'].apply(lambda v: v.split(' ')))
def test_w2v():
     count_w2v_base = dg.run_w2v(cc.null_print, records = records, fold_num = 0, np_seed = -1, w2v_kwargs = w2v_args)
     il_w2v_base = dg.pickle_files['word_embeds_fold0']
     w2v_args['min_count'] = 5
     count_w2v_mc5 = dg.run_w2v(cc.null_print, records = records, fold_num = 0, np_seed = -1, w2v_kwargs = w2v_args)
     w2v_args['min_count'] = 1
     w2v_args['size'] = 10
     count_w2v_s10 = dg.run_w2v(cc.null_print, records = records, fold_num = 0, np_seed = -1, w2v_kwargs = w2v_args)
     il_w2v_s10 = dg.pickle_files['word_embeds_fold0']

     return count_w2v_base, il_w2v_base, count_w2v_mc5, count_w2v_s10, il_w2v_s10

def test_random():
     count_rand_base = dg.randomize_embeddings(cc.null_print, records = records, fold_num = 0, min_count = 1, input_size = 20, np_seed = -1)
     il_rand_base = dg.pickle_files['word_embeds_fold0']
     count_rand_mc5 = dg.randomize_embeddings(cc.null_print, records = records, fold_num = 0, min_count = 5, input_size = 20, np_seed = -1)
     count_rand_s10 = dg.randomize_embeddings(cc.null_print, records = records, fold_num = 0, min_count = 1, input_size = 10, np_seed = -1)
     il_rand_s10 = dg.pickle_files['word_embeds_fold0']

     return count_rand_base, il_rand_base, count_rand_mc5, count_rand_s10, il_rand_s10

def test_labels():
     df = cc.df
     map_labels_base = dg.build_labels(cc.null_print, records = cc.df, train = cc.fold[0], fold_num = 0, token_fields = cc.schema['tokens'], min_label_count = 1, add_tasks = [], add_labels = {})  
     cc.df = df
     map_labels_tokAlt1 = dg.build_labels(cc.null_print, records = cc.df, train = cc.fold[0], fold_num = 0, token_fields = cc.schema['tokens1'], min_label_count = 1, add_tasks = [], add_labels = {})
     cc.df = df
     map_labels_tokAlt2 = dg.build_labels(cc.null_print, records = cc.df, train = cc.fold[0], fold_num = 0, token_fields = cc.schema['tokens2'], min_label_count = 1, add_tasks = [], add_labels = {})
     cc.df = df
     map_labels_mlc3 = dg.build_labels(cc.null_print, records = cc.df, train = cc.fold[0], fold_num = 0, token_fields = cc.schema['tokens2'], min_label_count = 3, add_tasks = [], add_labels = {})
     cc.df = df
     map_labels_at = dg.build_labels(cc.null_print, records = cc.df, train = cc.fold[0], fold_num = 0, token_fields = cc.schema['tokens'], min_label_count = 1, add_tasks = ['labeltest'], add_labels = {'labeltest':['test1']})
     cc.df = df
     map_labels_al = dg.build_labels(cc.null_print, records = cc.df, train = cc.fold[0], fold_num = 0, token_fields = cc.schema['tokens'], min_label_count = 1, add_tasks = [], add_labels = {'label1':['test1']})
     cc.df = df
     return map_labels_base, map_labels_tokAlt1, map_labels_tokAlt2, map_labels_mlc3, map_labels_at, map_labels_al

########## Test Calls (general runs)

class TestCleanerInternal(unittest.TestCase):
    def test_reqmethods_builddataset(self):
        self.assertTrue('build_dataset' in Data_Generator.__dict__.keys(), 'cannot find build_dataset in the script')
    def test_reqmethods_predsingle(self):
        self.assertTrue('predict_single' in Data_Generator.__dict__.keys(), 'cannot find predict_single in the script')
    def test_reqmethods_predmulti(self):
        self.assertTrue('predict_multi' in Data_Generator.__dict__.keys(), 'cannot find predict_mulit in the script')
    def test_reqmethods_getcacheddeps(self):
        self.assertTrue('get_cached_deps' in Data_Generator.__dict__.keys(), 'cannot find get_cached_deps in the script')
    def test_reqmethods_describe(self):
        self.assertTrue('describe' in Data_Generator.__dict__.keys(), 'cannot find describe in the script')

########## Test Files (simi-class specific runs)

class TestCleanerFiles(unittest.TestCase):
    def test_onlyFiles(self):
        self.assertTrue(sorted([v for v in os.listdir() if '__' not in v and v[0]!='.']) == ['BioWordVec.pickle','README.md','data_generator.py','documentation','testing'], 
               'there are files that should not be included in the module')

########## Test expected outputs (class specific)

class TestOutput(unittest.TestCase):
    _ = dg.build_vocab(cc.null_print, records = ittr_dataset(), fold = cc.fold, fold_num = 0, data_fields = cc.schema['data_fields'], **vocab_args)
    _ = dg.build_labels(cc.null_print, records = cc.df, train = cc.fold[0], fold_num = 0, token_fields = cc.schema['tokens'], min_label_count = 1, add_tasks = [], add_labels = {}) 
    dg.describe('', ['test'], {**vocab_args, "np_seed":0, "min_label_count":1, "add_tasks": [], "add_labels": {}}, [10], [{'label1':['a','b']}])
    def test_outputs_pickles(self):
        self.assertTrue(sorted(list(dg.pickle_files.keys())) == ['id2word_fold0','word_embeds_fold0'], f'expected pickle output is ["id2word_fold0","word_embeds_fold0"]; got {sorted(list(dg.pickle_files.keys()))}')
    def test_outputs_text(self):
        self.assertTrue(sorted(list(dg.text_files.keys())) == ['describe','id2labels_fold0'], f'expected text output is ["describe", "id2labels_fold0"]; got {sorted(list(dg.text_files.keys()))}')

########## Test Module (class specific runs)

class TestSplitterInternal(unittest.TestCase):
    bv_vt, bv_vtvt, bv_base, bv_fAlt, bv_s, bv_w2v, bv_m5 = test_vocab()
    w_base, i_w_base, w_c5, w_s10, i_w_s10 = test_w2v()
    r_base, i_r_base, r_c5, r_s10, i_r_s10 = test_random()
    l_base, l_tA1, l_tA2, l_c3, l_at, l_al = test_labels() 

    def test_vocab_base(self):
        self.assertTrue(self.bv_base==10, f'expected vocab to be 10, got {self.bv_base}')
    def test_vocab_train(self):
        self.assertTrue(self.bv_vt==7, f'expected vocab to be 7, got {self.bv_vt}')
    def test_vocab_tvt(self):
        self.assertTrue(self.bv_vtvt==8, f'expected vocab to be 8, got {self.bv_vt}')
    def test_vocab_fAlt(self):
        self.assertTrue(self.bv_fAlt==16, f'expected vocab to be 16, got {self.bv_fAlt}')
    def test_vocab_sec(self):
        self.assertTrue(self.bv_s==9, f'expected vocab to be 9, got {self.bv_s}')
    def test_vocab_w2v(self):
        self.assertTrue(self.bv_w2v==10, f'expected vocab to be 10, got {self.bv_w2v}')
    def test_vocab_min5(self):
        self.assertTrue(self.bv_m5==6, f'expected vocab to be 6, got {self.bv_m5}')
    def test_w2v_base(self):
        self.assertTrue(self.w_base==10, f'expected vocab to be 10, got {self.w_base}')
    def test_w2v_il_base(self):
        self.assertTrue(self.i_w_base.shape[1]==20, f'size expected to to be 20, got {self.i_w_base.shape[1]}')
    def test_w2v_min5(self):
        self.assertTrue(self.w_c5==6, f'expected vocab to be 6, got {self.w_c5}')
    def test_w2v_size10(self):
        self.assertTrue(self.w_s10==10, f'expected vocab to be 10, got {self.w_s10}')
    def test_w2v_il_size10(self):
        self.assertTrue(self.i_w_s10.shape[1]==10, f'size expected to to be 20, got {self.i_w_s10.shape[1]}')
    def test_rand_base(self):
        self.assertTrue(self.r_base==10, f'expected vocab to be 10, got {self.r_base}')
    def test_rand_il_base(self):
        self.assertTrue(self.i_r_base.shape[1]==20, f'size expected to to be 20, got {self.i_r_base.shape[1]}')
    def test_rand_min5(self):
        self.assertTrue(self.r_c5==6, f'expected vocab to be 6, got {self.r_c5}')
    def test_rand_size10(self):
        self.assertTrue(self.r_s10==10, f'expected vocab to be 10, got {self.r_s10}')
    def test_rand_il_size10(self):
        self.assertTrue(self.i_r_s10.shape[1]==10, f'size expected to to be 20, got {self.i_r_s10.shape[1]}')
    def test_label_base_keys(self):
        self.assertTrue(len(self.l_base.keys())==1, f'expected label to be 1, got {self.l_base.keys()}')
    def test_label_base_items(self):
        self.assertTrue(len(list(self.l_base.values())[0])==1, f'expected label count to be 1, got {list(self.l_base.values())[0]}')
    def test_label_altTok_multi_keys(self):
        self.assertTrue(len(self.l_tA1.keys())==2, f'expected labels to be 2, got {self.l_tA1.keys()}')
    def test_label_altTok_multi_items1(self):
        self.assertTrue(len(list(self.l_tA1.values())[0])==1, f'expected label count for first label to be 1, got {list(self.l_tA1.values())[0]}')
    def test_label_altTok_multi_items2(self):
        self.assertTrue(len(list(self.l_tA1.values())[1])==2, f'expected label count for second label to be 2, got {list(self.l_tA1.values())[1]}')
    def test_label_altTok_combined_keys(self):
        self.assertTrue(len(self.l_tA2.keys())==1, f'expected labels to be 1, got {self.l_tA2.keys()}')
    def test_label_altTok_combined_items(self):
        self.assertTrue(len(list(self.l_tA2.values())[0])==2, f'expected label count to be 2, got {list(self.l_tA2.values())[0]}')
    def test_label_min3_keys(self):
        self.assertTrue(len(self.l_c3.keys())==1, f'expected labels to be 1, got {self.l_c3.keys()}')
    def test_label_min3_items(self):
        self.assertTrue(len(list(self.l_c3.values())[0])==1, f'expected labels count to be 1, got {list(self.l_c3.values())[0]}')
    def test_label_addtask_task(self):
        self.assertTrue('labeltest' in self.l_at.keys(), f'expected to find the new task "labeltest" in the tasks, but the list was {", ".join(self.l_at.keys())}')
    def test_label_addtask_label(self):
        dict_str = '{0:"test1"}'
        self.assertTrue({0:'test1'}==self.l_at['labeltest'], f'new label list expected to be {dict_str} instead found {str(self.l_at["labeltest"])}')
    def test_label_addlabel(self):
        dict_str = '{0:"1",1:"test1"}'
        self.assertTrue(self.l_al['label1']=={0:'1',1:'test1'}, f'label list expected to be {dict_str} for label1, instead found {str(sorted([str(v) for v in self.l_al["label1"]]))}')


TestResult = unittest.main(exit=False)
fail_num = len(TestResult.result.failures)
error_num = len(TestResult.result.errors)
test_num = TestResult.result.testsRun
lines = ['successes: ' + str(test_num - (fail_num + error_num)), 'failures: ' + str(fail_num), 'errors: ' + str(error_num), 'Total tests: ' + str(test_num)]
with open(os.path.join('documentation','test_results.md'),'w') as doc:
    doc.write('\n'.join(lines))
