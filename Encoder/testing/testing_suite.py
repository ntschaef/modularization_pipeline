import unittest, sys, os, json, pickle, pandas as pd, sqlite3, numpy as np
os.chdir('..')
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from encoder import Encoder
temp_args = {'metadata_fields':['id1'],'reverse_tokens':False, 'remove_unks':False, 'sequence_length': 20}
vocab = {0:'a',1:'b',2:'c',3:'d',4:'e',5:'f',6:'g',7:'h',8:'i',9:'j',
         10:'k',11:'l',12:'m',13:'n',14:'o',15:'p',16:'q',17:'r',18:'s',19:'t',
         20:'u',21:'v',22:'w',23:'x',24:'y',25:'z',26:'recordbreaktoken',27:'<unk>'}
data_fields = ['text1', 'text2', 'text3']
index = ['ord_id1']
truth_fields = ['truth1']


class mockCacheClass():
    def __init__(self):
        self.saved_mod_args = {}
        cur_path = os.path.dirname(os.path.abspath(__file__))
        self.df = pd.read_pickle(os.path.join(cur_path,'testing','data','test_df'))
        self.fold = pd.read_pickle(os.path.join(cur_path, 'testing', 'data', 'fold'))
    def check_args(self, mod, args):
        return True
    def null_print(*string):
        pass

cc = mockCacheClass()
e = Encoder(cc)

## functions specific to the repo
def test_inputs():
     # testing the default with a small count of reports (9... only 5 will be in the train set), shows unk mapping, non-reverse, record size 20
     e.generate_data(cc.null_print, records = cc.df.iloc[:9], index=index, truth_fields=truth_fields, data_fields=data_fields, fold = cc.fold, fold_num = 0, vocab=vocab, **temp_args)
     base_5 = e.pickle_files.copy()
     # testing the default with larger report set (100) (mutliprocessing)
     e.generate_data(cc.null_print, records = cc.df, index=index, truth_fields=truth_fields, data_fields=data_fields, fold = cc.fold, fold_num = 0, vocab=vocab, **temp_args)
     base_15 = e.pickle_files.copy()
     # test both small and large with reverse tokens
     temp_args['reverse_tokens'] = True
     e.generate_data(cc.null_print, records = cc.df.iloc[:9], index=index, truth_fields=truth_fields, data_fields=data_fields, fold = cc.fold, fold_num = 0, vocab=vocab, **temp_args)
     rev_5 = e.pickle_files.copy()
     e.generate_data(cc.null_print, records = cc.df, index=index, truth_fields=truth_fields, data_fields=data_fields, fold = cc.fold, fold_num = 0, vocab=vocab, **temp_args)
     rev_15 = e.pickle_files.copy()
     # test both small and large with removing unknown terms
     temp_args['reverse_tokens'] = False
     temp_args['remove_unks'] = True
     e.generate_data(cc.null_print, records = cc.df.iloc[:9], index=index, truth_fields=truth_fields, data_fields=data_fields, fold = cc.fold, fold_num = 0, vocab=vocab, **temp_args)
     ru_5 = e.pickle_files.copy()
     e.generate_data(cc.null_print, records = cc.df, index=index, truth_fields=truth_fields, data_fields=data_fields, fold = cc.fold, fold_num = 0, vocab=vocab, **temp_args)
     ru_15 = e.pickle_files.copy()
     # test both small and large with different size report length (10)
     temp_args['remove_unks'] = False
     temp_args['sequence_length'] = 10
     e.generate_data(cc.null_print, records = cc.df.iloc[:9], index=index, truth_fields=truth_fields, data_fields=data_fields, fold = cc.fold, fold_num = 0, vocab=vocab, **temp_args)
     l10_5 = e.pickle_files.copy()
     e.generate_data(cc.null_print, records = cc.df, index=index, truth_fields=truth_fields, data_fields=data_fields, fold = cc.fold, fold_num = 0, vocab=vocab, **temp_args)
     l10_15 = e.pickle_files.copy()

     return base_5, base_15, rev_5, rev_15, ru_5, ru_15, l10_5, l10_15

########## Test Calls (general runs)

class TestCleanerInternal(unittest.TestCase):
    def test_reqmethods_builddataset(self):
        self.assertTrue('build_dataset' in Encoder.__dict__.keys(), 'cannot find build_dataset in the script')
    def test_reqmethods_predsingle(self):
        self.assertTrue('predict_single' in Encoder.__dict__.keys(), 'cannot find predict_single in the script')
    def test_reqmethods_predmulti(self):
        self.assertTrue('predict_multi' in Encoder.__dict__.keys(), 'cannot find predict_mulit in the script')
    def test_reqmethods_getcacheddeps(self):
        self.assertTrue('get_cached_deps' in Encoder.__dict__.keys(), 'cannot find get_cached_deps in the script')
    def test_reqmethods_describe(self):
        self.assertTrue('describe' in Encoder.__dict__.keys(), 'cannot find describe in the script')

########## Test Files (simi-class specific runs)

class TestCleanerFiles(unittest.TestCase):
    def test_onlyFiles(self):
        self.assertTrue(sorted([v for v in os.listdir() if '__' not in v and v[0]!='.']) == ['README.md','documentation','encoder.py','testing'], 
               'there are files that should not be included in the module')

########## Test expected outputs (class specific)

class TestOutput(unittest.TestCase):
    
    _ = e.generate_data(cc.null_print, records = cc.df.iloc[:9], index=index, truth_fields=truth_fields, data_fields=data_fields, fold = cc.fold, fold_num = 0, vocab=vocab, **temp_args)
    _ = e.generate_data(cc.null_print, records = cc.df, index=index, truth_fields=truth_fields, data_fields=data_fields, fold = cc.fold, fold_num = 0, vocab=vocab, **temp_args)
    e.describe('', temp_args)
    def test_outputs_pickles(self):
        self.assertTrue(sorted(list(e.pickle_files.keys())) == ["testMetadata_fold0", "testX_fold0", "testY_fold0", "trainMetadata_fold0", "trainX_fold0", "trainY_fold0", "valMetadata_fold0", "valX_fold0", "valY_fold0"], f'expected pickle output is ["testMetadata_fold0", "testX_fold0", "testY_fold0", "trainMetadata_fold0", "trainX_fold0", "trainY_fold0", "valMetadata_fold0", "valX_fold0", "valY_fold0"]; got {sorted(list(e.pickle_files.keys()))}')
    def test_outputs_text(self):
        self.assertTrue(sorted(list(e.text_files.keys())) == ['describe'], f'expected text output is ["describe"]; got {sorted(list(e.text_files.keys()))}')

########## Test Module (class specific runs)
    
class TestSplitterInternal(unittest.TestCase):
    ### TODO: size

    b5, b15, rt5, rt15, ru5, ru15, l5, l15 = test_inputs()
    def test_baseline5_train_length(self):
        self.assertTrue(list(map(len,self.b5["trainX_fold0"])) == [5,9,18,20,20], f'lengths of the records were not correct. Expected [5,9,18,20,20], got {list(map(len,self.b5["trainX_fold0"]))}.')
    def test_baseline5_val_length(self):
        self.assertTrue(list(map(len,self.b5["valX_fold0"])) == [7,20], f'lengths of the records were not correct. Expected [7,20], got {list(map(len,self.b5["valX_fold0"]))}.')
    def test_baseline5_test_length(self):
        self.assertTrue(list(map(len,self.b5["testX_fold0"])) == [20,4], f'lengths of the records were not correct. Expected [20,4], got {list(map(len,self.b5["testX_fold0"]))}.')
    def test_baseline5_unks(self):
        self.assertTrue(self.b5["trainX_fold0"][1][0] == 27, f'expected word {cc.df.iloc[1]["text1"].split(" ")[0]} to be unknown and map to 27.  Got mapping of {self.b5["trainX_fold0"][1][0]} instead which mapped to {vocab[self.b5["trainX_fold0"][1][0]]}') 
    def test_baseline5_1strec(self):
        self.assertTrue(self.b5["trainX_fold0"][0] == [4,3,2,26,26], f'expected record "e d c" to map to [4,3,2,26,26]. Instead {cc.df.iloc[0]} mapped to {self.b5["trainX_fold0"][0]}')
    def test_baseline15_train_length(self):
        self.assertTrue(list(map(len,self.b15["trainX_fold0"])) == [5,9,18,20,20,7,14,20,3,12,20,16,20,19,4], f'lengths of the records were not correct. Expected [5,9,18,20,20,7,14,20,3,12,20,16,20,19,4], got {list(map(len,self.b15["trainX_fold0"]))}.')
    def test_baseline15_val_length(self):
        self.assertTrue(list(map(len,self.b15["valX_fold0"])) == [7,20,12,20,16,5,20,19,8,6,20], f'lengths of the records were not correct. Expected [7,20,12,20,16,5,20,19,8,6,20], got {list(map(len,self.b15["valX_fold0"]))}.')
    def test_baseline15_test_length(self):
        self.assertTrue(list(map(len,self.b15["testX_fold0"])) == [20,4,5,20,19,8,6,20,13,7,20,18], f'lengths of the records were not correct. Expected [20,4,5,20,19,8,6,20,13,7,20,18], got {list(map(len,self.b15["testX_fold0"]))}.')
    def test_baseline15_unks(self):
        self.assertTrue(self.b15["trainX_fold0"][1][0], f'expected word {cc.df.iloc[1]["text1"].split(" ")[0]} to be unknown and map to 27. Got mapping of {self.b15["trainX_fold0"][1][0]} instead which mapped to {vocab[self.b15["trainX_fold0"][1][0]]}') 
    def test_baseline15_lastrec(self):
        self.assertTrue(self.b15["trainX_fold0"][-1] == [3,2,26,26], f'expected record "d c" to map to [3,2,26,26]. Instead {cc.df.iloc[-1]} mapped to {self.b15["trainX_fold0"][-1]}')
    def test_reverse5_train_length(self):
        self.assertTrue(list(map(len,self.rt5["trainX_fold0"])) == [5,9,18,20,20], f'lengths of the records were not correct. Expected [5,9,18,20,20], got {list(map(len,self.rt5["trainX_fold0"]))}.')
    def test_reverse5_1strec(self):
        self.assertTrue(self.rt5["trainX_fold0"][0] == [26,26,2,3,4], f'expected record "e d c" to map to [26,26,2,3,4]. Instead {cc.df.iloc[0]} mapped to {self.rt5["trainX_fold0"][0]}')
    def test_reverse15_train_length(self):
        self.assertTrue(list(map(len,self.rt15["trainX_fold0"])) == [5,9,18,20,20,7,14,20,3,12,20,16,20,19,4], f'lengths of the records were not correct. Expected [5,9,18,20,20,7,14,20,3,12,20,16,20,19,4], got {list(map(len,self.rt15["trainX_fold0"]))}.')
    def test_reverse15_lastrec(self):
        self.assertTrue(self.rt15["trainX_fold0"][-1] == [26,26,2,3], f'expected record "d c" to map to [26,26,2,3]. Instead {cc.df.iloc[-1]} mapped to {self.rt15["trainX_fold0"][-1]}')
    def test_removeunks5_train_length(self):
        self.assertTrue(list(map(len,self.ru5["trainX_fold0"])) == [5,7,15,18,20], f'lengths of the records were not correct. Expected [5,7,15,18,20], got {list(map(len,self.ru5["trainX_fold0"]))}.')
    def test_removeunks5_1strec(self):
        self.assertTrue(self.ru5["trainX_fold0"][0] == [4,3,2,26,26], f'expected record "e d c" to map to [4,3,2,26,26]. Instead {cc.df.iloc[0]} mapped to {self.ru5["trainX_fold0"][0]}')
    def test_removeunks15_train_length(self):
        self.assertTrue(list(map(len,self.ru15["trainX_fold0"])) == [5,7,15,18,20,6,14,20,3,11,19,16,20,19,4], f'lengths of the records were not correct. Expected [5,7,15,18,20,6,14,20,3,11,19,16,20,19,4], got {list(map(len,self.ru15["trainX_fold0"]))}.')
    def test_removeunks15_lastrec(self):
        self.assertTrue(self.ru15["trainX_fold0"][-1] == [3,2,26,26], f'expected record "d c" to map to [3,2,26,26]. Instead {cc.df.iloc[-1]} mapped to {self.ru15["trainX_fold0"][-1]}')
    def test_10length5_train_length(self):
        self.assertTrue(list(map(len,self.l5["trainX_fold0"])) == [5,9,10,10,10], f'lengths of the records were not correct. Expected [5,9,10,10,10], got {list(map(len,self.l5["trainX_fold0"]))}.')
    def test_10length5_1strec(self):
        self.assertTrue(self.l5["trainX_fold0"][0] == [4,3,2,26,26], f'expected record "e d c" to map to [4,3,2,26,26]. Instead {cc.df.iloc[0]} mapped to {self.l5["trainX_fold0"][0]}')
    def test_10length15_train_length(self):
        self.assertTrue(list(map(len,self.l15["trainX_fold0"])) == [5,9,10,10,10,7,10,10,3,10,10,10,10,10,4], f'lengths of the records were not correct. Expected [5,9,10,10,10,7,10,10,3,10,10,10,10,10,4], got {list(map(len,self.l15["trainX_fold0"]))}.')
    def test_10length15_lastrec(self):
        self.assertTrue(self.l15["trainX_fold0"][-1] == [3,2,26,26], f'expected record "d c" to map to [3,2,26,26]. Instead {cc.df.iloc[-1]} mapped to {self.l15["trainX_fold0"][-1]}')


TestResult = unittest.main(exit=False)
fail_num = len(TestResult.result.failures)
error_num = len(TestResult.result.errors)
test_num = TestResult.result.testsRun
lines = ['successes: ' + str(test_num - (fail_num + error_num)), 'failures: ' + str(fail_num), 'errors: ' + str(error_num), 'Total tests: ' + str(test_num)]
with open(os.path.join('documentation','test_results.md'),'w') as doc:
    doc.write('\n'.join(lines))
