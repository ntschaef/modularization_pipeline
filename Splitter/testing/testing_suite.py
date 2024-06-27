import unittest, sys, os, json, pandas as pd, sqlite3
os.chdir('..')
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from splitter import Splitter

args = {'split_type':'case', 'by_registry':False, 'test_split':('percent', .1), 'val_split':('percent',.1), 'sklearn_random_seed':0}

class mockCacheClass():
    def __init__(self):
        self.saved_mod_args = {}
        cur_path = os.path.dirname(os.path.abspath(__file__))
        self.df = pd.read_pickle(os.path.join(cur_path,'testing','data','test_df'))
        with open(os.path.join(cur_path,'testing','data','test_schema.json'), 'rb') as jb:
            self.schema = json.load(jb)
    def check_args(self, mod, args):
        return True
    def null_print(*string):
        pass

cc = mockCacheClass()
s = Splitter(cc)

def test_ids():
    counts_c = s.split_data(cc.null_print, cc.null_print, cc.df, cc.schema, args)
    args['split_type'] = 'patient'
    counts_p = s.split_data(cc.null_print, cc.null_print, cc.df, cc.schema, args)
    args['split_type'] = 'record'
    counts_r = s.split_data(cc.null_print, cc.null_print, cc.df, cc.schema, args)
    args['by_registry'] = True
    _ = s.split_data(cc.null_print, cc.null_print, cc.df, cc.schema, args)
    is_eq_splits=[]
    for sp in ['train','val','test']:
        df = s.pickle_files[f'{sp}_fold0']
        is_eq_splits.append(len(df[df['registryId']=='reg0'])==len(df[df['registryId']=='reg1']))
    return counts_c, counts_p, counts_r, is_eq_splits[0], is_eq_splits[1], is_eq_splits[2],

def test_folds(folds): 
    uni_train = folds[0][0]
    uni_val = folds[0][1]
    uni_test = folds[0][2]
    dup_idx = []
    for i in range(len(folds)):
        if i > 0:
            uni_train = uni_train.append(folds[i][0])
            uni_val = uni_val.append(folds[i][1])
            uni_test = uni_test.append(folds[i][2])
        fullset = folds[i][0] + folds[i][1] + folds[i][2]
        dup_idx.append(len(fullset) == len(list(set(fullset))))
    uni_train = uni_train.drop_duplicates()
    uni_val = uni_val.drop_duplicates()
    uni_test = uni_test.drop_duplicates()
    return folds, uni_train, uni_val, uni_test, dup_idx
 
########## Test Calls (general runs)

class TestCleanerInternal(unittest.TestCase):
    def test_reqmethods_builddataset(self):
        self.assertTrue('build_dataset' in Splitter.__dict__.keys(), 'cannot find build_dataset in the script')
    def test_reqmethods_predsingle(self):
        self.assertTrue('predict_single' in Splitter.__dict__.keys(), 'cannot find predict_single in the script')
    def test_reqmethods_predmulti(self):
        self.assertTrue('predict_multi' in Splitter.__dict__.keys(), 'cannot find predict_mulit in the script')
    def test_reqmethods_getcacheddeps(self):
        self.assertTrue('get_cached_deps' in Splitter.__dict__.keys(), 'cannot find get_cached_deps in the script')
    def test_reqmethods_describe(self):
        self.assertTrue('describe' in Splitter.__dict__.keys(), 'cannot find describe in the script')

########## Test Files (simi-class specific runs)

class TestCleanerFiles(unittest.TestCase):
    def test_onlyFiles(self):
        self.assertTrue(sorted([v for v in os.listdir() if '__' not in v and v[0]!='.']) == ['README.md','documentation','splitter.py','testing'], 
               'there are files that should not be included in the module')

########## Test expected outputs (class specific)

class TestOutput(unittest.TestCase):
    counts, rids = s.split_data(cc.null_print, cc.null_print, cc.df, cc.schema, args)
    s.describe('', cc.schema, len(cc.df), rids, counts, args)
    def test_outputs_pickles(self):
        self.assertTrue(sorted(list(s.pickle_files.keys())) == ['test_fold0','train_fold0','val_fold0'], f'expected pickle output is ["test_fold0","train_fold0","val_fold0"]; got {sorted(list(s.pickle_files.keys()))}')
    def test_outputs_text(self):
        self.assertTrue(sorted(list(s.text_files.keys())) == ['describe','fold_num'], f'expected text output is ["describe", "fold_num"]; got {sorted(list(s.text_files.keys()))}')

########## Test Module (class specific runs)
    
class TestSplitterInternal(unittest.TestCase):
    counts_c, counts_p, counts_r, r_tr, r_v, r_te = test_ids()
    folds_tc, uTr_tc, uV_tc, uTe_tc, di_tc = test_folds(
             s.cv_split(5,0,cc.df[['patientId','tumorId','registryId']].drop_duplicates()))
    ytv, yt = s.test_year(2017, cc.df.groupby(['patientId', 'tumorId','registryId'])['pathDateSpecCollect1_year'].max().reset_index(),['patientId','tumorId','registryId'])
    rtv, rt = s.test_year(.15, cc.df.groupby(['patientId', 'tumorId','registryId'])['pathDateSpecCollect1_year'].max().reset_index().sort_values(['pathDateSpecCollect1_year']+['patientId', 'tumorId','registryId'],ascending=False),['patientId','tumorId','registryId'])
    ptv, pt = s.percent_split(.36, 0, cc.df[['patientId','tumorId','registryId']].drop_duplicates())
    folds_vc, uTr_vc, uV_vc, uTe_vc, di_vc = test_folds(
             s.cv_split(2,0,ytv,yt))
    folds_vp, uTr_vp, uV_vp, uTe_vp, di_vp = test_folds(
             s.percent_split(.32, 0, ytv, yt))

    def test_baseRun_case_counts(self):
        self.assertTrue(len(self.counts_c[0][0]) == 3, f'the length of the counts is off. Expected 3 got {len(self.counts_c[0][0])}')
    def test_baseRun_case_fields(self):
        self.assertTrue(sorted(self.counts_c[1]) == ['patientId','registryId','tumorId'], f'the incorrect fields were passed for "case".  Expected ["patientId","registryId","tumorId"], got {sorted(self.counts_c[1])}')
    def test_baseRun_patient_counts(self):
        self.assertTrue(len(self.counts_p[0][0]) == 3, f'the length of the counts is off. Expected 3 got {len(self.counts_p[0][0])}')
    def test_baseRun_patient_fields(self):
        self.assertTrue(sorted(self.counts_p[1]) == ['patientId','registryId'], f'the incorrect fields were passed for "patient".  Expected ["patientId","registryId"], got {sorted(self.counts_p[1])}')
    def test_baseRun_record_counts(self):
        self.assertTrue(len(self.counts_r[0][0]) == 3, f'the length of the counts is off. Expected 3 got {len(self.counts_r[0][0])}')
    def test_baseRun_record_fields(self):
        self.assertTrue(sorted(self.counts_r[1]) == ['recordDocumentId','registryId'], f'the incorrect fields were passed for "record".  Expected ["recordDocumentId","registryId"], got {sorted(self.counts_r[1])}')
    def test_registry_check_train(self):
        self.assertTrue(self.r_tr, 'the train split is not splitting the registries evenly when expected')
    def test_registry_check_train(self):
        self.assertTrue(self.r_v, 'the vailidation split is not splitting the registries evenly when expected')
    def test_registry_check_train(self):
        self.assertTrue(self.r_te, 'the test split is not splitting the registries evenly when expected')
    def test_test_cv_1of5_train(self):
        self.assertTrue(len(self.folds_tc[0][0]) == 60, f'length of train set for test_split[0] = "cv" option was wrong. Expected 60 got {len(self.folds_tc[0][0])}')
    def test_test_cv_1of5_val(self):
        self.assertTrue(len(self.folds_tc[0][1]) == 20, f'length of val set for test_split[0] = "cv" option was wrong. Expected 20 got {len(self.folds_tc[0][1])}')
    def test_test_cv_1of5_test(self):
        self.assertTrue(len(self.folds_tc[0][2]) == 20, f'length of train set for test_split[0] = "cv" option was wrong. Expected 20 got {len(self.folds_tc[0][2])}')
    def test_test_cv_2of5_train(self):
        self.assertTrue(len(self.folds_tc[1][0]) == 60, f'length of train set for test_split[0] = "cv" option was wrong. Expected 60 got {len(self.folds_tc[1][0])}')
    def test_test_cv_2of5_val(self):
        self.assertTrue(len(self.folds_tc[1][1]) == 20, f'length of val set for test_split[0] = "cv" option was wrong. Expected 20 got {len(self.folds_tc[1][1])}')
    def test_test_cv_2of5_test(self):
        self.assertTrue(len(self.folds_tc[1][2]) == 20, f'length of train set for test_split[0] = "cv" option was wrong. Expected 20 got {len(self.folds_tc[1][2])}')
    def test_test_cv_3of5_train(self):
        self.assertTrue(len(self.folds_tc[2][0]) == 60, f'length of train set for test_split[0] = "cv" option was wrong. Expected 60 got {len(self.folds_tc[2][0])}')
    def test_test_cv_3of5_val(self):
        self.assertTrue(len(self.folds_tc[2][1]) == 20, f'length of val set for test_split[0] = "cv" option was wrong. Expected 20 got {len(self.folds_tc[2][1])}')
    def test_test_cv_3of5_test(self):
        self.assertTrue(len(self.folds_tc[2][2]) == 20, f'length of train set for test_split[0] = "cv" option was wrong. Expected 20 got {len(self.folds_tc[2][2])}')
    def test_test_cv_4of5_train(self):
        self.assertTrue(len(self.folds_tc[3][0]) == 60, f'length of train set for test_split[0] = "cv" option was wrong. Expected 60 got {len(self.folds_tc[3][0])}')
    def test_test_cv_4of5_val(self):
        self.assertTrue(len(self.folds_tc[3][1]) == 20, f'length of val set for test_split[0] = "cv" option was wrong. Expected 20 got {len(self.folds_tc[3][1])}')
    def test_test_cv_4of5_test(self):
        self.assertTrue(len(self.folds_tc[3][2]) == 20, f'length of train set for test_split[0] = "cv" option was wrong. Expected 20 got {len(self.folds_tc[3][2])}')
    def test_test_cv_5of5_train(self):
        self.assertTrue(len(self.folds_tc[4][0]) == 60, f'length of train set for test_split[0] = "cv" option was wrong. Expected 60 got {len(self.folds_tc[4][0])}')
    def test_test_cv_5of5_val(self):
        self.assertTrue(len(self.folds_tc[4][1]) == 20, f'length of val set for test_split[0] = "cv" option was wrong. Expected 20 got {len(self.folds_tc[4][1])}')
    def test_test_cv_5of5_test(self):
        self.assertTrue(len(self.folds_tc[4][2]) == 20, f'length of train set for test_split[0] = "cv" option was wrong. Expected 20 got {len(self.folds_tc[4][2])}')
    def test_test_cv_coverage_folds(self):
        self.assertTrue(self.di_tc == [0,0,0,0,0], f'duplicates were found between the train, val, and test sets of the following folds: {self.di_tc}')
    def test_test_cv_coverate_train(self):
        self.assertTrue(len(self.uTr_tc) == 100, f'there are {100 - len(self.uTr_tc)} that were ignored in the train sets.') 
    def test_test_cv_coverate_val(self):
        self.assertTrue(len(self.uV_tc) == 100, f'there are {100 - len(self.uV_tc)} that were ignored in the validation sets.') 
    def test_test_cv_coverate_test(self):
        self.assertTrue(len(self.uTe_tc) == 100, f'there are {100 - len(self.uTe_tc)} that were ignored in the test sets.') 
    def test_test_year_count_test(self):
        self.assertTrue(len(self.yt) == 36, f'there should be 36 records in the year test set. Got {len(self.yt)}')
    def test_test_year_count_trainval(self):
        self.assertTrue(len(self.ytv) == 64, f'there should be 64 records in the year trainval set. Got {len(self.ytv)}')
    def test_test_recent_count_test(self):
        self.assertTrue(len(self.rt) == 15, f'there should be 15 records in the recent test set. Got {len(self.rt)}')
    def test_test_recent_count_trainval(self):
        self.assertTrue(len(self.rtv) == 85, f'there should be 85 records in the recent trainval set. Got {len(self.rt)}')
    def test_test_percent_count_test(self):
        self.assertTrue(len(self.pt) == 36, f'there should be 36 records in the percent test set. Got {len(self.pt)}')
    def test_test_percent_count_trainval(self):
        self.assertTrue(len(self.ptv) == 64, f'there should be 64 records in the percent trainval set. Got {len(self.ptv)}')
    def test_test_cv_1of2_train(self):
        self.assertTrue(len(self.folds_vc[0][0]) == 32, f'length of train set for val_split[0] = "cv" option was wrong. Expected 32 got {len(self.folds_vc[0][0])}')
    def test_test_cv_1of2_val(self):
        self.assertTrue(len(self.folds_vc[0][1]) == 32, f'length of val set for val_split[0] = "cv" option was wrong. Expected 32 got {len(self.folds_vc[0][1])}')
    def test_test_cv_1of2_test(self):
        self.assertTrue(len(self.folds_vc[0][2]) == 36, f'length of test set for val_split[0] = "cv" option was wrong. Expected 36 got {len(self.folds_vc[0][2])}')
    def test_test_cv_2of2_train(self):
        self.assertTrue(len(self.folds_vc[1][0]) == 32, f'length of train set for val_split[0] = "cv" option was wrong. Expected 32 got {len(self.folds_vc[1][0])}')
    def test_test_cv_2of2_val(self):
        self.assertTrue(len(self.folds_vc[1][1]) == 32, f'length of val set for val_split[0] = "cv" option was wrong. Expected 32 got {len(self.folds_vc[1][1])}')
    def test_test_cv_2of2_test(self):
        self.assertTrue(len(self.folds_vc[1][2]) == 36, f'length of test set for val_split[0] = "cv" option was wrong. Expected 36 got {len(self.folds_vc[1][2])}')
    def test_test_cv_coverage_folds(self):
        self.assertTrue(self.di_vc == [0,0], f'duplicates were found between the train, val, and test sets of the following folds: {self.di_vc}')
    def test_test_cv_coverate_train(self):
        self.assertTrue(len(self.uTr_vc) == 64, f'there are {64 - len(self.uTr_vc)} that were ignored in the train sets.') 
    def test_test_cv_coverate_val(self):
        self.assertTrue(len(self.uV_vc) == 64, f'there are {64 - len(self.uV_vc)} that were ignored in the validation sets.') 
    def test_test_cv_coverate_test(self):
        self.assertTrue(len(self.uTe_vc) == 36, f'there are {36 - len(self.uTe_vc)} that were ignored in the test sets.') 
    def test_test_pecent_train(self):
        self.assertTrue(len(self.folds_vp[0][0]) == 32, f'length of train set for val_split[0] = "percent" option was wrong. Expected 32 got {len(self.folds_vp[0][0])}')
    def test_test_percent_val(self):
        self.assertTrue(len(self.folds_vp[0][1]) == 32, f'length of val set for val_split[0] = "percent" option was wrong. Expected 32 got {len(self.folds_vp[0][1])}')
    def test_test_percent_test(self):
        self.assertTrue(len(self.folds_vp[0][2]) == 36, f'length of test set for val_split[0] = "percent" option was wrong. Expected 36 got {len(self.folds_vp[0][2])}')
    def test_test_percent_coverage_folds(self):
        self.assertTrue(self.di_vp == [0], f'duplicates were found between the train, val, and test sets of the following folds: {self.di_vp}')

TestResult = unittest.main(exit=False)
fail_num = len(TestResult.result.failures)
error_num = len(TestResult.result.errors)
test_num = TestResult.result.testsRun
lines = ['successes: ' + str(test_num - (fail_num + error_num)), 'failures: ' + str(fail_num), 'errors: ' + str(error_num), 'Total tests: ' + str(test_num)]
with open(os.path.join('documentation','test_results.md'),'w') as doc:
    doc.write('\n'.join(lines))
