import unittest, sys, os, json, pandas as pd, sqlite3
os.chdir('..')
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from filter import Filter

args = {'tasks': [], 'only_single': 'none', 'exclude_field_scores': {}, 
        'window_days': [0], 'window_fields': [], 'min_year': 0}

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
s = Filter(cc)

########## Test Calls (general runs)

class TestCleanerInternal(unittest.TestCase):
    def test_reqmethods_builddataset(self):
        self.assertTrue('build_dataset' in Filter.__dict__.keys(), 'cannot find build_dataset in the script')
    def test_reqmethods_predsingle(self):
        self.assertTrue('predict_single' in Filter.__dict__.keys(), 'cannot find predict_single in the script')
    def test_reqmethods_predmulti(self):
        self.assertTrue('predict_multi' in Filter.__dict__.keys(), 'cannot find predict_mulit in the script')
    def test_reqmethods_getcacheddeps(self):
        self.assertTrue('get_cached_deps' in Filter.__dict__.keys(), 'cannot find get_cached_deps in the script')
    def test_reqmethods_describe(self):
        self.assertTrue('describe' in Filter.__dict__.keys(), 'cannot find describe in the script')

########## Test Files (simi-class specific runs)

class TestCleanerFiles(unittest.TestCase):
    def test_onlyFiles(self):
        self.assertTrue(sorted([v for v in os.listdir() if '__' not in v and v[0]!='.']) == ['README.md','documentation','filter.py','testing'], 
               'there are files that should not be included in the module')

########## Test expected outputs (class specific)

class TestOutput(unittest.TestCase):
    _ = s.filter_data(cc.null_print, cc.df, cc.schema, args, 'pathDateSpecCollect1')
    dups_picks = sorted(list(s.pickle_files.keys()))
    dups_texts = sorted(list(s.text_files.keys()))
    cc.df=cc.df[:-1]
    counts = s.filter_data(cc.null_print, cc.df, cc.schema, args, 'pathDateSpecCollect1')
    s.describe('', counts, args, 'pathDateSpecCollect1')
    nd_picks = sorted(list(s.pickle_files.keys()))
    nd_texts = sorted(list(s.text_files.keys()))
    def test_dups_outputs_pickles(self):
        self.assertTrue(self.dups_picks == ['filtered_ids'], f'expected pickle output is ["filtered_ids"]; got {self.dups_picks}')
    def test_dups_outputs_text(self):
        self.assertTrue(self.dups_texts == [], f'expected text output is []; got {self.dups_texts}')
    def test_nodups_outputs_pickles(self):
        self.assertTrue(self.nd_picks == ['filtered_ids'], f'expected pickle output is ["filtered_ids"]; got {self.nd_picks}')
    def test_nodups_outputs_text(self):
        self.assertTrue(self.nd_texts == ['describe'], f'expected text output is ["describe"]; got {self.nd_texts}')
 
########## Test Module (class specific runs)
    
class TestFilterInternal(unittest.TestCase):
    counts = s.filter_data(cc.null_print, cc.df, cc.schema, args, 'pathDateSpecCollect1')
    t_s_d, t_s_n = s.filter_gold(cc.null_print, cc.df, ['behavior'])
    t_m_d, t_m_n = s.filter_gold(cc.null_print, cc.df, ['behavior','grade'])
    t_c_d, t_c_n = s.filter_gold(cc.null_print, cc.df, ['behavior+grade'])
    s_p_d, s_p_n = s.filter_by_single(cc.null_print, cc.df, 'patient', cc.schema)
    s_c_d, s_c_n = s.filter_by_single(cc.null_print, cc.df, 'case', cc.schema)
    rf_v_d, rf_v_n = s.filter_by_field(cc.null_print, cc.df, {'first_filter': ["1","2"]})
    rf_r_d, rf_r_n = s.filter_by_field(cc.null_print, cc.df, {'first_filter': ["<=2"]})
    rf_b_d, rf_b_n = s.filter_by_field(cc.null_print, cc.df, {'first_filter': ["<2","3"]})
    rf_m_d, rf_m_n = s.filter_by_field(cc.null_print, cc.df, {'first_filter': [">=2"], 'second_filter':["a1","c3"]})
    w_s_d, w_s_n = s.filter_by_window(cc.null_print, cc.df, ([10], ['dateOfDiagnosis']))
    w_d_d, w_d_n = s.filter_by_window(cc.null_print, cc.df, ([-9,8], ['dateOfDiagnosis']))
    w_m_d, w_m_n = s.filter_by_window(cc.null_print, cc.df, ([4], ['dateOfDiagnosis','dateOfSurg']))
    y_d, y_n = s.filter_by_early_date(cc.null_print, cc.df, 2010, "pathDateSpecCollect1")

    def test_baseRun_data(self):
        self.assertTrue(s.pickle_files['filtered_ids'].equals(cc.df[['recordDocumentId','registryId']]), 'the base run data is not correct')
    def test_baseRun_count(self):
        self.assertTrue(self.counts == (0,0,0,0,0), f'counts for the base run are incorrect, {self.counts}')
    def test_taskFilter_single_data(self):
        self.assertTrue(self.t_s_d.equals(cc.df[:2].append(cc.df[3:])), 'single task filtered incorrectly')
    def test_taskFilter_single_count(self):
        self.assertTrue(self.t_s_n == 1, 'single task filtered the wrong number')
    def test_taskFilter_multi_data(self):
        self.assertTrue(self.t_m_d.equals(cc.df[3:]), 'multi task filtered incorrectly')
    def test_taskFilter_multi_count(self):
        self.assertTrue(self.t_m_n == 3, 'multi task filtered the wrong number')
    def test_taskFilter_combined_data(self):
        self.assertTrue(self.t_c_d.equals(cc.df[3:]), 'combined task filtered incorrectly')
    def test_taskFilter_combined_count(self):
        self.assertTrue(self.t_c_n == 3, 'combined task filtered the wrong number')
    def test_single_patient_data(self):
        self.assertTrue(self.s_p_d.equals(cc.df[3:]), 'single patient filtered incorrectly')
    def test_single_patient_count(self):
        self.assertTrue(self.s_p_n == 3, 'single patient filtered the wrong number')
    def test_single_case_data(self):
        self.assertTrue(self.s_p_d.equals(cc.df[3:]), 'single patient filtered incorrectly')
    def test_single_case_count(self):
        self.assertTrue(self.s_p_n == 3, 'single patient filtered the wrong number')
    def test_restrictFields_values_data(self):
        self.assertTrue(self.rf_v_d.equals(cc.df[:2]), 'restricted fields values filtered incorrectly')
    def test_restrictFields_values_count(self):
        self.assertTrue(self.rf_v_n == 2, 'restricted fields values filtered the wrong number')
    def test_restrictFields_ranges_data(self):
        self.assertTrue(self.rf_r_d.equals(cc.df[:2]), 'restricted fields ranges filtered incorrectly')
    def test_restrictFields_ranges_count(self):
        self.assertTrue(self.rf_r_n == 2, 'restricted fields ranges filtered the wrong number')
    def test_restrictFields_both_data(self):
        self.assertTrue(self.rf_b_d.equals(cc.df[:1].append(cc.df[2:3])), 'restricted fields both filtered incorrectly')
    def test_restrictFields_both_count(self):
        self.assertTrue(self.rf_b_n == 2, 'restricted fields both filtered the wrong number')
    def test_restrictFields_multiple_data(self):
        self.assertTrue(self.rf_m_d.equals(cc.df[2:3]), 'restricted fields multiple fields filtered incorrectly')
    def test_restrictFields_multiple_count(self):
        self.assertTrue(self.rf_m_n == 3, 'restricted fields multiple fileds filtered the wrong number')
    def test_window_singleInt_data(self):
        self.assertTrue(self.w_s_d.equals(cc.df[:3]), 'window single number filtered incorrectly')
    def test_window_singleInt_count(self):
        self.assertTrue(self.w_s_n == 1, 'window single number filtered the wrong number of records')
    def test_window_diffInt_data(self):
        self.assertTrue(self.w_d_d.equals(cc.df[:2]), 'window multiple numbers filtered incorrectly')
    def test_window_diffInt_count(self):
        self.assertTrue(self.w_d_n == 2, 'window multiple numbers filtered the wrong number of records')
    def test_window_multiDate_data(self):
        self.assertTrue(self.w_m_d.equals(cc.df[:2].append(cc.df[3:])), 'restricted fields ranges filtered incorrectly')
    def test_window_multiDate_count(self):
        self.assertTrue(self.w_m_n == 1, 'restricted fields ranges filtered the wrong number')
    def test_min_year_data(self):
        self.assertTrue(self.y_d.equals(cc.df[2:]), 'starting year filtered incorrectly')
    def test_min_year_count(self):
        self.assertTrue(self.y_n == 2, 'starting year filtered the wrong number')

TestResult = unittest.main(exit=False)
fail_num = len(TestResult.result.failures)
error_num = len(TestResult.result.errors)
test_num = TestResult.result.testsRun
lines = ['successes: ' + str(test_num - (fail_num + error_num)), 'failures: ' + str(fail_num), 'errors: ' + str(error_num), 'Total tests: ' + str(test_num)]
with open(os.path.join('documentation','test_results.md'),'w') as doc:
    doc.write('\n'.join(lines))
