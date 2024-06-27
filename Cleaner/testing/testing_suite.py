import unittest, sys, os, json, pandas as pd
os.chdir('..')
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from cleaner import Cleaner

args = {'remove_breaktoken':'none', 'remove_whitespace':False, 'remove_longword':False,
        'remove_punc':'none','convert_breaktoken':False, 'convert_escapecode':False,
        'convert_general':False,'lowercase':False, 'stem':False, 'fix_clocks':False}

class mockCacheClass():
    def __init__(self):
        self.saved_mod_args = {}
        cur_path = os.path.dirname(os.path.abspath(__file__))
        self.df = pd.read_pickle(os.path.join(cur_path,'testing','data','dfraw'))
        with open(os.path.join(cur_path,'testing','data','schema.json'),'r') as jd:
            self.schema = json.load(jd)
    def check_args(self, mod, args):
        return True
    def null_print(*string):
        pass

cc = mockCacheClass()
c = Cleaner(cc)

def run_clean(df):
    def cd_run(newdf):
        c.clean_data(cc.null_print, newdf, cc.schema, args)
        return c.pickle_files['df_cleaned']
    overTen = cd_run(df)
    underTen = cd_run(df[:3])
    cleaned = {}
    for i in range(13):
        temp_args = {k:v for k,v in args.items()}
        arg_key = df['textPathSuppReportsAddenda'].iat[i].split("|")[0]
        arg_value = True if "|" not in df['textPathSuppReportsAddenda'].iat[i] else df['textPathSuppReportsAddenda'].iat[i].split("|")[1]
        temp_args[arg_key] = arg_value
        samp_text = df['textPathComments'].iat[i]
        k = df['textPathSuppReportsAddenda'].iat[i]
        cleaned[df['textPathSuppReportsAddenda'].iat[i]] = c.clean_text(samp_text, temp_args, cc.null_print)
    c.describe('', args)
    return overTen, underTen, cleaned

########## Test Calls (general runs)

class TestCleanerInternal(unittest.TestCase):
    def test_reqmethods_builddataset(self):
        self.assertTrue('build_dataset' in Cleaner.__dict__.keys(), 'cannot find build_dataset in the script')
    def test_reqmethods_predsingle(self):
        self.assertTrue('predict_single' in Cleaner.__dict__.keys(), 'cannot find predict_single in the script')
    def test_reqmethods_predmulti(self):
        self.assertTrue('predict_multi' in Cleaner.__dict__.keys(), 'cannot find predict_multi in the script')
    def test_reqmethods_getcacheddeps(self):
        self.assertTrue('get_cached_deps' in Cleaner.__dict__.keys(), 'cannot find get_cached_deps in the script')
    def test_reqmethods_describe(self):
        self.assertTrue('describe' in Cleaner.__dict__.keys(), 'cannot find describe in the script')

########## Test Files (simi-class specific runs)

class TestCleanerFiles(unittest.TestCase):
    def test_onlyFiles(self):
        self.assertTrue(sorted([v for v in os.listdir() if '__' not in v and v[0]!='.']) == ['README.md','cleaner.py','documentation','testing'], 
               'there are files that should not be included in the module')

########## Test expected outputs (class specific)

class TestOutput(unittest.TestCase):
    
    def test_outputs_pickles(self):
        self.assertTrue(sorted(list(c.pickle_files.keys())) == ['df_cleaned'], f'expected pickle output is ["df_cleaned"]; got {sorted(list(c.pickle_files.keys()))}')
    def test_outputs_text(self):
        self.assertTrue(sorted(list(c.text_files.keys())) == ['describe'], f'expected text output is ["describe"]; got {sorted(list(c.text_files.keys()))}')
 
########## Test Module (class specific runs)
    
class TestParserInternal(unittest.TestCase):
    oT, uT, c = run_clean(cc.df)
    def test_runMoreThan10(self):
        self.assertTrue(isinstance(self.oT,pd.DataFrame), 'clean_data will not run multiprocessing')
    def test_runLessThan10(self):
        self.assertTrue(isinstance(self.uT,pd.DataFrame), 'clean_data will not run typical run')
    def test_runCleanBreakToken(self):
        self.assertTrue(self.c['convert_breaktoken']==' breaktoken  breaktoken  breaktoken ',
               'convert_breaktoken is not running as expected')
    def test_runConvertEscape(self):
        self.assertTrue(self.c['convert_escapecode']==' \x01  \xaa  \xff  \x0a  \x0d ', 
               'convert_escapecode is not working as expected')
    def test_runConvertGeneral(self):
        self.assertTrue(self.c['convert_general']==' floattoken   largeinttoken   (  (   )  )   *  *    ',
               'convert_general is not working as expected')
    def test_runFixClocks(self):
        self.assertTrue(self.c['fix_clocks']=='twentythreeoclock to oneoclock',
               'fix_clocks is not working as expected.')
    def test_runLowercase(self):
        self.assertTrue(self.c['lowercase']=='lowercase',
               'lowercase is not working as expected')
    def test_runRemoveBreaktoken_all(self):
        self.assertTrue(self.c['remove_breaktoken|all']=='',
               'remove_breaktoken (set to all) is not working as expected') 
    def test_runRemoveBreaktoken_dups(self):
        self.assertTrue(self.c['remove_breaktoken|dups']==' \n  \t  \r  breaktoken ',
               'remove_breaktoken (set to dups) is not working as expected') 
    def test_runRemoveLongword(self):
        self.assertTrue(self.c['remove_longword']=='',
               'remove_longword is not working as expected')
    def test_runRemovePunc_all(self):
        self.assertTrue(self.c['remove_punc|all']=='                          ',
               'remove_punc (set to all) is not working as expected')
    def test_runRemovePunc_dups(self):
        self.assertTrue(self.c['remove_punc|dups']==' <  >  ,  .  ?  /  :  ;  "  \'  {  [  }  ]  !  @  #  $  %  ^  &  *  (  )  _  -  +  =  ₹ ',
               'remove_punc (set to dups) is not working as expected')
    def test_runRemovePunc_most(self):
        self.assertTrue(self.c['remove_punc|most']=='      ',
               'remove_punc (set to most) is not working as expected')
    def test_runRemoveWhitespace(self):
        self.assertTrue(self.c['remove_whitespace']==' ',
               f'remove_whitespace is not working as expected  "{self.c["remove_whitespace"]}"')
    def test_runStem(self):
        self.assertTrue(self.c['stem']=='type type type',
               'stem is not working as expected')

########## Generate Outputs
       
TestResult = unittest.main(exit=False)
fail_num = len(TestResult.result.failures)
error_num = len(TestResult.result.errors)
test_num = TestResult.result.testsRun
lines = ['successes: ' + str(test_num - (fail_num + error_num)), 'failures: ' + str(fail_num), 'errors: ' + str(error_num), 'Total tests: ' + str(test_num)]
with open(os.path.join('documentation','test_results.md'),'w') as doc:
    doc.write('\n'.join(lines))
