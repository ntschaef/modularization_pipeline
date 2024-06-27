import unittest, sys, os, pandas as pd
os.chdir('../')
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from caching import CacheClass

def caching_test():
    cc = CacheClass(20)
    '''
    a test run of all the functions to ensure they work. This is a sample of how to run the scripts.
    '''
    p_pack = cc.build_package('Parser')
    cc.hash_package('Parser',p_pack) #generate the cache name
    ################### make needed changes (will be reset later)
    if 'CACHE_DIR' in os.environ:
        temp_cache_base = os.environ['CACHE_DIR']
    else: 
        temp_cache_base = None
    os.environ['CACHE_DIR'] = os.path.join('testing','test_cache')
    if not os.path.exists(os.environ['CACHE_DIR']):
        os.mkdir(os.environ['CACHE_DIR'])
    cc.saved_cache_hashes['Parser'] = 'test-dirty' #force the cache to be dirty
    pk_dict = {'pstring':'teststring', 'pdict':{'test':'testdict'}, 'plist':['testlist1','testlist2'], 'df':pd.DataFrame([[i for i in range(50000)]])}
    txt_dict = {'tstring':'teststring', 'tdict':{'test':'testdict'}, 'tlist':['testlist1','testlist2']}
    ################### continue with script
    cache_test_1 = cc.test_cache('Parser',del_dirty=True) # this will delete the dirty cache and identify it as bad
#    print('test_cache results1:', cache_good)
    build_and_protect = cc.build_cache('Parser', pk_dict, txt_dict) # build new cache based on the files above
    cache_test_2 = cc.test_cache('Parser') # this time the 'dirty' cache will pass
#    print('test_cache_results2:', cache_good)

    datadiff = []
    typediff = []
    # retrieve all files and test to see if they are identical to the original
    for (k,v) in {**pk_dict,**txt_dict}.items(): 
        new_file = cc.get_cache_file('Parser',k)
        if type(new_file) == pd.DataFrame:
            dfwronglist = v[0]!=new_file[0]
            if len(v[dfwronglist]) > 0:
                datadiff.append((k,v,new_file))
        else:
            new_file = cc.get_cache_file('Parser', k)
            if v != new_file:
                datadiff.append((k,v,new_file))
        if type(v) != type(new_file):
            typediff.append((k,v,new_file))
    ################### reset the environment
    if temp_cache_base is not None:
        os.environ['CACHE_DIR'] = temp_cache_base
    else:
        os.environ.pop('CACHE_DIR')

    # return test results
    return cache_test_1, build_and_protect, cache_test_2, datadiff, typediff

class TestCaching(unittest.TestCase):
    f_rm_dirt, t_protected, t_keep_dirt, datadiff, typediff = caching_test()
    def test_remove_dirty(self):
        self.assertFalse(self.f_rm_dirt, 'dirty cache was expected to be removed, but a cache was found.')
    def test_build_is_protected(self):
        self.assertTrue(self.t_protected, 'cache could not be protected.')
    def test_keep_dirty(self):
        self.assertTrue(self.t_keep_dirt, 'dirty cache was not kept.')
    def test_retrieve_pickle(self):
        self.assertTrue(self.datadiff == [], f'The following vaules mismatched when recieved from cache: {self.datadiff}')#{",".join(["|".join([t for t in o]) for o in self.err_load_pick])}')
    def test_retrieve_text(self):
        self.assertTrue(self.typediff == [], f'The following data types mismatched when recieved from cache: {self.typediff}')#{",".join(["|".join([t for t in o]) for o in self.err_load_text])}')
 
#class TestModules(unittest.TestCase):
#    def test_parserExists(self):
#        self.assertTrue(os.path.exists('Parser'), 'expected a "Parser" subrepo')
#    def test_cleanerExists(self):
#        self.assertTrue(os.path.exists('Cleaner'), 'expected a "Cleaner" subrepo')
#    def test_sanitizerExists(self):
#        self.assertTrue(os.path.exists('Sanitizer'), 'expected a "Sanitizer" subrepo')
#    def test_encoderExists(self):
#        self.assertTrue(os.path.exists('Encoder'), 'expected a "Encoder" subrepo')
#    def test_sequencerExists(self):
#        self.assertTrue(os.path.exists('Sequencer'), 'expected a "Sequencer" subrepo')
#    def test_modelSuiteExists(self):
#        self.assertTrue(os.path.exists('Model_Suite'), 'expected a "Model_Suite" subrepo')
#    def test_parserTestExists(self):
#        self.assertTrue(os.path.exists(os.path.join('Parser','testing_suite.py')), 'expected a testing suite in the "Parser" subrepo')
#    def test_cleanerTestExists(self):
#        self.assertTrue(os.path.exists(os.path.join('Cleaner','testing_suite.py')), 'expected a testing suite in the "Cleaner" subrepo')
#    def test_sanitizerTestExists(self):
#        self.assertTrue(os.path.exists(os.path.join('Sanitizer','testing_suite.py')), 'expected a testing suite in the "Sanitizer" subrepo')
#    def test_encoderTestExists(self):
#        self.assertTrue(os.path.exists(os.path.join('Encoder','testing_suite.py')), 'expected a testing suite in the "Encoder" subrepo')
#    def test_sequencerTestExists(self):
#        self.assertTrue(os.path.exists(os.path.join('Sequencer','testing_suite.py')), 'expected a testing suite in the "Sequencer" subrepo')
#    def test_modelSuiteTestExists(self):
#        self.assertTrue(os.path.exists(os.path.join('Model_Suite','testing_suite.py')), 'expected a testing suite in the "Model_Suite" subrepo')
#
#class TestTools(unittest.TestCase):
#    def test_runPipelineExists(self):
#        self.assertTrue(os.path.exists('run_pipeline.py'), 'expected the script "run_pipeline.py" to exist')
#    def test_cachingExists(self):
#        self.assertTrue(os.path.exists('caching.py'), 'expected the script "caching.py" to exist')
#    def test_pyEnvManagerExists(self):
#        self.assertTrue(os.path.exists('py_env_manager.py'), 'expected the script "py_env_manager.py" to exist')
#    def test_envArgManagerExists(self):
#        self.assertTrue(os.path.exists('env_args_manager.py'), 'expected the script "env_args_manager.py" to exist')
#    def test_argDistributionExists(self):
#        self.assertTrue(os.path.exists('arg_distribution.py'), 'expected the script "arg_distribution.py" to exist')
#    def test_quickPredictExists(self):
#        self.assertTrue(os.path.exists('quick_predict.py'), 'expected the script "quick_predict.py" to exist')
#    def test_documentationExists(self):
#        self.assertTrue(os.path.exists('documentation.py'), 'expected the script "documentation.py" to exist')
#    def test_scriptorExists(self):
#        self.assertTrue(os.path.exists('scriptor.py'), 'expected the script "scriptor.py" to exist')
#
#class TestDocumentation(unittest.TestCase):
#    def test_dirExists(self):
#        self.assertTrue(os.path.exists('documentation'), 'expected a documentation directory')
#    def test_currentCacheStructureExists(self):
#        self.assertTrue(os.path.exists(os.path.join('documentation','caching.json')), 'expected the cacheManager to document the cache mapping and dependencies in "caching.json"')
#    def test_pythonEnvironmentExists(self):
#        self.assertTrue(os.path.exists(os.path.join('documentation','environment.yml')), 'expected the pythonEnvironmentManager to document a single environment that does not conflict with the python packages used in each of the submodules.')
#    def test_systemEnvironmentExists(self):
#        self.assertTrue(os.path.exists(os.path.join('documentation','env_args.json')), 'expected the environmentalArgsManager to record the environmentalArgs and will not be lost')
#    def test_arguementDescripitonsExists(self):
#        self.assertTrue(os.path.exists(os.path.join('documentation','arg_description.yml')), 'expected a description of all the arguments expected to be passed into the submodules created by the argumentDescriptionManager')
#    def test_predictionsExist(self):
#        self.assertTrue(os.path.exists(os.path.join('documentation','predictions.csv')), 'expected a list of all the predictions and softmax scores of a test set')
#    def test_modelDescriptionExists(self):
#        self.assertTrue(os.path.exists(os.path.join('documentation','model_desc.md')), 'expected a description of the current model to be generated')
#    def test_modelScriptExists(self):
#        self.assertTrue(os.path.exists(os.path.join('documentation','model_script.py')), 'expected a script to be produced that will build the model from scratch')

   

TestResult = unittest.main(exit=False)
fail_num = len(TestResult.result.failures)
error_num = len(TestResult.result.errors)
test_num = TestResult.result.testsRun
lines = ['successes: ' + str(test_num - (fail_num + error_num)), 'failures: ' + str(fail_num),'errors: '+str(error_num), 'Total tests: '+str(test_num)]
with open(os.path.join('documentation','test_results.md'),'w') as doc:
    doc.write('\n'.join(lines))
