# basic setup of imports
import hashlib, argparse, os, shutil, logging, pickle, json, pandas as pd, numpy as np
from contextlib import contextmanager
from datetime import datetime
dirpath = os.path.split(os.path.abspath(__file__))[0]

class CacheClass():
    '''
    This class will create a unique repeatable cache directory for the modules passed through.

    Attributes:
        logging (package): a way to pass logging to all the other classes.
        saved_cache_hashes (dict): strings of the hashes that have been created.
        saved_cache_paths (dict): strings of the unique paths created for each module.
        saved_packages (dict): lists of the packages used to create the hashes.
        saved_mod_args (dict): a holding area for all the modules model_args, where it resides and it's description.
                               each value has the form <var_name: (var_value, home_module, var_desciription)>.
        deps (dict): lists of predefined properties needed to ensure the cache is duplicated 
                     and unique. Only the entries listed will be used to create the cache directory.
    '''

    def __init__(self, verb=20):
        '''
        Initializer of the CacheClass. Defines the "attributes".

        Parameters:
            verb (int): sets the logging level of the script.
        '''
        logging.root.setLevel(verb)

        ######################### The next section is needed to manage the pipeline ########################

        # define the structure of the pipeline from the deps.json file
        ## dependencies take the form <module name>: [[<modules>],[<system_args>],[<module_args>],[<packages>]]
        with open(os.path.join(dirpath,'deps.json'), 'r') as jd:
            deps = json.load(jd)
        ## parse for the cacher and ensure ordering
        self.deps = {k:[sorted(v['modules']), 
                        sorted(v['env_vars']),
                        sorted(v['mod_args']),
                        sorted(v['py_packs'])] for k,v in deps.items()}       

        ######################  everything below should be independent of the pipeline structure ###########

        self.describe = ''
        self.saved_packages = {mod:"" for mod in self.deps.keys()}
        self.saved_cache_hashes = {mod:"" for mod in self.deps.keys()}
        self.saved_cache_paths = {mod:"" for mod in self.deps.keys()}
        self.saved_mod_args = {}
        self.logging = logging

        # set the base cache call if it is not already set
        if 'CACHE_DIR' not in os.environ:
            
            if os.path.isfile(os.path.join(dirpath,'data_args.json')):
                with open(os.path.join(dirpath,'data_args.json'),'r') as jd:
                    os.environ['CACHE_DIR'] = json.load(jd)['Pipeline']['cache_dir']
            else:
                os.environ['CACHE_DIR']='/mnt/nci/scratch/api_cache/'
        
        if 'CACHE_CHECK_MD5' not in os.environ.keys():
            self.md5_check = True
        else:
            self.md5_check = os.environ['CACHE_CHECK_MD5'].lower() not in ['false','no','0']

    @contextmanager 
    def timing(self, msg, print_after=False):
        '''
        will print out the time it takes to run a "with" statement

        Parameters:
            msg (string): message to be report with the timing
            print_after (boolean): will the message be reported only at the end of the run?
        '''
        start = datetime.now()
        try:
            if not print_after:
                print('    ' + msg.ljust(20) + ' ... ', end='', flush=True)
            yield start
        finally:
            end = datetime.now()
            diff = end - start
            if print_after:
                print(f'{msg} took {diff}')
            else:
                print(f'done {diff}')

    def mod_args(self, key):
        '''
        returns the value saved for the model args and ignores the module and description

        Parameters:
            key (string): name of the model arg desired.

        Returns:
            various data: first item stored in the saved_model_args value. None if key doesn't exist.
        '''
        return self.saved_mod_args[key][0] if key in self.saved_mod_args.keys() else None

    def all_mod_args(self, mod=None):
        '''
        returns all value saved for the model args and ignores the module and description
        
        Returns:
            dict: all keys and first item stored in the saved_model_args value.
        '''
        return {k:v[0] for k,v in self.saved_mod_args.items()}

    def hash_package(self, mod_name, mod_package, production=[False]):
        '''
        Verifies the package is correct and adjusts the provided input to the ordered format.

        Parameters:
            mod_name (string): name of the module that will be used to call the 'deps' value.
            mod_package (dict): all the modules predefined information needed create the cache.
                                This should match the relative schema in the deps.
                                It will have the form [list, list, dict, dict]
        Variables:
            mmods (list): all the module hashes that are expected in deps, but not yet generated
            env_adj (list): a list of all the environmental variables listed in the mod_package refined to the schema in deps.
            modhash_adj (list): a list of all the module hashes expected from deps.
            modargs_adj (list): a list of all the model arg values in the mod_package in order of what is listed in deps.
            mod_pack_adj (dict): refined and fomalized mod_package based on the schema of deps.
        '''
        assert mod_name in self.deps.keys(), f'{mod_name} is not a known module.' # check to see that the 'mod_name' is expected
        
        # check to see that each list in the package were expected 
        for i,l in enumerate(self.deps[mod_name]):
            mis_entry = [v for v in l if v not in mod_package[i]] 
            assert mis_entry == [], f'the entries {",".join(mis_entry)} were missing from index {i} of {mod_name}. May need to update data_args.json'
        # check to see that the caches were built
        mis_entry = [m for m in self.deps[mod_name][0] if self.saved_cache_hashes[m] == ""]
        assert mis_entry == [], f'{",".join(mis_entry)} do not have caches generated'

        # check to see that the git module exists
        import git
        logging.debug(os.path.join(dirpath, mod_name))
        assert os.path.exists(os.path.join(dirpath, mod_name)), f'{mod_name} does not exist. Current directory is {dirpath}.'
            

        # since everything is built correct, format them into the expected package for hashing
        modhash_adj = [str(self.saved_cache_hashes[m]) for m in self.deps[mod_name][0]]
        env_adj = [str(os.environ[ev]) if ev in os.environ else '' for ev in self.deps[mod_name][1]]
        modargs_adj = [sorted(mod_package[2][arg], key=lambda ele: (0,ele) if str(ele)==ele else (1,ele)) 
                              if isinstance(mod_package[2][arg], list) else 
                              mod_package[2][arg] for arg in self.deps[mod_name][2]]
        pack_vers = [str(mod_package[3][pack]) for pack in self.deps[mod_name][3]]
        if production[0]:
            repo_hex = production[1][mod_name]
            repo_dirty = False
        else:
            repo = git.Repo(os.path.join(dirpath,mod_name))
            repo_hex = repo.head.commit.hexsha
            repo_dirty = repo.is_dirty()
        mod_pack_adj = [modhash_adj, env_adj, modargs_adj, pack_vers, repo_hex]
        logging.debug(f'mod_pack_adj: {mod_pack_adj}')

        # create the custom hash
        hash_name = self.create_cache_hash([modhash_adj, env_adj, modargs_adj, repo_hex])

        # save the states for later
        self.saved_cache_hashes[mod_name] = mod_name + hash_name + f'{"-dirty" if repo_dirty or ([v for v in mod_package[0].values() if "-dirty" in v] != []) else ""}'
        self.saved_packages[mod_name] = mod_pack_adj
        logging.debug(f'cache path for {mod_name} is {self.saved_cache_hashes[mod_name]}')

    def create_cache_hash(self, mod_pack_adj):
        '''
        Creates and stores the unique and repeatable name of the cache for the module.

        Parameters:
            mod_pack_adj (dict): refined and fomalized mod_package based on the schema of deps.

        Returns:
            String: the hash of the cache based on the refined package passed in    
        '''
        import hashlib
        # the package [['subhash1','subhash2'],['env_args1'], [], 'commit'] will become
        #             'subhash1^subhash2|env_args1||commit' then hashed as
        #             'b6352f028abd0636bd1c6d080867f812'
        def listconv(v):
            if isinstance(v, (list,tuple)):
                v = '|'.join(sorted([listconv(v1) for v1 in v]))
            else:
                v = str(v)
            return v
 
        return hashlib.md5(listconv(mod_pack_adj).encode('utf-8')).hexdigest()

    def remove_cache_dir(self, fp):
        '''
        Remove cache directory

        Parameters:
            fp (string): full directory path of the cache to be removed.
        '''
        # change the permissions of the file to make them removable. 
        # try is included because this will crash on wsl systems
        try:
            for f in os.listdir(fp):
                os.chmod(os.path.join(fp,f),0o700)
            os.chmod(fp,0o700)
        except:
            logging.warning('cannot change permissions on a wls system')
        shutil.rmtree(fp) #remove the directory and all contents
        logging.info(f'cache {fp} has been removed.')

    def rm_dirty_cache(self, del_dirty, fp):
        '''
        Remove dirty cache if it exists and is appropriate
        
        Parameters:
            del_dirty (bool): the answer to "should a dirty cache be deleted and rebuilt?"
            fp (string): full directory path of the cache to be removed.
        '''
        if del_dirty:
            if os.path.exists(fp):
                if fp[-6:] == '-dirty':
                    logging.debug(f'cache {fp} is dirty and will be removed.')
                    self.remove_cache_dir(fp)

   
    def build_package(self, mod_name, mod_args_dict={}):
        '''
        Uses the pipeline to create the a package to be hashed

        Parameters:
            mod_name (string): name of the module that will be used to call the stored values.
        
        Returns:
             list: all the modules predefined information needed create the cache.
                   This should match the relative schema in the deps.
                   It will have the form [dict, dict, dict, dict]
        '''
        # Reminder that dependencies take the form <module name>: [[<modules>],[<system_args>],[<module_args>],[<packages>]]
        mod_hash_list = []
        for pre_mod in self.deps[mod_name][0]:
            if self.saved_cache_hashes[pre_mod] == "":
                mod_pack = self.build_package(pre_mod, mod_args_dict=mod_args_dict) #designed to build the hashes retroactively if needed.
                self.hash_package(pre_mod, mod_pack)

        mod_hash_list = {k:v for k,v in self.saved_cache_hashes.items() if k in self.deps[mod_name][0]}

        sysargs_list = {k:os.environ[k] if k in os.environ.keys() else "" for k in self.deps[mod_name][1]}

        if mod_args_dict=={}:
            assert os.path.isfile(os.path.join(dirpath,'data_args.json')), 'data_args.json file is missing.'
            with open(os.path.join(dirpath,'data_args.json'),'r') as jd:
                raw_config = json.load(jd)
            config = {k1:v1 for v in raw_config.values() for k1,v1 in v.items()}
            mod_args_dict = {k:v for k,v in config.items() if k in self.deps[mod_name][2]}
        else:
            mod_args_dict = {k:v for k,v in mod_args_dict.items() if k in self.deps[mod_name][2]}
        
        pypack_dict = {}
        for p in self.deps[mod_name][3]:
            exec(f'import {p}')
            try:
                pypack_dict[p] = eval(f'{p}.__version__')
            except:
                pypack_dict[p] = eval(f'{p}.version')
        
        return [mod_hash_list, sysargs_list, mod_args_dict, pypack_dict]

    def hashfile(self, file):
        '''
        quick script used multiple places to check hashes of files
        
        Parameters:
            file (string): full path of the file that needs to be hashed
            
        Returns:
            string: md5 hash of the file
        '''
        hf = hashlib.md5()
        with open(file, 'rb') as fb:
            for block in iter(lambda: fb.read(65536), b''):
                hf.update(block)
        return hf.hexdigest()
        
    def finalize_cache(self, mod_name):
        '''
        finalize the cache by creating an md5 file that will be used to capture the md5sums and making it read only.
        
        Parameters:
            mod_name (string): name of the module that will be used to call the stored values.
 
        Variable: 
            fp (string): the unique and repeatable relative fullpath cache directory created for the module.

        Returns:
            boolean: Was the cache protected?
        '''
        fp = os.path.join(dirpath,os.environ['CACHE_DIR'],self.saved_cache_hashes[mod_name])
        with open(os.path.join(fp, 'deps_package'),'w+') as jd:
            jd.write('\n'.join([str(o) for o in self.saved_packages[mod_name]]))

        cache_files = sorted(os.listdir(fp)) # identify all the files in the cache
        with open(os.path.join(fp, 'filled_files.md5'),'w') as md5d:
            hashes = []
            for f in cache_files:
                fpf = os.path.join(fp,f)
                logging.debug(f'hashing file {fpf}') 
                hf = self.hashfile(fpf)
                hashes.append(fpf+" "+hf)
            logging.debug('hashes returned are', ' '.join(hashes))
            md5d.write('\n'.join(hashes))
        # make all the files r+x
        # try is included because it will crash a wsl system
        try:
            for f in os.listdir(fp):
                os.chmod(os.path.join(fp,f),0o500)
            os.chmod(fp, 0o500)
        except:
            logging.warning(f'could not protect cache {fp}') 
            return False 
        logging.info(f'cache {fp} successfully created')
        return True

    def check_pypack(self, mod_name):
        '''
        Check python package to see if it matches what was saved in the cached directory

        Parameters:
            mod_name (strin): name of the module that will be used to call the stored values.
        '''
        fp = os.path.join(dirpath,os.environ['CACHE_DIR'],self.saved_cache_hashes[mod_name])
        with open(os.path.join(fp,'deps_package')) as fd:
            saved_pack = fd.readlines()[3][1:-2].replace("'","").replace(' ','').split(',')
        for i,v in enumerate(self.deps[mod_name][3]):
            if saved_pack[i]!=self.saved_packages[mod_name][3][i]:
                print(f'package {v} version mismatches: {saved_pack[i]} vs {self.saved_packages[mod_name][3][i]}')

    def test_cache(self, mod_name, del_dirty=False, mod_args={}):
        
        '''        
        Check to see if the cache is fully built.

        Parameters:
            mod_name (string): name of the module that will be used to call the stored values.
            del_dirty (bool): identifies if dirty caches should be removed.

        Variables:
            fp (string): the unique and repeatable relative fullpath cache directory created for the module.

        Note: this will use the environmental variable CACHE_DIR to call the cache root directory.
              the default of this is '/mtn/nci/scratch/api_cache/'

        Returns:
            Bool: indication if the cache exists and is ready to use.
            fp (string): the unique and repeatable full cache directory created for the module.
        '''
        assert mod_name in self.deps.keys(), f'{mod_name} is an unknown module'

        if self.saved_cache_paths[mod_name] != "": # this is the only place that this is captured. If it is set then it has been tested already.
            return True

        # build the hashes if they don't already exist
        if self.saved_cache_hashes[mod_name] == "":
            mod_package = self.build_package(mod_name, mod_args_dict=mod_args)
            # this is to ensure that the hash can be built
            deps_flag = True
            for i in range(len(self.deps[mod_name])):
                if self.deps[mod_name][i] != sorted(list(mod_package[i].keys())):
                    deps_flag = False
            if deps_flag and os.path.exists(os.path.join(dirpath,mod_name)):
                logging.debug(mod_package)
                if 'commits' in mod_args:
                    self.hash_package(mod_name, mod_package, [True,mod_args['commits']])
                else:
                    self.hash_package(mod_name, mod_package)
            else:
                logging.debug(f'{mod_name} depends on a module that is not yet built')
                return False
        fp = os.path.join(dirpath,os.environ['CACHE_DIR'],self.saved_cache_hashes[mod_name])
        logging.info(f'checking {mod_name} cache at {fp}')
        self.rm_dirty_cache(del_dirty, fp) # remove dirty cache if requested

        if not os.path.exists(fp): # check to make sure that the cache exists
            logging.debug(f'the directory {fp} does not exist')
            return False

        elif not os.path.exists(os.path.join(fp, 'filled_files.md5')):
            logging.debug(f'the file filled_files.md5 does not exist in cache {fp}.') 
            self.remove_cache_dir(fp)
            return False

        if self.md5_check:
            logging.debug(f'cache {fp} was fully built. Checking hashes') 
            # check to ensure the hashes match what they should be
            hash = hashlib.md5()
            with open(os.path.join(fp, 'filled_files.md5'), 'r') as md5d:
                hashes = md5d.readlines()
            for h in hashes:
                f,fh = h.split()
                f = os.path.join(fp,os.path.split(f)[1])
                logging.debug(f'checking hashing of file {f}')
                hf = self.hashfile(f)
                logging.debug(f'hash for {f} is {hf}. Expected {fh}.')
                if hf!=fh:
                    logging.warning(f'{f} did not retrieve the correct hash. Removing cache {fp}')
                    self.remove_cache_dir(fp)
                    return False
                
        # everything checks out
        self.saved_cache_paths[mod_name] = fp
        self.check_pypack(mod_name)
        return True

    def build_cache(self, mod_name, pick_files, text_files, mod_args={}):
        '''
        fill the cache with the files needed in other modules.
       
        Parameters:
            mod_name (string): name of the module that will be used to call the stored values.
            pick_files (list): all the datasets that are pickle appropriate (list, dictionaries and dataframes).
            text_files (list): all the datasets that will be exported to a text document.

        Variables:
            fp (string): the unique and repeatable relative fullpath cache directory created for the module.

            Note: this function uses the environmental variable 'CACHE_DIR' to build the path.
        Returns
            boolean: was the cache protected?
        '''
        if self.saved_cache_hashes[mod_name] == "":
            mod_package = self.build_package(mod_name, mod_args_dict=mod_args)
            self.hash_package(mod_name, mod_package)
        cp = os.path.join(dirpath, os.environ['CACHE_DIR'],self.saved_cache_hashes[mod_name])
        # check to ensure only the datatypes expected are passed to the cache
        not_type = [n for n,p in pick_files.items() if not isinstance(p,(str,dict,list,pd.DataFrame,np.ndarray))] + \
                  [n for n,t in text_files.items() if not isinstance(t,(str,dict,list))]
        assert not_type == [], f'there are cache files ({", ".join(not_type)}) that are an unexpected different format'

        # directory should not exist at this point and will need to be created
        assert not os.path.exists(cp), 'this directory exists. You need to test the cache first'

        act_dir = ""
        for d in cp.split(os.sep):
            act_dir += d+os.sep
            if not os.path.exists(act_dir):
                os.mkdir(act_dir)

        # write the files into the cache        
        for p,pv in pick_files.items():
            assert '^' not in p, 'the file name cannot contain "^". That is reserved for dataframes saving schema.'
            if isinstance(pv,(str,dict,list)):
                vpp = os.path.join(cp,p+'.pickle')
                with open(vpp, 'wb+') as pickdat:
                    pickle.dump(pv,pickdat)
            elif isinstance(pv, np.ndarray):
                # numpy arrays are expected to be all numeric. They are
                # much faster to use the .save command
                vpp = os.path.join(cp,p+'.npy')
                np.save(vpp,pv)
            else:
                # dataframes may become too big to pickle directly, 
                # so they will be split and spliced back together.
                for i in range(round(len(pv)/500000) + 1):
                    vpp = os.path.join(cp,f'^{p}^{i}')
                    pv[i*500000:(i+1)*500000].to_pickle(vpp)
        for t,tv in text_files.items():
            vpt = os.path.join(cp,t)
            if isinstance(tv, dict):
                with open(vpt+'.json', 'w+') as jd:
                    json.dump(tv,jd)
            else:
                ext = '.txt'
                if isinstance(tv,list):
                    tv = '\n'.join(tv)
                    ext = '.list'
                with open(vpt+ext, 'w+') as td:
                    td.write(tv)

        # create a check to ensure the files don't change and change the permissions.
        # a False return means the system didn't allow it, True otherwise
        _ = self.finalize_cache(mod_name)
        # check the cache to ensure it was built right and saved the path
        return self.test_cache(mod_name)

    def get_mod_list(self):
        '''
        Retreives the list of modules from the current cache

        Returns
            list: all the caches that have been populated
        '''
        built_mods = []
        for k in self.saved_cache_paths.keys():
            if self.test_cache(k):
                built_mods.append(k)
            else:
                break
        return built_mods 

    def get_cache_list(self, mod_name):
        '''
        Retreives list of saved files in cache

        Parameters:
            mod_name (string): name of the module that will be used to call the stored values.
    
        Returns:
            list: all the names of the saved items        
        '''
        # check that the required arguments are ready, if the path has been saved, this test can be skipped.
        logging.debug(f'the non built saved_cache_paths looks like this: {self.saved_cache_paths}')
        if self.saved_cache_paths[mod_name] == "":
            # if not build it check the hash and build it if necessary
            assert self.test_cache(mod_name), 'something went wrong while checking the cache.'
        logging.debug(f'the built saved_cache_paths looks like this: {self.saved_cache_paths}')
        
        cp = self.saved_cache_paths[mod_name]
        # identify the datatype
        ## identify if the file is a pd.DataFrame they are all saved in the form '^{variable name}^'
        f_list = []
        for f in [v for v in os.listdir(cp) if v not in ['filled_files.md5','deps_package']]:
            if '^' not in f and f not in f_list:
                f_list.append(f.split('.')[0])
            if '^' in f and f.split('^')[1] not in f_list:
                f_list.append(f.split('^')[1])
        return f_list
        
    def get_cache_file(self, mod_name, vn):
        '''
        Retrieves a saved file
        
        Parameters:
            mod_name (string): name of the module that will be used to call the stored values.
            vn (string): the variable being retrieved.

        Variables:
            cp (string): the full path to the cache. Retrieved from saved_cache_paths.
        
        Returns:
            file: the data being retrieved in the cache
        '''
        # check that the required arguments are ready, if the path has been saved, this test can be skipped.
        logging.debug(f'the non built saved_cache_paths looks like this: {self.saved_cache_paths}')
        if self.saved_cache_paths[mod_name] == "":
            # if not build it check the hash and build it if necessary
            assert self.test_cache(mod_name), "something went wrong during build. check logging (`-l 10`)"
            
        logging.debug(f'the built saved_cache_paths looks like this: {self.saved_cache_paths}')
        
        cp = self.saved_cache_paths[mod_name]
        # identify the datatype
        ## identify if the file is a pd.DataFrame they are all saved in the form '^{variable name}^'
        filelist = [f for f in os.listdir(cp) if vn in f]
        assert len(filelist)>0, f'{vn} is not a name of a saved file in {mod_name}'

        if len([f for f in filelist if '^' not in f])==0: #all files are part of a batched save of the dataframe
            assert len([f for f in filelist if f'^{vn}^' in f]) == len(filelist), f'the variable {vn} substring of the files {",".join(filelist)}. All variable names need to be distinct'
            dfstitch = []
            for i in range(len(filelist)):
                vp = os.path.join(cp,f'^{vn}^{i}')
                dfstitch.append(pd.read_pickle(vp))
            dfstitch = pd.concat(dfstitch)
            return dfstitch

        assert len(filelist) == 1, f'the variable {vn} substring of the files {",".join(filelist)}. All variable names need to be distinct'

        if f'{vn}.npy' == filelist[0]:
            return np.load(os.path.join(cp, filelist[0]), allow_pickle=True)
        if f'{vn}.pickle' == filelist[0]:
            with open(os.path.join(cp, filelist[0]), 'rb') as vnd:
                return pickle.load(vnd)

        with open(os.path.join(cp, filelist[0]), 'r') as vnd:
            if f'{vn}.json' == filelist[0]:
                return json.load(vnd)
 
            var_str = vnd.read()
        if f'{vn}.list' == filelist[0]:
            return var_str.split('\n')

        assert f'{vn}.txt' == filelist[0], f'filelist[0] is an unknown filetype'
        return var_str

    class ittr_dataset():
        '''
        Retrieves a saved file
        
        Parameters:
            mod_name  (string):    name of the module that will be used to call the stored values.
            file_name (string):    the variable being retrieved.
            func      (function):  the function to be preformed on database when itterated

        Variables:
            cp (string): the full path to the cache. Retrieved from saved_cache_paths.
        
        Returns:
            itterator: the data being retrieved in the cache
        '''
        def __init__(s, self, mod_name, file_name):
            s.file_name = file_name
            # check that the required arguments are ready, if the path has been saved, this test can be skipped.
            logging.debug(f'the non built saved_cache_paths looks like this: {self.saved_cache_paths}')
            if self.saved_cache_paths[mod_name] == "":
                # if not build it check the hash and build it if necessary
                assert self.test_cache(mod_name), "something went wrong during build. check logging (`-l 10`)"
                
            logging.debug(f'the built saved_cache_paths looks like this: {self.saved_cache_paths}')
            
            s.cp = self.saved_cache_paths[mod_name]
            ## identify if the file is a pd.DataFrame they are all saved in the form '^{variable name}^'
            s.filelist = [f for f in os.listdir(s.cp) if s.file_name in f]
            assert len(s.filelist)>0, f'{s.file_name} is not a name of a saved file in {mod_name}'
    
            assert len([f for f in s.filelist if f'^{s.file_name}^' in f]) == len(s.filelist), f'the variable {s.file_name} is not a pandas dataframe'
            s.func = []

        def __iter__(s,column_num=0):
            for i in range(len(s.filelist)):
                vp = os.path.join(s.cp,f'^{s.file_name}^{i}')
                df = pd.read_pickle(vp)
                for f in s.func:
                    df = f(df)
                for i in range(len(df)):
                    yield df.iat[i, column_num]

        def add_func(s, func):
            s.func.append(func)

        def concat_db(s):
            concat_db = []
            for i in range(len(s.filelist)):
                vp = os.path.join(s.cp,f'^{s.file_name}^{i}')
                df = pd.read_pickle(vp)
                for f in s.func:
                    df = f(df)
                concat_db.append(df)
            return pd.concat(concat_db)


####### test the variables to make sure the expected cache args are the same as the ones declared
    def check_args(self, module, args):
        '''
        this sanity check is run at the init of each of the modules to ensure that the cache uses all the args

        Parameters:
            module (string): identification of the module that is being checked
            args (dict): locals from the init

        Returns:
            boolean: Either the lists match or they don't
        '''
        new_args = sorted([v for v in args if v not in ['self','cc', 'kwargs']])
        self.logging.debug(f'stored args: {sorted(self.deps[module][2])}\npassed args: {sorted(new_args)}')
        return sorted(self.deps[module][2]) == new_args

if __name__=='__main__':
    import argparse
    
    # provide options: filename, module, logging
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--module', type=str, default="",
                        help='name of the module you want to pull from. If not included all modules will be listed.')
    parser.add_argument('--filename', type=str, default="",
                        help='name the file that you want to retrieve. If not included all cached files will be listed.')
    parser.add_argument('--logging', '-l', type=int, default=20,
                        help='this is the logging level that you will see. Debug is 10')

    args = parser.parse_args()
    cc=CacheClass(args.logging)
    logging.debug(f'the args passes are {args}')
    mod_list = cc.get_mod_list()
    if args.module == "":
        print('modules used:')
        for m in mod_list:
            print(f'    {m.ljust(20)}: {cc.saved_cache_paths[m]}')    
######  these were removed because of the environment uses different names than the packages themselves.
######  the environment needs to be built with `conda env export > env_file.yml` and checked

#        with open(os.path.join('documentation','environment.yml'), 'w') as pyenv:
#            pyenv.write('\n'.join(ppl))
#        print(f'environment saved to "{os.path.join("documentation","environment.yml")}"') 

    elif args.filename == "":
        print('located at:', cc.saved_cache_paths[args.module])
        print('\nfile list:')
        for f in cc.get_cache_list(args.module):
            print("   ", f)
    else:
        f = cc.get_cache_file(args.module, args.filename)
        if isinstance(f, pd.DataFrame):
            f.to_csv('cached_file', index=False)
        elif isinstance(f, (list,dict,np.ndarray)):
            if isinstance(f, np.ndarray):
                f = f.tolist()
            with open('cached_file', 'w') as fd:
                json.dump(f, fd, indent=2)
        else:
            with open('cached_file', 'w') as fd:
                fd.write(str(f))
        print(f)
        print('file is saved at "./cached_file"')
        print('located at:', cc.saved_cache_paths[args.module])
