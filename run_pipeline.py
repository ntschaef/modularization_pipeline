import argparse, json, logging, os, sys, torch, yaml
# provide options for user input
file_path = os.path.split(os.path.abspath(__file__))[0]
relative_string=""
if os.path.abspath('.') == os.path.split(file_path)[0]:
    sys.path.append(os.path.split(file_path)[1])
    relative_sting ="."
from caching import CacheClass

logg = 20
remove_dirty = False
write_default_config = False
################ SECTION FOR INTERNAL VARIABLES THAT NEED TO BE SET

call_order = ['Parser','Cleaner','Sanitizer','Filter','Splitter','Data_Generator','Encoder'] # This is the order the modules will run.

################ SECTION FOR INTERNAL VARIABLES END
# set logging level
logging.root.setLevel(logg)


logging.debug(f'args provided for this run: logging: {logg}, remove_dirty: {remove_dirty}, write_default_config: {write_default_config}')
#initialize Caching system
cc = CacheClass(logg)
cc.remove_dirty=remove_dirty
logging.debug(f'cache initialization: {cc.__dict__}')
#originally default of keep dirty was True
## remnaming it from remove_dirty was more meaning during the change
## since the variable is used throughout the pipeline as "remove_dirty" it was kept.
    
def write_config():
    # Load the default values from each of the modules listed above. Then write them all into a model_args.json. 
    # Additionally write a true_arg_descrpitons to check against the current one (arg_descriptions).

    pipeline_args = {'cache_dir': ['api_cache','path where the cache will be stored.'],
                     'data_output': ['output','output path in the data directory. leaving blank will use the next free integer']} 

    for mod in call_order:
        exec(f'from {mod}.{mod.lower()} import {mod} as curr_mod', globals())
        curr_mod(cc,**{})
    with open('data_args.json', 'w+') as jd:
        # group the args by the repo they are called in
        args = {m:{} for m in call_order}
        for arg_name, arg_value_list in cc.saved_mod_args.items():
            # tack on the new args in addition to the ones already added
            arg_value = arg_value_list[0]
            mod = arg_value_list[1]
            arg_descript = arg_value_list[2]
            args[mod] = {**args[mod],"  DESCRIPTION " + arg_name: arg_descript, arg_name: arg_value}
        args['Pipeline'] = {arg_name: arg_value[0] for arg_name,arg_value in pipeline_args.items()} 
        json.dump(args, jd, indent=2)
    with open(os.path.join('documentation','arg_descriptions'), 'w+') as jd:
        args = [{arg_name:arg_value_list} for arg_name, arg_value_list in cc.saved_mod_args.items()]
        args += [{arg_name:(arg_value_list[0],'Pipeline',arg_value_list[1])} for arg_name, arg_value_list in pipeline_args.items()]
        yaml.safe_dump(args, jd, indent=2, default_flow_style=False)
    sys.exit('the data_args.json is built and documented')

def get_args(source=None):
    # check to see if a model_args exists, even if not populated.
    if source:
        return get_args_from_model(source)
    else:
        return get_args_from_json()

def get_args_from_model(source):
    if torch.cuda.is_available():
        mod = torch.load(source)
    else:
        mod = torch.load(source, map_location=torch.device('cpu'))
    package = mod['metadata_package']['model_metadata']
    cc.saved_mod_args['ms_kwargs'] = (mod['metadata_package']['mod_args'], "None", "model suite arguements")
    cc.saved_mod_args['commits'] = ({mod: package[mod]['commit'] for mod in package.keys()}, "None", "commits for the saved modules")
    cc.saved_mod_args['mod_NN'] = ({layer_name:layer for layer_name, layer in mod.items() if layer_name != 'metadata_package'}, "None", "model NN")
    args = {arg_name: arg_value for mod, data_pack in package.items() if mod!='Model_Suite' for arg_name, arg_value in data_pack['mod_args'].items()}
    args['commits'] = {mod_name: data_pack['commit'] for mod_name, data_pack in package.items() if mod!='Model_Suite'}
    return args

def get_args_from_json():
    args = {}
    try:
        with open('data_args.json', 'r') as jd:
            base_data_args = json.load(jd)
    except FileNotFoundError:
        exit("FileNotFoundError: data_args.json does not exist. Create a data_args file by running 'python run_pipeline --write_default_config'")
    # make args independent of modules
    for mod_args in base_data_args.values():
        for arg_name, arg_value in mod_args.items():
            args[arg_name] = arg_value
    return args
   
def build_dataset(data_args, build_cache=True):
    # run pipeline to build caches and run scripts.
    # if this is meant to be an API, you need to load the data beforehand.
    mod_list = []
    for mod in call_order:
        logging.debug(f'mod being processed is {mod}')
        exec(f'from {relative_string}{mod}.{mod.lower()} import {mod} as curr_mod',globals())
        logging.info(f'running the {mod}.')
        module = curr_mod(cc, **data_args)
        logging.debug(f'module {mod} has been initiated')
        mod_list.append([module,mod])
        if build_cache:
            module.build_dataset(cc)
            logging.info(f'{mod} is cached at {cc.saved_cache_paths[mod]}')
    return cc, mod_list

def predict_startup(source=None,filters=False,fold=0, production=False, build_cache=True):
    # make sure all modules are running and build if needed
    data_args = get_args(source, production)
    return build_dataset(data_args, build_cache)

def predict_single(data, stored, args, mods):
    '''
    utilize the predict_single pipeline. Assume everything is as expected (json) to save time.

    Perameters:
        data (pd.DataFrame): initial dataset to be predicted
        stored (dict): all the stored data that will be needed to preprocess and predict the data
        args (dict): list of the arguements used for preprocessing
        mods (list): list of the modules used.

    '''
    # once stored data has been pulled, and the modules are running, the reports will be predicted
    for mod,mod_name in mods:
        data = mod.predict_single(data, stored, args, False)
    return data

def predict_multi(datapath, stored, args, mods, filt, use_path=True):
    '''
    formats data to correct output for mulitple reports or dataset.

    Perameters:
        data (pd.DataFrame): initial dataset to be predicted
        stored (dict): all the stored data that will be needed to preprocess and predict the data
        args (dict): list of the arguements used for preprocessing
        mods (list): list of the modules used.
        filt (boolen): will the dataset be filtered?

    '''
    # once stored data has been pulled, and the modules are running, the reports will be predicted
    def null_print(*strs):
        pass
    if isinstance(datapath, str):
        dt_is_str = True
        data = [datapath]
    else:
        dt_is_str = False
        data = datapath
    for mod,mod_name in mods:
        if mod_name == mods[-1][1]:
            if dt_is_str:
                idfields = [idfield.split(' ')[-1] for idfield in 
                       stored['schema'][stored['schema']['tables'][0]]['order_id_fields']]
                truthfields = [truthfield.split(' ')[-1] for truthfield in 
                       stored['schema'][stored['schema']['tables'][1]]['baseline_fields']]
                truth = data.set_index(idfields)[truthfields]
            else: 
                truth = None
        if dt_is_str or mod_name!='Parser':
            data = mod.predict_multi(data, stored, args, filt, null_print)
        elif mod_name=='Parser':
            for c in stored['schema'][stored['schema']['tables'][0]]['data_fields']:
                if c not in data.columns:
                    data[c] = ""
    return data, truth


if __name__=='__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--logging', '-l', default=20, type=int,
                        help='logging level for the pipeline. These values are the standard levels. 10 is the DEBUG level.')
    parser.add_argument('--keep_dirty', '-kd', action='store_true', help='for all repos that are not clean, using this will keep them from being removed.')
    parser.add_argument('--write_default_config', '-wdc', action='store_true', help='use this to write the default model_args.json file based on the other modules.')
    
    args = parser.parse_args()
    logg = args.logging
    cc.remove_dirty = args.keep_dirty==False
    write_default_config = args.write_default_config

    if write_default_config:
        write_config()
    else:
        data_args = get_args()
        logging.debug(f'data_args: {data_args}')
        _ = build_dataset(data_args)
