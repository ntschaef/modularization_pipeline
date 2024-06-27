import run_pipeline, os, json, shutil, pandas as pd, numpy as np

# create new data file and save it for future use
i = 0
while str(i) in os.listdir('data'):
    i += 1

assert os.path.isfile('data_args.json'), 'data_args.json is missing'
with open('data_args.json', 'r') as jd:
    config = json.load(jd)

# check to see if particular file is wanted
if 'Pipeline' in config.keys():
    if 'data_output' in config['Pipeline']:
        if config['Pipeline']['data_output'] != '':
            i = config['Pipeline']['data_output']

basedir = os.path.join('data',str(i))

# ensure file doesn't exist. Add overwrite option
if os.path.isdir(basedir):
    exit(f'while {basedir} exists, the data cannot be generated')

# generate the pipeline
data_args = run_pipeline.get_args()
cc,_ = run_pipeline.build_dataset(data_args)

# make empty directory
os.makedirs(basedir)

# establish the fold number
i = max([int(v.split('_')[1][4:]) for v in cc.get_cache_list('Encoder') if '_' in v])+1
for fold in range(i):
    # create database to list all the metadata, X and Y
    full_df = []
    for split in ['train','test','val']:
        dfMet = cc.get_cache_file('Encoder',f'{split}Metadata_fold{fold}')
        npX = cc.get_cache_file('Encoder', f'{split}X_fold{fold}')
        dfMet['X'] = npX
        dfY = cc.get_cache_file('Encoder', f'{split}Y_fold{fold}')
        dfMet = dfMet.join(dfY)
        dfMet['split'] = split
        full_df.append(dfMet.copy())
    pd.concat(full_df).reset_index().to_csv(os.path.join(basedir,f'data_fold{fold}.csv'),index=False)

    # make a json for all the id2word mappings
    with open(os.path.join(basedir,f'id2word_fold{fold}.json'),'w') as jd:
        json.dump(cc.get_cache_file('Data_Generator', f'id2word_fold{fold}'), jd, indent=2)

    # make a json for all the id2labels mappings
    with open(os.path.join(basedir,f'id2labels_fold{fold}.json'),'w') as jd:
        json.dump(cc.get_cache_file('Data_Generator', f'id2labels_fold{fold}'), jd, indent=2)

    # make a text file for word_embeds_fold
    with open(os.path.join(basedir,f'word_embeds_fold{fold}.npy'),'wb') as jd:
        np.save(jd,cc.get_cache_file('Data_Generator', f'word_embeds_fold{fold}'))

    with open(os.path.join(basedir,'schema.json'), 'w') as jd:
        json.dump(cc.get_cache_file('Parser', 'db_schema'), jd, indent=2)

    with open(os.path.join(basedir,'query.txt'), 'w') as qt:
        qt.write('/n'.join(cc.get_cache_file('Parser', 'duckdb_queries')))

# get metadata
with open('deps.json', 'r') as jd:
    deps = json.load(jd)

# combined packages and print to json file
pack = {}
for mod in cc.saved_packages.keys():
    if cc.saved_packages[mod] != '':
        pack[mod] = {title:{v:k for v,k in zip(sorted(deps[mod][title]), cc.saved_packages[mod][i])} for i,title in enumerate(deps[mod].keys())}
        pack[mod]['commit'] = cc.saved_packages[mod][4]

with open(os.path.join(basedir, 'metadata.json'),'w') as jd:
    json.dump(pack, jd, indent=2)

print(f'\n\n\nData was saved at {basedir}\n\n\n')
