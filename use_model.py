'''
UseModel will pull in code from other scripts to create and predict on a provided model (h5) which
was trained with the Model_Suite (v2)

external methods:
    build_dataset (class, list): uses saved arguements to generate cache names
    predict_single (list): produces a list of size 1 for tokenized reports from the pipeline
    predict_multi (list, pd.DataFrame): produces a list of tokenized reports from the pipeline
                                        with a dataframe of truth values if applicable (otherwise 
                                        it is None). 
    get_args(dict): produces the arguements that were saved in the model.

functions:
    check_commits: verifies the commits stored are the same as the commits being used
    get_stored (dict): stored data that is found in the cache 
    getX_from_dataset(list, torch.tensor, pd.DataFrame): produces the ids, X, and truth value given
                                                         the path to a database
    getX_from_record (torch.tensor): the X values from a report formatted as a dictionary 
    getX (list, torch.tensor, pd.Dataframe): produces ids, X and truth for given data. If single
                                             report the ids and truth are None.
    predict (dict, dict): the confidences and predictions given a list of tokenized reports
'''
import argparse, git, json, numpy as np, os, sqlite3, sys, time, torch, random
import torch.nn.functional as F
from pandas import read_sql
## setting the path so it will find the right submodules if run from a different directory
file_path = os.path.split(os.path.abspath(__file__))[0]
if os.path.abspath('.') == os.path.split(file_path)[0]:
    cur_path = os.path.split(file_path)[1]
    sys.path.append(cur_path)
## importing methods from run_pipeline
from Model_Suite.models.mthisan import MTHiSAN
from Model_Suite.models.mtcnn import MTCNN
from run_pipeline import build_dataset, get_args, predict_multi, predict_single

class UseModel():
    def __init__(self, model_path, filters=False, production=False, clc_flag=False, repo_path='pipeline'):
        # 0. verify and generate data
        if not os.path.exists(model_path):
            raise IOError("Provided model path does not exist")
        ### setup initial data for building the model
        self.device = torch.device('cpu')
        fold = 0
        self.data_args = get_args(source=model_path)
        if not production:
            self.check_commits(self.data_args['commits'], repo_path)
        cc, self.module_list = build_dataset(self.data_args, build_cache=True)#build_cache)
        word_embeds, num_classes = self.load_data(cc, fold)
        self.model_load(cc, word_embeds, num_classes)

    def check_commits(self, commits, repo_path):
        '''
        subfunction to verify that the saved commits match the commits currently being used
        
        Argsuments:
            commits (dict): a list of the expected commits according to the stored hashes.
            repo_path (string): the base directory for the repo. 
        '''
        mismatch_commits = {}
        for mod, commit in commits.items():
            repo = git.Repo(os.path.join(repo_path,mod))
            repo_hex = repo.head.commit.hexsha
            repo_dirty = repo.is_dirty()
            if repo_dirty:
                repo_hex = repo_hex+'-dirty'
            if 'dirty' in commit:
                print(f'WARNING: data was built on a dirty {mod} repo. Unless you have the original data, it may not run correctly')
            if repo_hex != commit:
                mismatch_commits[mod] = (commit,repo_hex)
        if len(mismatch_commits) > 0:
            nl = '\n'
            raise Exception(f'the following commits were mismatched (what is saved/what it currently is) - \n {nl.join([m+": " + v[0] + "/" + v[1] for m,v in mismatch_commits.items()])}')

    def load_data(self, cc, fold):
        '''
        pull data that will be used for the model building and predictions

        Arguments:
            cc (class): the cache class which keeps track of the caching and arguments.
            fold (int): the fold reference for the model being loaded.
                        Unless cv (not yet tested) is turned on, this will be 0.

        Returns:
            numpy.array - embedding layer for model initiation
            list - number of logits expected for each task when predicting
        ''' 
        self.mod_args= cc.mod_args('ms_kwargs')
        word_embeds = cc.get_cache_file('Data_Generator', f'word_embeds_fold{fold}')
        self.id2label = cc.get_cache_file('Data_Generator', f'id2labels_fold{fold}' )
        if self.mod_args['abstain_kwargs']['abstain_flag']:
            # add the abstain label to each task if it was trained on it
            for task, labels in self.id2label.items():
                label_len = len(labels.keys())
                self.id2label[task][str(label_len)] = 'abstain' 
            if self.mod_args['abstain_kwargs']['ntask_flag']:
                # add in the labels for the Ntask which wasn't referenced while building the data
                self.id2label['Ntask'] = {0:'predict',1:'abstain'}
        # create a dict to count the expected number of labels for each task
        num_classes = {}
        for task in self.mod_args['data_kwargs']['tasks']:
            label_count = len(self.id2label[task].keys())
            num_classes[task] = label_count
        if self.mod_args['abstain_kwargs']['ntask_flag']:
            num_classes["Ntask"] = 1
        
        self.stored = self.get_stored(cc, fold)
        self.describe = cc.get_cache_file('Encoder', 'describe')

        # set seeds
        if self.mod_args['data_kwargs']['reproducible']:
            seed = self.mod_args['data_kwargs']['random_seed']
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        else:
            seed = None

        return word_embeds, num_classes

    def model_load(self, cc, word_embeds, num_classes):
        '''
        loading the model to be used for predictions

        Arguments:
            cc (class): the cache class which keeps track of the caching and arguments.
            word_embeds (numpy.array) - premade first layer used for model structuring
            num_classes (dict) - number of labels expected from each task
        '''
        # 1. pull in all necessary components for loading the model
        if self.mod_args['model_type']=='mthisan':
            mod_build = MTHiSAN
            spec_mod_args = self.mod_args['MTHiSAN_kwargs']
        elif self.mod_args['model_type'] == 'mtcnn':
            mod_build = MTCNN
            spec_mod_args = self.mod_args['MTCNN_kwargs']

        # 2. load the model
        self.model = mod_build(word_embeds, num_classes, **spec_mod_args)
        self.model.to(self.device)
        self.model.eval()
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(model)

        model_dict = {key.replace('module.',''): layer for key,layer in cc.mod_args('mod_NN').items()}
        self.model.load_state_dict(model_dict)

        print('model loaded')

    def get_stored(self, cc, fold):
        '''
        calss method used to get the stored components out of the cache

        Arguments:
            cc (class): the cache class which keeps track of the caching and arguments.
            fold (string): indication of the fold that is being loaded.

        Returns:
            dict - storage of all the data needed to load the model
        '''
        schema = cc.get_cache_file('Parser', 'db_schema')
        id2word = cc.get_cache_file('Data_Generator', f'id2word_fold{fold}')
        query = cc.get_cache_file('Parser', 'duckdb_queries')

        word_map = None 
        if cc.mod_args('deidentify'):
            word_map = cc.get_cache_file('Sanitizer', 'word_map')
            invert_word_map = {idx:word for word,idx in word_map.items()}
            id2word = {token:invert_word_map[int(idx)] for token,idx in id2word.items() if idx not in ['<pad>','recordbreaktoken', '<unk>', '']}
            id2word[0] = '<pad>'
            id2word[1] = ''
            id2word[len(id2word.keys())] = 'recordbreaktoken'
            id2word[len(id2word.keys())] = '<unk>'
            
        stored = {'word_map': word_map, 'schema': schema,
                  'word2tok': {word:token for token,word in id2word.items()},
                  'vocab': list(id2word.values()),
                  'query': query}

        return stored

    def getX_from_dataset(self, data, def_query=False):
        '''
        run the data through the run_pipeline and process a batch of data (sqlite connection) for predicting

        Arguments:
            data (string or pandas.DataFrame): the source for the dataset to be score. 
                                    If string, it will be a path to a database to be queried
                                    If dataframe, it will be processed as is
            def_query (boolean): flag to show if using the query stored in the model.

        Returns:
            list - the ids ordered in the same way as the inputs
            torch.tensor - the X values that will be fed to the model
            dict - the gold standard to be tested against
        '''
        df = data
        if isinstance(data, str):
            query = self.stored['query']
            if def_query:
                query = ("select filename as recordDocumentId, patientIdNumber as patientId, tumorRecordNumber as tumorId, "
                                "textPathClinicalHistory, textStagingParams, textPathMicroscopicDesc, textPathNatureOfSpecimens, "
                                "textPathSuppReportsAddenda, textPathFormalDx, textPathFullText, textPathComments, "
                                "textPathGrossPathology from epath;")
            conn = sqlite3.connect(data)
            df = read_sql(query,conn)
        X, truth = predict_multi(df, self.stored, self.data_args, self.module_list, False)
        doc_len = self.mod_args['data_kwargs']['doc_max_len']
        X = [(list(x1) + [0]*doc_len)[:doc_len] for x1 in list(X)]

        # explicitly declare returns
        rec_ids = list(df['recordDocumentId'])
        X_tensor = torch.tensor(X)

        return rec_ids, X_tensor, truth

    def getX_from_record(self, record):
        '''
        run the data through the run_pipeline and process a single report (json format)

        Arguments:
            record (dict): a the dictionary with the stored IDs and text.

        Returns:
            torch.tensor - X values to be fed into the model.
        '''
        X = predict_single(record, self.stored, self.data_args, self.module_list)
        doc_len = self.mod_args['data_kwargs']['doc_max_len']
        X = [(x1 + [0]*doc_len)[:doc_len] for x1 in X]
        X_tensor = torch.tensor(X)
        return X_tensor

    def getX(self, data_path, multi=False, def_query=False):
        '''
        switchboard for data to see how to process it

        Arguments:
            data_path (string): path to the data being used
            multi (boolean): flag to indicate if a multiple records will be scored (default: False)
            def_query (boolean): flag to indicate if a preformed query will be used (default: False)

        Returns:
            list - the ids ordered in the same way as the inputs
            torch.tensor - the X values that will be fed to the model
            dict - the gold standard to be tested against
        '''
        if not os.path.exists(data_path):
            raise IOError("Provided data to be predicted on does not exist")
        if multi:
            return self.getX_from_dataset(data_path, def_query)
        else:
            with open(data_path, 'r') as j:
                doc = json.load(j)
            # explicitly declare returns for single run
            rec_id = [doc['recordDocumentId']]
            X_tensor = self.getX_from_record(doc)
            truth = None
            return rec_id, X_tensor, truth

    def predict(self, X, multi=False, clc_flag=False):
        '''
        using loaded model and generated X, produce predictions
           Note: much of this code should match the Model_Suite v2 predictions.predict -> _predict function
                 but that was forgone mostly for speed, and also that code assumes some objects we don't have

        Arguments:
            X (torch.tensor): inputs for the model
            multi (boolean): flag to indicate if a multiple records will be scored (default: False)
            clc_flag (boolean): flag to indicate if the case level context method needs to be used (default: False

        Returns:
            dict - confidences for all the labels
            dict - predictions for all the tasks 
        '''
        rec_num = X.shape[0]
        logits = []
        logits = self.model(X) # get logit predictions

        # store the data as it is expected to be output from the Model_Suite _predict function
        predictions = [{} for i in range(rec_num)]
        confidence = [{} for i in range(rec_num)]
        for i in range(rec_num):
            for j, task in enumerate(self.mod_args['data_kwargs']['tasks']):
                if self.mod_args['train_kwargs']['multilabel']:
                    confidence[i][task] = torch.logits(logits[task][i])
                else:
                    confidence[i][task] = F.softmax(logits[task][i], dim=0)
                outputs = logits[task][i].detach().cpu().numpy()
                pred_idx = np.argmax(outputs)
                predictions[i][task] = self.id2label[task][str(pred_idx)]
            if self.mod_args['abstain_kwargs']['ntask_flag']:
                # run only the Ntask through the sigmoid
                ntask_prob = torch.sigmoid(logits[-1])[:, -1].detach().cpu().numpy()[i]
                predictions[i]['Ntask'] = self.id2label['Ntask'][round(ntask_prob)]
                confidence[i]['Ntask'] = np.array([1-ntask_prob,ntask_prob])
        return confidence, predictions

    def main_function(self, data_path, multi, def_query, test):
        '''
        the main run of the script if it is run from the command line.

        Arguments:
            data_path (string): path to the data being used
            multi (boolean): flag to indicate if a multiple records will be scored
            def_query (boolean): flag to indicate if a preformed query will be used (default: False)
            test (boolean): flag indicating if the model statistics needs to be produced

        Outputs:
            the predictions and confidences are saved to a local file    
        '''
        ids, X, truth = um.getX(data_path, multi=multi, def_query=def_query)
        rec_num = X.shape[0]
        confidence = []
        preds = []
        for i in range(0,rec_num,500):
            X_red = X[i:i+500]
            confidence_temp, preds_temp = um.predict(X_red, multi=multi)
            confidence += confidence_temp
            preds += preds_temp
        preds = {ids[idx]: pred for idx,pred in enumerate(preds)}
        with open('preds.json', 'w') as j:
            json.dump(preds, j, indent=2)
        for i in range(rec_num):
            for task in confidence[i].keys():
                confidence[i][task] = {label: eval(str(confidence[i][task][idx])) for idx,label in enumerate(um.id2label[task].values())}
        confidence = {ids[i]: confs for i, confs in enumerate(confidence)}
        with open('confidences.json', 'w') as j:
            json.dump(confidence, j, indent=2)
        print('\n\nthe predictions are saved at preds.json and confidences.json\n\n')
        if multi:
            exit()
        # basic tests to make sure the model is running consistently
        if test:
            confidence, preds = um.predict(X)
            start = time.time()
            for i in range(500):
                ids, X, truth = um.getX(data_path, multi=multi, def_query=def_query)
                confidence, pred = um.predict(X, multi)
            print(f'500 reports were consumed and predicted averaging {(time.time()-start)*2} millisecionds')
            conf1, pred1 = um.predict(X)
            print(f"the reports all have the same confidence? {all([(conf1[0][t] == confidence[0][t]).all() for t in conf1[0].keys()])}")
            print(f"the reports predict the same thing? {all([(pred1[0][t]==pred[0][t]) for t in pred[0].keys()])}")
            print(f"the difference between the confidences is {sum([(confidence[0][t] - conf1[0][t]).sum() for t in confidence[0].keys()])}")

if __name__== "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_path', type=str,
                        help='the relative path for the model that will be used')
    parser.add_argument('--json_report', '-jr', type=str, default='',
                        help='the relative path of a single report to be predicted on')
    parser.add_argument('--model_data', '-md', type=str, default='',
                        help='path to where the pregenerated data is stored')
    parser.add_argument('--dataset', '-d', type=str, default='',
                        help='the testing dataset that will be predicted on')
    parser.add_argument('--general_query', '-gq', action='store_true',
                        help='will run a geneneral query on the dataset instead of the save one')
    parser.add_argument('--filter_data', '-fd', action='store_true', 
                        help='will filter the dataset pointed at')
    parser.add_argument('--test', '-t', action='store_true',
                        help='will test the model for consistency and speed')

    args = parser.parse_args()
    data_path = args.json_report
    multi = False

    if not os.path.exists(args.model_path):
        exit(f'no file exists at {args.model_path}.')
    if not os.path.exists(args.json_report) and not os.path.exists(args.dataset):
        exit(f'no file exists at {args.json_report}.')
    if args.json_report == '':
        data_path=args.dataset
        multi=True
    um = UseModel(args.model_path, repo_path="")

    um.main_function(data_path, multi=multi, def_query=args.general_query, test=args.test)
