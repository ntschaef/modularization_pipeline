import os, re
from nltk.stem import SnowballStemmer

class Cleaner():
    r'''
    This class will clean the text and store the results for other modules.

    Attributes:
        cc (class): the CachedClass will be passed through all the modules 
                    and keep track of the pipeline arguements.
        remove_whitespace (boolean): combine groups of spaces into one space?
        remove_longword (boolean): remove words over 25 characters in length?
        remove_breaktoken (string): indicates which punctuation will be removed: none, dups, or all.
        remove_punc (string): indicates which punctuation will be removed: none, dups, most, or all.
        lowercase (boolean): make all alpha characters lowercase?
        convert_breaktoken (boolean): convert all \n \r and \t to "breaktokens"
        convert_escapecode (boolean): convert "\x??" tokens to "\n" or " " (instead of being left alone)
        convert_general (boolean): convert numbers with decimals to "floattoken" and large number (>=100) to "largeinttoken"
        stem (boolean): regularize terms into their "root" category (aka use stemmer package)
        fix_clocks (boolean): regularize the "time" terminology.
    '''

    def __init__(self, cc, remove_breaktoken='dups', remove_whitespace=True, remove_longword=True, remove_punc='most', 
                 convert_breaktoken=True, convert_escapecode=True, convert_general=True,
                 lowercase=True, stem=False, fix_clocks=False, **kwargs):
        r'''
        Initializer of the Parser. Defines the "attributes".

        Parameters:
            cc (class): the CachedClass will be passed through all the modules 
                        and keep track of the pipeline arguements.
            remove_whitespace (boolean): combine groups of spaces into one space?
            remove_longword (boolean): remove words over 25 characters in length?
            remove_breaktoken (string): indicates which punctuation will be removed: none, dups, or all.
            remove_punc (string): indicates which punctuation will be removed: none, dups, most, or all.
            lowercase (boolean): make all alpha characters lowercase?
            convert_breaktoken (boolean): convert all \n \r and \t to "breaktokens"
            convert_escapecode (boolean): convert "\x??" tokens to "\n" or " " (instead of being left alone)
            convert_general (boolean): convert numbers with decimals to "floattoken" and large number (>=100) to "largeinttoken" and normalize punctuation
            stem (boolean): regularize terms into their "root" category (aka use stemmer package)
            fix_clocks (boolean): regularize the "time" terminology.
        '''
        assert cc.check_args("Cleaner",locals()), 'mismatch between expected args and what is being used'
        assert remove_punc in ['none', 'dups', 'most', 'all'], f'{remove_punc} is an unknown option. the "remove_punc" model_args should be either "none", "dups", "most", or "all".'
        assert remove_breaktoken in ['none', 'dups', 'all'], f'{remove_breaktoken} is an unknown option. the "remove_breaktoken" model_args should be either "none", "dups", or "all".'
        assert not convert_breaktoken or (remove_breaktoken != 'all'), 'the breaktokens will not be removed if they were converted first.'
        cur_path = os.path.dirname(os.path.abspath(__file__))
        self.pickle_files = {}
        self.text_files = {}
        cc.saved_mod_args['convert_breaktoken'] = (convert_breaktoken, 'Cleaner', r'Will "\n", "\r", and "\t" be converted to "breaktoken"?')
        cc.saved_mod_args['convert_escapecode'] = (convert_escapecode, 'Cleaner', r'Will the "\x??" tokens be convert to "\n" or " " (instead of being left alone)')
        cc.saved_mod_args['convert_general'] = (convert_general, 'Cleaner', r'convert numbers with decimals to "floattoken" and large number (>=100) to "largeinttoken" and normalize punctuation')
        cc.saved_mod_args['remove_whitespace'] = (remove_whitespace, 'Cleaner', 'Will groups of spaces be combined into one space?')
        cc.saved_mod_args['remove_longword'] = (remove_longword, 'Cleaner', 'Will words over 25 characters in length be removed?')
        cc.saved_mod_args['remove_breaktoken'] = (remove_breaktoken, 'Cleaner', 'Indicates which punctuation will be removed: none, dups, or all.')
        cc.saved_mod_args['remove_punc'] = (remove_punc, 'Cleaner', 'Indicates which punctuation will be removed: none, dups, most, or all')
        cc.saved_mod_args['lowercase'] = (lowercase, 'Cleaner', 'Will all alpha characters be converted to lowercase?')
        cc.saved_mod_args['stem'] = (stem, 'Cleaner', 'Will the terms be categorized into their "root" category (aka use stemmer package)')
        cc.saved_mod_args['fix_clocks'] = (fix_clocks, 'Cleaner', 'Will the "time" terminology be regularized?')

    ############################## introduction of cleaner lists created for later calls #######################
        
        # get rid of odd codes
        self.escape_nonregular = [
            (re.compile(r'(\\|)&#(.{3});'), lambda pat: "\\"+pat.group(2)),
            (re.compile(r'(\\(0|)x[0-9a-f]{2})(\\(?!x)|)', flags=re.IGNORECASE), 
                             lambda pat: pat.group(1).lower())]
        #####list of characters that show up that have trailing \
        ##### examples of each are:
        ##### \t\,\r\,\e\,\n\,\s\,\.br\,\e  e\,\fs12\ul0, x0a\, \x0d0a0d0a\
        odd_list = [r'\\[t|r|e|n|s]\\(?![t|r|e|n|s])',r'\\.br((\$|\^|).br)*\\(?!.br)', r'\\e[\n| ]{1,2}e\\',
                    r'(?<!\\)x.{2,3}\\',r'\\x(0.){2,}\\']
        self.odd_breaks = [re.compile(v+'(?!(x|u))', flags=re.IGNORECASE) for v in odd_list]
        error_list_ignorecase = [r'(x(?![0-9a-f]{2}))', r'(u)']
        error_list_casesensitive = [r'(N)']
        self.error_codes_ic = [(re.compile(r'\\' + v, flags=re.IGNORECASE), lambda pat: r'\\ ' + pat.group(1)) for v in error_list_ignorecase]
        self.error_codes_cs = [(re.compile(r'\\' + v), lambda pat: r'\\ ' + pat.group(1)) for v in error_list_casesensitive]

        # clock conversion
        time_to_words = {
            0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six',
            7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten', 11: 'eleven', 12: 'twelve',
            13: 'thirteen', 14: 'fourteen', 15: 'fifteen', 16: 'sixteen',
            17: 'seventeen', 18: 'eighteen', 19: 'nineteen', 20: 'twenty', 21: 'twentyone',
            22: 'twentytwo', 23: 'twentythree', 24: 'twentyfour'}
        self.clock_conversions = [
#            (re.compile(r'\b0?([0-9])(:[0-9]{2})? *?oclock\b', flags=re.IGNORECASE), r'\1 oclock'),
            (re.compile(r'\b([0-9]{1,2})(:[0-9]{2})? *?oclock\b', flags=re.IGNORECASE), r'\1 oclock'),
            (re.compile(r'oclock *?- *', flags=re.IGNORECASE), r'oclock to '),
            (re.compile(r'([012]?[0-9]) *?oclock'), lambda m: time_to_words.get(int(m.group(1)), m.group(1))+'oclock')]

        # breaktokens conversions
        bt_refined = '\n\r\t'
        bt_list = '|'.join([t for t in bt_refined]+['breaktoken'])
        self.break_dups = re.compile('(' + bt_list + r')\1{1,}')
        self.break_regular = re.compile('(' + bt_list + ')')

        # punctuation conversions
        punc_list = r'!\*\+,-./:;<=>\?\\_\|\(\)"\'\{\[\}\]'
        self.punc_dups = re.compile('([^0-9A-Za-z ' + bt_refined + r'])\1{1,}')
        self.punc_nonregulars = re.compile('[^0-9A-Za-z '+ punc_list + bt_refined + ']')
        self.punc_nonalphanum = re.compile(r'([^0-9A-Za-z ' + bt_refined + '])')
       
        # general conversions
        self.gen_conv = [
            (re.compile(r'[0-9]+\.[0-9]+'), ' floattoken '),
            (re.compile(r'[0-9]{3,}'), ' largeinttoken '),
            (re.compile(r'[\{\[]'), '('),
            (re.compile(r'[\}\]]'), ')'),
            (re.compile('[^0-9a-zA-z ' + bt_refined + punc_list + ']'), '*'),
            (re.compile(r'[\'"]'), ' ')]

        # load stemming
        self.stemmer = SnowballStemmer("english")

    def build_dataset(self, cc):
        '''
        this is the function used to continue the build of the model within the pipeline

        Parameters:
            cc (class): the CachedClass will be passed through all the modules
                        and keep track of the pipeline arguements.
        Modules:
            build_query (string): this will construct the queries needed to refine the data to the necessary fields
            import_database (pd.DataFrame): this will dump the fields of the database into a dataframe
        '''
        mod_name = self.__module__.split('.')[-2]
        mod_args = cc.all_mod_args()
        if not cc.test_cache(mod_name=mod_name, del_dirty=cc.remove_dirty, mod_args=mod_args):
            with cc.timing("extracting data for cleaner"):
                data, schema, describe = self.get_cached_deps(cc)
            with cc.timing("Cleaning data"):
                self.clean_data(cc.logging.warning, data, schema, mod_args)
                self.describe(describe, mod_args)
            cc.build_cache(mod_name=mod_name, 
                           pick_files=self.pickle_files, 
                           text_files=self.text_files,
                           mod_args=mod_args)

    def predict_single(self, data, stored, args, filters):
        '''
        will clean the text of data passed through

        Parameters:
            data (dict): the raw data
            stored (dict): data that is needed to predict the record
            args (dict): arguments passed through
            filters (boolean): includes additional filter fields, Default: False
        Return:
            dict: the cleaned data
        '''
        data_fields = stored['schema'][stored['schema']['tables'][0]]['data_fields']
        for f in data_fields:
            if data[f] == "":
                text = ""
            else:
                text = self.clean_text(data[f], args, print)
            data[f] = text
        return data

    def predict_multi(self, data, stored, args, filters, logging):
        '''
        will create a dataframe based on the schema of the args

        Parameters:
            data (dict): path for the database needing to be converted
            stored (dict): data that is needed to predict the record
            args (dict): arguments passed through
            filters(bool): will the data be filtered according to the args?
            logging (function): print fuction
        Return:
            pd.DataFrame: the converted data
        '''
        self.clean_data(logging, data, stored['schema'],args)
        return self.pickle_files['df_cleaned']

    def get_cached_deps(self, cc):
        '''
        This function pulls the prior cached data.
        
        Parameters:
            cc (class): the CachedClass will identify the previous caches.
        Return:
            data (tuple): collection of the different cached data that was recieved:
                          df_raw (pd.DataFrame), schema (dict), describe (string)
        '''
        data = (
             cc.get_cache_file('Parser', 'df_raw'),
             cc.get_cache_file('Parser', 'db_schema'),
             cc.get_cache_file('Parser', 'describe'))
        return data

    def clean_text(self, s, mod_args,logging):
        '''
        This function will clean the raw text based on the options provided.

        Parameters:
            s (string): the text that needs to be cleaned
            mod_args (dict): the arguements showing how the text needs to be cleaned
        '''
        if s is None or r'\S\\S\PDF\S\CD' in s:
            # remove None strings and PDF dumps
            return None
        try:
           # clean up text in the following order convert_escapecode, lowercase, fix_clocks, remove_punc, remove_breaktoken, stem, remove_longword, remove_whitespace
            if mod_args['lowercase']:
                # convert all alpha characters to lowercase
                s = s.lower()+" "
            else:
                s = s+" "
            if mod_args['convert_escapecode']:
                # clean up random error codes
                for regex, replacement in self.escape_nonregular:
                    s = regex.sub(replacement, s)
                for regex in self.odd_breaks:
                    s = regex.sub(' ',s)
                for regex, replacement in self.error_codes_ic+self.error_codes_cs:
                    s = regex.sub(replacement, s)
                s = s.encode('utf8').decode('unicode-escape')
            if mod_args['convert_general']:
                # normalize punctuation and numbers to tokens
                for regex, replacement in self.gen_conv:
                    s = regex.sub(replacement, s)
            if mod_args['fix_clocks']:
                # regulariz all the time notation
                for regex, replacement in self.clock_conversions:
                    s = regex.sub(replacement, s)
            # determine how to deal with breaktokens
            if mod_args['convert_breaktoken']:
                # convert all breaktokens to one term
                s = re.sub(self.break_regular, 'breaktoken', s)
            if mod_args['remove_breaktoken'] == 'dups':
                # remove all duplicate break tokens
                s = re.sub(self.break_dups, r'\1', s)
            # determine how to deal with punctuation
            if mod_args['remove_punc'] in ['dups', 'most']:
                # remove duplicate punctuation
                s = re.sub(self.punc_dups, r'\1', s)
                if mod_args['remove_punc']=='most':
                # remove non-regular punctuation
                    s = re.sub(self.punc_nonregulars, ' ', s)
            if mod_args['remove_punc']=='all':
                # remove all punctuation
                s = re.sub(self.punc_nonalphanum, ' ', s)
            else:
                # seperate all punctuation
                s = re.sub(self.punc_nonalphanum, r' \1 ', s)
            # split the message into an array of words.
            if mod_args['remove_breaktoken'] == 'all':
                # remove all breaktokens
                s = s.split()
            else:
                # add spacing to breaktokens then split on spaces
                s = re.sub(self.break_regular, r' \1 ', s)
                s = s.split(' ')
            if mod_args['remove_longword']:
                # remove excessively long words (this eliminates most "PDF" reports)
                s = [w for w in s if len(w) < 25]
            if mod_args['stem']:
                # normalize words
                s = [self.stemmer.stem(w) for w in s]
            # recombine the message (remove the extra whitespace at the end)
            if len(s)>1:
                if s[-1] == "":
                    s = s[:-1]
            s = ' '.join(s)
            if mod_args['remove_whitespace']:
                # remove excess whitespace
                s = re.sub(' {2,}', ' ', s)
        except Exception as e:
            logging(f'could not clean string {s}: {e}')
        return s

    def clean_data(self, logging, dfraw, schema, mod_args):
        '''
        This function will be implement cleaning the textfields in the dataframe. 

        Parameters:
            logging (function): will return the print statements
            dfraw (pd.DataFrame): the raw dataframe expecting to be cleaned
            schema (dict): the schema that has the text field names
            mod_args (dict): the saved model args to identify how to clean the text as well as everything else

        Modules:
            applyfield (null): this will call the one field at a time by passing the text through the 'clean_text' module  
        '''
        fields = schema[schema['tables'][0]]['data_fields']
        def applyfield(d,df,f,mod_args,logging):
            d[f] = df[f].apply(lambda s: self.clean_text(s,mod_args,logging) if s!=None else None)
        if dfraw.shape[0] > 10:
            # use multiprocessing for increased speed
            from multiprocessing import Process, Manager
            import math
            cleantext = {f:[] for f in fields}
            for i in range(0,len(dfraw),100000):
                manager = Manager()
                cleantext_temp = manager.dict()
                job = [Process(target=applyfield, args=(cleantext_temp,dfraw[i:i+100000],f,mod_args,logging)) for f in fields]
                _ = [p.start() for p in job]
                _ = [p.join() for p in job]
                for k in fields:
                    cleantext[k] += list(cleantext_temp[k])
        else: # avoid forking for small dataframes
            cleantext = {f: dfraw[f].apply(lambda s: self.clean_text(s,mod_args,logging) if s!=None else None) 
                for f in dfraw.columns if f in fields}
            fields = [f for f in dfraw.columns if f in fields]
        cleaned = dfraw.copy()
        for f in fields:
            cleaned[f] = cleantext[f]
        self.pickle_files['df_cleaned'] = cleaned

    def describe(self, describe, mod_args):
        lines = []
        lines.append(describe)
        lines.append('\nCLEANER:')
        lines.append('Text was cleaned by taking the following actions:')
        if mod_args['convert_escapecode']:
            lines.append("Convert \"\\x??\" tokens to \"\\n\" or \" \" (instead of being left alone)")
        if mod_args["lowercase"]:
            lines.append("Make all alpha characters lowercase") 
        if mod_args["convert_general"]:
            lines.append("Convert numbers with decimals to \"floattoken\" and large number (>=100) to \"largeinttoken\" and normalize punctuation")
        if mod_args["fix_clocks"]:
            lines.append("Regularize the \"time\" terminology")
        if mod_args["convert_breaktoken"]:
            lines.append("Convert all \\n \\r and \\t to \"breaktokens\"")
        if mod_args["remove_breaktoken"]=="dups":
            lines.append("Remove duplicate breaktokens (\\n, \\r, \\t, or \"breaktoken\" if the were converted)")
        if mod_args["remove_breaktoken"]=="none":
            lines.append("Add spacing around all the breaktokens (\\n, \\r, \\t, or \"breaktoken\" if the were converted)")
        if mod_args["remove_punc"] == "dups":
            lines.append("Remove duplicate for all characters that are not alphanumeric or a spacing character (aka punctuation)")
        if mod_args["remove_punc"]=="most":
            lines.append("Remove duplicates of all standard punctuation ( !*+,-./:;<=>?\\_|() ) and removes all other nonalphanumeric non spacing characters")
        if mod_args["remove_punc"] == "all":
            lines.append("Removes all characters that are not alphanumeric and not spacing (aka punctuation)")
        if mod_args["remove_punc"] == "none":
            lines.append("Creates space around all characters that are not alphanumeric and not spacing (aka punctuation)")
        if mod_args["remove_breaktoken"]=="all":
            lines.append("Removes all break tokens (\\n, \\r, \\t, or \"breaktoken\" if the were converted)")
        if mod_args["remove_longword"]:
            lines.append("Remove words over 25 characters in length (this is done to remove PDF formatting issues)")
        if mod_args["stem"]:
            lines.append("Regularize terms into their \"root\" category by using the nltk stemmer package")
        if mod_args["remove_whitespace"]:
            lines.append("Combine groups of spaces into one")
        self.text_files['describe']='\n'.join(lines)

if __name__=='__main__':
    import argparse

    # provide options
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # reqired text
    parser.add_argument('text', type=str,
                        help='text needing to be cleaned')
    # optional cleaner arguements
    parser.add_argument('--convert_breaktoken', '-cb', type=lambda x: (str(x).lower() in ['true','1','yes']), default=False,
                        help=r'convert all \n \r and \t to "breaktokens"')
    parser.add_argument('--convert_escapecode', '-ce', type=lambda x: (str(x).lower() in ['true','1','yes']), default=False,
                        help=r'convert "\x??" tokens to "\n" or " ".')
    parser.add_argument('--convert_general', '-cg', type=lambda x: (str(x).lower() in ['true','1','yes']), default=False,
                        help='convert numbers with decimals to "floattoken" and large number (>=100) to "largeinttoken".')
    parser.add_argument('--fix_clocks', '-fc', type=lambda x: (str(x).lower() in ['true','1','yes']), default=False,
                        help='regularize the "time" terminology.')
    parser.add_argument('--lowercase', '-lc', type=lambda x: (str(x).lower() in ['true','1','yes']), default=False,
                        help='make all alpha characters lowercase?')
    parser.add_argument('--remove_breaktoken', '-rb', type=str, default='none',
                        help='indicates which punctuation will be removed: none, dups, or all.')
    parser.add_argument('--remove_longword', '-rl', type=lambda x: (str(x).lower() in ['true','1','yes']), default=False,
                        help='remove words over 25 characters in length?')
    parser.add_argument('--remove_punc', '-rp', type=str, default='none',
                        help='indicates which punctuation will be removed: none, dups, most, or all.')
    parser.add_argument('--remove_whitespace', '-rw', type=lambda x: (str(x).lower() in ['true','1','yes']), default=False,
                        help='combine groups of spaces into one space?')
    parser.add_argument('--stem', '-s', type=lambda x: (str(x).lower() in ['true','1','yes']), default=False,
                        help='regularize terms into their "root" category.')

    # logging argument
    parser.add_argument('--logging', '-l', type=int, default=20,
                        help='this is the logging level that you will see. Debug is 10')

    args = parser.parse_args()
    class tempClass():
        def __init__(self, log_int):
            import logging
            logging.root.setLevel(log_int)
            self.logging = logging
            self.cur_path = os.path.dirname(os.path.abspath(__file__))
            self.saved_mod_args = {}
        def check_args(self, mod, args):
            return True

    def null_print(*string):
        pass

    mod_args = {k:getattr(args,k) for k in vars(args)}
    temp_self = tempClass(mod_args['logging'])
    temp_self.pickle_files = {}
    temp_self.text_files = {}
    for k,v in mod_args.items():
        if k not in ['logging','text']:
            temp_self.logging.debug(f'{k}: {v}')
    c = Cleaner(temp_self,**mod_args)
    print(c.clean_text(args.text, mod_args,temp_self.logging))
