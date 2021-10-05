#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 18:55:40 2021

@author: willtyree
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem.porter import PorterStemmer
import sys
from numpy.linalg import norm
from scipy.spatial.distance import pdist
import seaborn as sns
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

class GutenbergDocumentProcessor:
    
    def __init__(self, epub_list, chap_pats, OHCO=None):
        self.epub_list = epub_list
        self.chap_pats = chap_pats
        self.OHCO = ['book_id', 'chap_num', 'para_num', 'sent_num', 'token_num']
    
    '''
    Written by Ralph Alvarado at UVA
    '''    
    def acquire_epubs(self):
        
        my_lib = []
        my_doc = []
    
        for epub_file in self.epub_list:
            
            # Get PG ID from filename
            book_id = int(epub_file.split('-')[-1].split('.')[0].replace('pg',''))
            print("BOOK ID", book_id)
            
            # Import file as lines
            lines = open(epub_file, 'r', encoding='utf-8-sig').readlines()
            df = pd.DataFrame(lines, columns=['line_str'])
            df.index.name = 'line_num'
            df.line_str = df.line_str.str.strip()
            df['book_id'] = book_id
            
            # FIX CHARACTERS TO IMPROVE TOKENIZATION
            df.line_str = df.line_str.str.replace('—', ' — ')
            df.line_str = df.line_str.str.replace('-', ' - ')
            
            # Get book title and put into LIB table -- note problems, though
            book_title = re.sub(r"The Project Gutenberg eBook( of|,) ", "", df.loc[0].line_str, flags=re.IGNORECASE)
            book_title = re.sub(r"Project Gutenberg's ", "", book_title, flags=re.IGNORECASE)
            
            # Remove cruft
            a = self.chap_pats[book_id]['start_line'] - 1
            b = self.chap_pats[book_id]['end_line'] + 1
            df = df.iloc[a:b]
            
            # Chunk by chapter
            chap_lines = df.line_str.str.match(self.chap_pats[book_id]['chapter'])
            chap_nums = [i+1 for i in range(df.loc[chap_lines].shape[0])]
            df.loc[chap_lines, 'chap_num'] = chap_nums
            df.chap_num = df.chap_num.ffill()
    
            # Clean up
            df = df[~df.chap_num.isna()] # Remove chapter heading lines
            df = df.loc[~chap_lines] # Remove everything before Chapter 1
            df['chap_num'] = df['chap_num'].astype('int')
            
            # Group -- Note that we exclude the book level in the OHCO at this point
            df = df.groupby(self.OHCO[1:2]).line_str.apply(lambda x: '\n'.join(x)).to_frame() # Make big string
            
            # Split into paragrpahs
            df = df['line_str'].str.split(r'\n\n+', expand=True).stack().to_frame().rename(columns={0:'para_str'})
            df.index.names = self.OHCO[1:3] # MAY NOT BE NECESSARY UNTIL THE END
            df['para_str'] = df['para_str'].str.replace(r'\n', ' ').str.strip()
            df = df[~df['para_str'].str.match(r'^\s*$')] # Remove empty paragraphs
            
            # Set index
            df['book_id'] = book_id
            df = df.reset_index().set_index(self.OHCO[:3])
    
            # Register
            my_lib.append((book_id, book_title, epub_file))
            my_doc.append(df)
    
        docs = pd.concat(my_doc)
        library = pd.DataFrame(my_lib, columns=['book_id', 'book_title', 'book_file']).set_index('book_id')
        return library, docs
    
    
    def tokenize(self, doc_df,remove_pos_tuple=False, ws=False):
        
        # Paragraphs to Sentences
        df = doc_df.para_str\
            .apply(lambda x: pd.Series(nltk.sent_tokenize(x)))\
            .stack()\
            .to_frame()\
            .rename(columns={0:'sent_str'})
        
        # Sentences to Tokens
        # Local function to pick tokenizer
        def word_tokenize(x):
            if ws:
                s = pd.Series(nltk.pos_tag(nltk.WhitespaceTokenizer().tokenize(x)))
            else:
                s = pd.Series(nltk.pos_tag(nltk.word_tokenize(x)))
            return s
                
        df = df.sent_str\
            .apply(word_tokenize)\
            .stack()\
            .to_frame()\
            .rename(columns={0:'pos_tuple'})
        
        # Grab info from tuple
        df['pos'] = df.pos_tuple.apply(lambda x: x[1])
        df['token_str'] = df.pos_tuple.apply(lambda x: x[0])
        if remove_pos_tuple:
            df = df.drop('pos_tuple', 1)
        
        # Add index
        df.index.names = self.OHCO
        
        return df
    
    def reduceAddStopwordsAndStems(self, TOKEN):
        TOKEN['term_str'] = TOKEN['token_str'].str.lower().str.replace('[\W_]', '')
        VOCAB = TOKEN.term_str.value_counts().to_frame().rename(columns={'index':'term_str', 'term_str':'n'})\
            .sort_index().reset_index().rename(columns={'index':'term_str'})
        VOCAB.index.name = 'term_id'
        sw = pd.DataFrame(nltk.corpus.stopwords.words('english'), columns=['term_str'])
        sw = sw.reset_index().set_index('term_str')
        sw.columns = ['dummy']
        sw.dummy = 1        
        VOCAB['stop'] = VOCAB.term_str.map(sw.dummy)
        VOCAB['stop'] = VOCAB['stop'].fillna(0).astype('int')
        stemmer = PorterStemmer()
        VOCAB['p_stem'] = VOCAB.term_str.apply(stemmer.stem)
        return TOKEN, VOCAB
    
    def saveCSVs(self, LIB, DOC, TOKEN, VOCAB, DIR):
        DOC.to_csv(f'{DIR}/DOC.csv')
        LIB.to_csv(f'{DIR}/LIB.csv')
        VOCAB.to_csv(f'{DIR}/VOCAB.csv')
        TOKEN.to_csv(f'{DIR}/TOKEN.csv')
        
    def saveCSVsPostTFIDF(self, VOCAB, TFIDF, DIR):
        VOCAB.to_csv(f'{DIR}/VOCAB.csv')
        TFIDF.to_csv(f'{DIR}/TFIDF.csv')
        
    def getParameters(self):
        tf_methods = ['sum', 'max', 'log', 'double_norm', 'raw', 'binary']
        idf_methods = ['standard', 'max', 'smooth']
        data_dir = input("Please input the path to your documents.\n")
        tokens = pd.read_csv(data_dir + 'TOKEN.csv').set_index(self.OHCO)
        vocab = pd.read_csv(data_dir + 'VOCAB.csv').set_index('term_id')
        bag_level = int(input("\nInput the number associated with the OHCO level would you like to bag your data on.\n 1. Book\n 2. Chapter\n 3. Paragraph\n"))
        if bag_level > 3:
            print("\nQuitting. Try again and input a valid number 1-3.")
            sys.exit()
        bag_level = self.OHCO[:bag_level]
        count_type = input("\nInput the letter associated with the type of count you would like to use.\n n: Tokens\n c: Distinct tokens (terms)\n")
        if count_type != 'n' and count_type != 'c':
            print("\nQuitting. Try again and input a valid type (n, c)")
            sys.exit()
        tf_type = int(input("\nWhat term frequency method would you like to use?\n 1. Sum\n 2. Max\n 3. Log\n 4. Double Norm\n 5. Raw\n 6. Binary\n"))
        if tf_type > 6:
            print("\nQuitting. Try again and input a valid number 1-6.")
            sys.exit()
        tf_type = tf_methods[tf_type-1]
        idf_type = int(input("\nWhat inverse document frequency method would you like to use?\n 1. Standard\n 2. Max\n 3. Smooth\n"))
        if idf_type > 3:
            print("\nQuitting. Try again and input a valid number 1-3.")
            sys.exit()
        idf_type = idf_methods[idf_type-1]
        return tokens, vocab, bag_level, count_type, tf_type, idf_type
        
    def getBagOfWords(self, vocab, tokens, bag_level):
        print(f"\nCreating the bag of words at the {bag_level} level...")
        vocab = vocab.dropna()
        tokens = tokens.dropna()
        tokens['term_id'] = tokens.term_str.map(vocab.reset_index().set_index('term_str').term_id)
        bow = tokens.groupby(bag_level+['term_id']).term_id.count()\
            .to_frame().rename(columns={'term_id':'n'})
        bow['c'] = bow.n.astype('bool').astype('int')
        return bow, tokens
    
    def getTF(self, tf_type, dtcm):
        print(f"\nComputing the {tf_type} term frequencies...")
        if tf_type == 'sum':
            tf = dtcm.T / dtcm.T.sum()
        elif tf_type == 'max':
            tf = dtcm.T / dtcm.T.max()
        elif tf_type == 'log':
            tf = np.log10(1 + dtcm.T)
        elif tf_type == 'raw':
            tf = dtcm.T
        elif tf_type == 'double_norm':
            tf = dtcm.T / dtcm.T.max()
            tf = 0.5 + (1 - 0.5) * tf[tf > 0]
        elif tf_type == 'binary':
            tf = dtcm.T.astype('bool').astype('int')
        tf = tf.T
        return tf
    
    def getIDF(self, idf_type, dtcm):
        print(f"\nComputing the {idf_type} inverse document frequency...")
        df = dtcm[dtcm > 0].count()
        n = dtcm.shape[0]
        if idf_type == 'standard':
            idf = np.log10(n / df)
        elif idf_type == 'max':
            idf = np.log10(df.max() / df) 
        elif idf_type == 'smooth':
            idf = np.log10((1 + n) / (1 + df)) + 1
        return idf
    
    def getTFIDF(self, tokens=None, vocab=None, bag_level=None, count_type=None, tf_type=None, idf_type=None):
        if [x for x in (tokens, vocab, bag_level, count_type, tf_type, idf_type) if x is None]:
            tokens, vocab, bag_level, count_type, tf_type, idf_type = self.getParameters()
        bow, tokens = self.getBagOfWords(vocab, tokens, bag_level)
        dtcm = bow[count_type].unstack().fillna(0).astype('int')
        df = dtcm[dtcm > 0].count()
        tf = self.getTF(tf_type, dtcm)
        idf = self.getIDF(idf_type, dtcm)
        print("\nComputing the TF-IDF (term frequency-inverse document frequency)...")
        tfidf = tf * idf
        print("\nSuccess!")
        vocab['pos_max'] = tokens.groupby(['term_id', 'pos']).count().iloc[:,0].unstack().idxmax(1)
        vocab['df'] = df
        vocab['idf'] = idf
        vocab['tfidf_sum'] = tfidf.sum()
        return tfidf, vocab
    
    def createNormalizedTables(self, TFIDF, ohco_level):
        TFIDF = TFIDF.groupby([ohco_level]).mean()
        L0 = TFIDF.astype('bool').astype('int')
        L1 = TFIDF.apply(lambda x: x / x.sum(), 1)
        L2 = TFIDF.apply(lambda x: x / norm(x), 1)
        return TFIDF, L0, L1, L2

    def createPairsTableAndComputeDistances(self, DOC, TFIDF, L0, L1, L2):
        PAIRS = pd.DataFrame(index=pd.MultiIndex.from_product([DOC.index.tolist(), DOC.index.tolist()])).reset_index()
        PAIRS = PAIRS[PAIRS.level_0 < PAIRS.level_1].set_index(['level_0','level_1'])
        PAIRS.index.names = ['doc_a', 'doc_b']
        PAIRS['cityblock'] = pdist(TFIDF, 'cityblock')
        PAIRS['euclidean'] = pdist(TFIDF, 'euclidean')
        PAIRS['cosine'] = pdist(TFIDF, 'cosine')
        PAIRS['jaccard'] = pdist(L0, 'jaccard')
        PAIRS['dice'] = pdist(L0, 'dice')
        PAIRS['js'] = pdist(L1, 'jensenshannon')
        PAIRS['euclidean2'] = pdist(L2, 'euclidean')
        PAIRS['hamming'] = pdist(TFIDF, 'hamming')
        return PAIRS
    
    def plotPairs(self, PAIRS):
        if PAIRS.shape[0] > 1000:
            SAMPLE = PAIRS.sample(1000)
        else:
            SAMPLE = PAIRS
        sns.pairplot(SAMPLE)
        
    def hca(self, sims, DOC, linkage_method='ward', color_thresh=.3, figsize=(10, 10)):
        tree = sch.linkage(sims, method=linkage_method)
        labels = DOC.title.values
        plt.figure()
        fig, axes = plt.subplots(figsize=figsize)
        dendrogram = sch.dendrogram(tree, 
                                    labels=labels, 
                                    orientation="left", 
                                    count_sort=True,
                                    distance_sort=True,
                                    above_threshold_color='.75',
                                    color_threshold=color_thresh
                                   )
        plt.tick_params(axis='both', which='major', labelsize=14)

    
    def fullProcessingPipeline(self):
        print("\nRegistering and chunking the Project Gutenberg books...\n")
        LIB, DOC = self.acquire_epubs()
        print("\nTokenizing and annotating the document...\n")
        TOKEN = self.tokenize(DOC, ws=True)
        print("Extracting a vocabulary and adding stopwords and stems...\n")
        TOKEN, VOCAB = self.reduceAddStopwordsAndStems(TOKEN)
        self.saveCSVs(LIB, DOC, TOKEN, VOCAB, 'DATA')
        TFIDF, VOCAB = self.getTFIDF()
        print("\nSaving all data into the specified directory\n")
        self.saveCSVsPostTFIDF(VOCAB, TFIDF, 'DATA')
        return LIB, DOC, TOKEN, VOCAB, TFIDF

        