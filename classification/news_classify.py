# coding=UTF-8
import re # regular expression
import jieba as jb
import os
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
import codecs

# config jieba
jb.load_userdict('resource/location+words.txt')

# params
categories = [
	'china_world','entertainment','finance','lifestyle','news','sport']
categories_index = {
	'china_world':0,'entertainment':1,'finance':2,'lifestyle':3,'news':4,'sport':5}


# input a news txt, output list of parsed string
def jieba_parse(txt):
	lines = [l.replace('\n',' ') for l in open(txt)]
	if len(lines) <= 1:
		return None
	# skip first line
	line = ''.join(lines[1:])
	# replace punctuations with space
	line = re.sub(
		ur"[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）：；《）《》“”()»〔〕-]+",
		" ", line.decode("utf8"))
	# print line
	words = jb.cut(line, cut_all=False)
	words_spaced = ' '.join(words)
	# print words_spaced
	return words_spaced
	

# prepare X.txt, Y.txt
# X.txt: word1 word2 word3 ... for each line
# Y.txt: china_world / entertainment / finance / lifestyle / news / sport
def prepare_XY():
	src_folder = 'data/'
	
	fileX_tr = 'data_processed_train/X.txt'
	fileY_tr = 'data_processed_train/Y.txt'
	foutX_tr = codecs.open(fileX_tr, 'w', 'utf-8')
	foutY_tr = codecs.open(fileY_tr, 'w', 'utf-8')
	fileX_te = 'data_processed_test/X.txt'
	fileY_te = 'data_processed_test/Y.txt'
	foutX_te = codecs.open(fileX_te, 'w', 'utf-8')
	foutY_te = codecs.open(fileY_te, 'w', 'utf-8')
	# first 8/10 as train
	for c in categories:
		count = 0
		txts = os.listdir(src_folder+'/'+c+'/')
		for f in txts[0:len(txts)*8/10]:
			if f.endswith('.txt'):
				count += 1
				if (count % 100 == 0):
					print c, count
				words_spaced = jieba_parse(src_folder+'/'+c+'/'+f)
				if words_spaced is not None:
					foutX_tr.write(words_spaced+'\n')
					foutY_tr.write(str(categories_index[c])+'\n')
	# last 2/10 as train
	for c in categories:
		count = 0
		txts = os.listdir(src_folder+'/'+c+'/')
		for f in txts[len(txts)*8/10+1:]:
			if f.endswith('.txt'):
				count += 1
				if (count % 100 == 0):
					print c, count
				words_spaced = jieba_parse(src_folder+'/'+c+'/'+f)
				if words_spaced is not None:
					foutX_te.write(words_spaced+'\n')
					foutY_te.write(str(categories_index[c])+'\n')
	foutX_tr.close()
	foutY_tr.close()
	foutX_te.close()
	foutY_te.close()

# gather X and Y, split into train and test
def gather_XY():
	X_train = [l 		for l in open('data_processed_train/X.txt')]
	Y_train = [int(l) 	for l in open('data_processed_train/Y.txt')]
	X_test = [l 		for l in open('data_processed_test/X.txt')]
	Y_test = [int(l) 	for l in open('data_processed_test/Y.txt')]
	assert(len(X_train) == len(Y_train))
	assert(len(X_test) == len(Y_test))
	return X_train, Y_train, X_test, Y_test


def train_and_benchmark(X_train, Y_train, X_test, Y_test, save_model_path=''):
	# train SVM
	from sklearn.feature_extraction.text import CountVectorizer
	from sklearn.feature_extraction.text import TfidfTransformer
	from sklearn.pipeline import Pipeline
	from sklearn.linear_model import SGDClassifier
	from sklearn import metrics
	import numpy as np
	text_clf = Pipeline([('vect', CountVectorizer()),
	                     ('tfidf', TfidfTransformer()),
	                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
	                                           alpha=1e-3, n_iter=5, random_state=42)),
	                     ])
	_ = text_clf.fit(X_train, Y_train)
	# benchmark train
	predicted = text_clf.predict(X_train)
	print 'Train acc = ', np.mean(predicted == Y_train)
	print (metrics.classification_report(Y_train, predicted,target_names=categories))
	print metrics.confusion_matrix(Y_train, predicted)
	# benchmark test
	predicted = text_clf.predict(X_test)
	print 'Test acc = ', np.mean(predicted == Y_test)
	print (metrics.classification_report(Y_test, predicted,target_names=categories))
	print metrics.confusion_matrix(Y_test, predicted)


def load_model():
	pass
	
def predict_text(string):
	pass


# prepare_XY()
X_train, Y_train, X_test, Y_test = gather_XY()
train_and_benchmark(X_train, Y_train, X_test, Y_test, '')



