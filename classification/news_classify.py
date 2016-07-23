# coding=UTF-8
import re # regular expression
import jieba as jb
import os
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
import codecs

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
import numpy as np


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



def train_and_benchmark(X_train, Y_train, X_test, Y_test, save_model_path=None):
	# train SVM
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
	# save model
	if save_model_path:
		from sklearn.externals import joblib
		joblib.dump(text_clf, save_model_path)



# load the model for the classifier
def load_clf(save_model_path):
	from sklearn.externals import joblib
	text_clf = joblib.load(save_model_path)
	return text_clf

def predict_testset(save_model_path, X_test, Y_test):
	text_clf = load_clf(save_model_path)
	# benchmark test
	predicted = text_clf.predict(X_test)
	print 'Test acc = ', np.mean(predicted == Y_test)
	print (metrics.classification_report(Y_test, predicted,target_names=categories))
	print metrics.confusion_matrix(Y_test, predicted)





''' Training code '''
# prepare_XY()
# X_train, Y_train, X_test, Y_test = gather_XY()
# save_model_path = 'model/text_clf_v1.model.pkl'
# train_and_benchmark(X_train, Y_train, X_test, Y_test, save_model_path) # training and benchmark
# predict_testset(save_model_path, X_test, Y_test) # predict on testing set




''' Deploy code '''
class NewsClassifier:
	def __init__(self,save_model_path):
		# config jieba
		jb.load_userdict('resource/location+words.txt')

		# params
		self.categories = [
			'china_world','entertainment','finance','lifestyle','news','sport']
		self.categories_index = {
			'china_world':0,'entertainment':1,'finance':2,'lifestyle':3,'news':4,'sport':5}
		self.text_clf = self.__load_clf(save_model_path)

	def predict(self,string):
		string_preprocessed = self.__preprocess_text(string)
		prediction = self.text_clf.predict([string_preprocessed])
		category = self.categories[prediction]
		return category

	def __load_clf(self,save_model_path):
		from sklearn.externals import joblib
		text_clf = joblib.load(save_model_path)
		return text_clf

	def __preprocess_text(self,txt):
		# replace punctuations with space
		txt = re.sub(
			ur"[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）：；《）《》“”()»〔〕-]+",
			" ", txt.decode("utf8"))
		words = jb.cut(txt, cut_all=False)
		words_spaced = ' '.join(words)
		return words_spaced

# string = '<匯港通訊> 新宇環保(08068)獲批准將於主板上市及從創業板撤銷上市之股份轉板上市。股份於創業板買賣(股票代號:8068)的最後一日將為二零一六年七月二十九日。股份將於二零一六年八月一日上午九時正開始於主板買賣(股票代號:436)。'
string = '藝人劉佩玥（Moon）與「宅男女神」陳瀅老友鬼鬼，經常糖黐豆一齊行街。前日黃昏六時許，兩女相約到九龍一個商場閒逛，吸引不少途人注目。初時兩人漫無目的邊行邊傾，未知是否因為陳瀅剛與賭王四房兒子何猷亨分手而心情差，故找劉佩玥傾訴呢？記者上前問陳瀅心情如何？她笑笑口表示無受情傷影響：「冇情傷過，不嬲好開心，啱啱食完嘢散吓步。」再問她是否隨時迎接第二春？她即拉住身旁的劉佩玥笑說：「佢咪係我第二春囉！」還懂得說笑，相信陳瀅心情不俗。'
news_clf = NewsClassifier('model/text_clf_v1.model.pkl')
print 'Category = ', news_clf.predict(string)






