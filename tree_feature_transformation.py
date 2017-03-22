from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomTreesEmbedding,RandomForestClassifier,GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

class TreeTransformClf(object):
	
	'''Tree feature transformation classifier: it users a tree classification technique and one hot encoding to transform the feature
	space and then fits a logistic regression using the newly generated features
	Parameters:
	------------
	tree_clf: the tree classification model used to transform the features (default is random_forest)

	subparameters: pass those directly into the classification method
		tree_params is a dictionary which contains the hyper parameters of the tree model build in stage 1
		Ex:n_estimators: number of trees to build (default is 10)
		   max_depth: the maximum depth of the tree (default is 3)
		random_state: random seed for the process
		logit_C: regularization parameter for the downstream logitstic regression (controlling how sparse the classifier is)
	'''

	def __init__(self,tree_clf=RandomForestClassifier,tree_params=dict(),logit_C=1.0,random_state=0):
		'''initialize'''
		#passing in the right parameters in the first stage model
		tree_params['random_state']=random_state
		self.tree_clf = tree_clf(**tree_params)
		self.encoder = OneHotEncoder()
		self.logit_clf = LogisticRegression(penalty='l1',C=logit_C)
		self.random_state=random_state

	def _fit_level_1_model(self,X,y):
		'''builds the level one model which is either a Random Forest, Random
		Tree Embedding or Gradient Boosted Trees.'''
		return self.tree_clf.fit(X,y)

	def _build_encoder(self,X):
		'''builds a one hot encoder using the pre-trained level 1 classifier
		it is inteded to be used with the data used in the level 1 model'''
		#if Random Forest is at level 1
		if isinstance(self.tree_clf, RandomForestClassifier):
			self.encoder.fit(self.tree_clf.apply(X))
			return self
		#if Random Tree Embedding is at level 1 - no need to do anything, Tree Embedding already does this by default
		if isinstance(self.tree_clf, RandomTreesEmbedding):
			pass
		#if Gradient Boosted Trees are at level 1
		if isinstance(self.tree_clf, GradientBoostingClassifier):
			self.encoder.fit(self.tree_clf.apply(X)[:,:,0])
			return self

	def _feature_transformer(self,X):
		'''method which uses a pre-built encoder to tansform the data using the 
		leaves of the pre-trained classifier
		Takes raw feature set X as an input'''
		if isinstance(self.tree_clf, RandomForestClassifier):
			return self.encoder.transform(self.tree_clf.apply(X))
		#if Random Tree Embedding is at level 1
		if isinstance(self.tree_clf, RandomTreesEmbedding):
			return self.tree_clf.transform(X)
		#if Gradient Boosted Trees are at level 1
		if isinstance(self.tree_clf, GradientBoostingClassifier):
			return self.encoder.transform(self.tree_clf.apply(X)[:,:,0])	

	def _fit_level_2_model(self,X,y):
		'''builds the level 2 model which is a sparse classifier (logit with L1)
		it is meant to take in data which was not seen by the level 1 model and pre-processes
		it with the one hot encoder
		inputs are an X matrix and a target vector
		the transformer converts X into a sparse matrix using the decision trees at level 1'''
		#if Random Forest is at level 1
		return self.logit_clf.fit(self._feature_transformer(X),y)

	def fit(self,X,y,split_size=0.5):
		
		'''trains the model - the data is split (equally by default) and the first half is used to build a tree classifier
		and the second half is used to train the logistic regression
		Parameters:
		----------
		X: feature set [n_samples,m_features]
		y: target vector [n_samples, 1]
		split_size: the proportion split between the data used for 1st and 2nd level model
		Returns:
		self.object
		'''
		#splitting the train data into two portions - X1 is used to train the tree classifier and X2 is used to train the logistic regression
		X1,X2,y1,y2 = train_test_split(X,y,test_size=split_size,random_state=self.random_state)
		
		#building the level 1 model
		self._fit_level_1_model(X1,y1)

		#building the onehot encoder using the first level data and passing it through the trained decision trees
		self._build_encoder(X1)

		#building the level 2 model
		self._fit_level_2_model(X2,y2)

	def predict(self,X):
			
		'''makes a class prediction using the model
		Parameters:
		-----------
		X: feature set (usually unseen data) format: [n_samples, m_features]
		Returns:
		y_pred: an array of predicted classes for the target variable'''
		
		return self.logit_clf.predict(self._feature_transformer(X))	
	
	def predict_proba(self,X):

		'''makes a probability prediction using the model
		Parameters:
		-----------
		X: feature set (usually unseen data) format: [n_samples, m_features]
		Returns:
		y_pred: an array of predicted probabilities for the target variable'''

		return self.logit_clf.predict_proba(self._feature_transformer(X))



