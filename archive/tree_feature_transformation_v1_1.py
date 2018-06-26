from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomTreesEmbedding,RandomForestClassifier,GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline, Pipeline
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

	def __init__(self,tree_clf=RandomForestClassifier,tree_params=dict(),logit_C=1.0,random_state=0,cv_stack='No'):
		'''initialize'''
		#passing in the right parameters in the first stage model
		tree_params['random_state']=random_state
		self.tree_clf = tree_clf(**tree_params)
		self.encoder = OneHotEncoder()
		self.logit_clf = LogisticRegression(penalty='l1',C=logit_C)
		self.random_state=random_state
		self.cv_stack = cv_stack

	def fit(self,X,y,split_size=0.5):
		
		if self.cv_stack == 'No':
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
			
			#In the case of using a random forest
			if isinstance(self.tree_clf, RandomForestClassifier):
				#build a tree
				self.tree_clf.fit(X1,y1)
				#build an encoder on the tree leaves (apply-function returns leaves)
				self.encoder.fit(self.tree_clf.apply(X1))
				#use the encoder on new unseen data as a feature set to train a logistic classifier
				self.logit_clf.fit(self.encoder.transform(self.tree_clf.apply(X2)),y2)
				return self
			#In the case of using Random Trees Embedding
			if isinstance(self.tree_clf, RandomTreesEmbedding):
				self.tree_clf.fit(X1,y1)
				self.logit_clf.fit(self.tree_clf.transform(X2),y2)
				return self
			#In the case of Gradient Boosting
			if isinstance(self.tree_clf, GradientBoostingClassifier):
				self.tree_clf.fit(X1,y1)
				self.encoder.fit(self.tree_clf.apply(X1)[:,:,0])
				self.logit_clf.fit(self.encoder.transform(self.tree_clf.apply(X2)[:,:,0]),y2)

		if self.cv_stack == 'Yes':
			

	def predict(self,X):
			
		if self.cv_stack == 'No':	
			'''makes a class prediction using the model
			Parameters:
			-----------
			X: feature set (usually unseen data) format: [n_samples, m_features]
			Returns:
			y_pred: an array of predicted classes for the target variable'''
			
			#In the case of using a random forest
			if isinstance(self.tree_clf, RandomForestClassifier):
				return self.logit_clf.predict(self.encoder.transform(self.tree_clf.apply(X)))
			#In the case of using Random Trees Embedding
			if isinstance(self.tree_clf, RandomTreesEmbedding):
				return self.logit_clf.predict(self.tree_clf.transform(X))
			#In the case of Gradient Boosting
			if isinstance(self.tree_clf, GradientBoostingClassifier):
				return self.logit_clf.predict(self.encoder.transform(self.tree_clf.apply(X)[:,:,0]))
	
	def predict_proba(self,X):

		if self.cv_stack == 'No':
			'''makes a probability prediction using the model
			Parameters:
			-----------
			X: feature set (usually unseen data) format: [n_samples, m_features]
			Returns:
			y_pred: an array of predicted probabilities for the target variable'''

			#In the case of using a random forest
			if isinstance(self.tree_clf, RandomForestClassifier):
				return self.logit_clf.predict_proba(self.encoder.transform(self.tree_clf.apply(X)))
			#In the case of using Random Trees Embedding
			if isinstance(self.tree_clf, RandomTreesEmbedding):
				return self.logit_clf.predict_proba(self.tree_clf.transform(X))
			#In the case of Gradient Boosting
			if isinstance(self.tree_clf, GradientBoostingClassifier):
				return self.logit_clf.predict_proba(self.encoder.transform(self.tree_clf.apply(X)[:,:,0]))

