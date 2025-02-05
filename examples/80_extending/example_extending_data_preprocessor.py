"""
=======================================================
Extending Auto-Sklearn with Data Preprocessor Component
=======================================================

The following example demonstrates how to turn off data preprocessing step in auto-skearn.
"""

import autosklearn.classification
import autosklearn.pipeline.components.data_preprocessing
import sklearn.metrics
from ConfigSpace.configuration_space import ConfigurationSpace
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import SPARSE, DENSE, UNSIGNED_DATA, INPUT
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


############################################################################
# Create NoPreprocessing component for auto-sklearn
# =================================================
class NoPreprocessing(AutoSklearnPreprocessingAlgorithm):

    def __init__(self, **kwargs):
        """ This preprocessors does not change the data """
        self.preprocessor = None

    def fit(self, X, Y=None):
        self.preprocessor = 0
        self.fitted_ = True
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return X

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            'shortname': 'no',
            'name': 'NoPreprocessing',
            'handles_regression': True,
            'handles_classification': True,
            'handles_multiclass': True,
            'handles_multilabel': True,
            'handles_multioutput': True,
            'is_deterministic': True,
            'input': (SPARSE, DENSE, UNSIGNED_DATA),
            'output': (INPUT,)
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs


# Add NoPreprocessing component to auto-sklearn.
autosklearn.pipeline.components.data_preprocessing.add_preprocessor(NoPreprocessing)

############################################################################
# Create dataset
# ==============

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

############################################################################
# Fit the model without performing data preprocessing
# ===================================================

clf = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=120,
    include={
        'data_preprocessor': ['NoPreprocessing']
    },
    # Bellow two flags are provided to speed up calculations
    # Not recommended for a real implementation
    initial_configurations_via_metalearning=0,
    smac_scenario_args={'runcount_limit': 5},
)
clf.fit(X_train, y_train)

############################################################################
# Print prediction score and statistics
# =====================================

y_pred = clf.predict(X_test)
print("accuracy: ", sklearn.metrics.accuracy_score(y_pred, y_test))
print(clf.show_models())
