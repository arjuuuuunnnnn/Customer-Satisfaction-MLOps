import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
class Model(ABC):
    '''
    Abstract class for all models
    '''
    @abstractmethod
    def train(self, X_train, y_train):
        '''
        :param X_train:
        :param y_train:
        :return: none
        '''
        pass

class LinearRegressionModel(Model):
    '''
    linear regression model
    '''
    def train(self, X_train, y_train, **kwargs):
        '''
        trains the model
        :param X_train:
        :param y_train:
        :return:
        '''
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model training completed")
            return reg
        except Exception as e:
            logging.error(f"Error while training model: {e}")
            raise e