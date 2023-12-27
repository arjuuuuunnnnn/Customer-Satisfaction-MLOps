from zenml.steps import BaseParameters
class ModelNameConfig(BaseParameters):
    '''
    Model configs
    '''
    model_name: str = "LinearRegression"