import torch
import torch.nn as nn

class ModelTemplate(nn.Module):
    def __init__(self,parameter1,parameter2) -> None:
        super(ModelTemplate,self).__init__()
    
    @staticmethod
    def get_argument_of_this_model() -> dict:
        from HParams import HParams
        h_params = HParams()
        model_argument:dict = h_params.model.ModelTemplate
        model_argument["parameter1"] = h_params.preprocess.parameter1
        model_argument["parameter2"] = h_params.preprocess.parameter2
        return model_argument

    def get_test_input(self):
        return torch.rand((4,2,72000))