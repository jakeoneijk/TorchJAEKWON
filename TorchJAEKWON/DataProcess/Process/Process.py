from abc import ABC, abstractmethod

from HParams import HParams

class Process(ABC):

    def __init__(self):
        super().__init__()
        self.h_params = HParams()

    @abstractmethod
    def preprocess_input_for_inference(self,input,additional_dict=None):
        raise NotImplementedError

    @abstractmethod
    def preprocess_training_data(self,data_dict,additional_dict=None):
        raise NotImplementedError