

class LearningModule(object):

    def __init__(self):
        self._model = None
        self._phase = 'eval'
    
    @property
    def phase(self):
        return self._phase
    
    @phase.setter
    def phase(self, new_phase):
        if type(new_phase) != str:
            raise TypeError('Phase new_phase must be a string.')
        if not new_phase in ['train', 'eval']:
            raise ValueError('Phase new_phase must be in {"train", "eval"}') 
        
        self._phase = new_phase
        
        
    def _build(self):
        raise NotImplementedError


    def __call__(self, input_fn):
        raise NotImplementedError