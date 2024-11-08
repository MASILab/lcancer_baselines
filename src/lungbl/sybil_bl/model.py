from torch.nn import Module
from sybil import Sybil, Serie

class SybilPred(Module):
    def __init__(self):
        super().__init__()
        self.model = Sybil("sybil_base")

    def forward(self, m1):
        """
        m1: path to nifti file
        """
        serie = Serie(m1, file_type='nifti')
        latent, pred = self.model.hidden(serie)
        return pred, latent
        