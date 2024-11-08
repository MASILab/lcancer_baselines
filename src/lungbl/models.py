

from lungbl.liao.model import CaseNet
from lungbl.sybil_bl.model import SybilPred
from lungbl.tdvit.model import MaskedFeatViT, FeatViT
from lungbl.dlstm.model import DisCRNNClassifier
from lungbl.dls.model import MultipathModelBL

MODELS = {
    "liao": lambda : CaseNet(),
    "sybil": lambda : SybilPred(),
    "tavit": lambda : MaskedFeatViT(
        num_feat=5,
        feat_dim=128,
        num_classes=2,
        dim=128,
        depth=4,
        heads=4,
        mlp_dim=2*128,
        qkv_bias=False,
        dropout=0.1,
    ),
    "tevit_masked": lambda: MaskedFeatViT(
        num_feat=5,
        feat_dim=128,
        num_classes=2,
        dim=128,
        depth=4,
        heads=4,
        mlp_dim=2*128,
        qkv_bias=False,
        dropout=0.1,
        time_embedding="AbsTimeEncoding",
        time_aware=False,
    ),
    "tevit": lambda : FeatViT(
        num_feat=5,
        feat_dim=128,
        num_classes=2,
        dim=128,
        depth=4,
        heads=4,
        mlp_dim=2*128,
        qkv_bias=False,
        dropout=0.1,
        time_embedding="AbsTimeEncoding",
    ),
    "dlstm": lambda : DisCRNNClassifier(
        time=2,
        drop=True,
        mode="infor_exp",
        dim=128,
    ),
    "dls": lambda: MultipathModelBL(1),
}
def init_model(config_name):
    if config_name in MODELS:
        return MODELS[config_name]()
    else:
        raise ValueError(f'Model name [{config_name}] not implemented')
    
if __name__ == "__main__":
    print('h')