from .base.evoformer import evoformer_base as base_evoformer
from .openfold.evoformer import EvoformerBlock


def openfold_evoformer():
    return EvoformerBlock(
        c_m=256,
        c_z=128,
        c_hidden_msa_att=32,
        c_hidden_opm=32,
        c_hidden_mul=128,
        c_hidden_pair_att=32,
        no_heads_msa=8,
        no_heads_pair=4,
        transition_n=4,
        msa_dropout=0.15,
        pair_dropout=0.15,
        inf=1e4,
        eps=1e-4,
        is_multimer=False,
    ).eval()


__all__ = ["openfold_evoformer", "base_evoformer"]

