import packages.utils.data_utils as du

def load_folded_data(data_dir):
    ecfp_tr, ic50_tr, ecfp_va, ic50_va = du.load_data(data_dir)
    ecfp_tr = du.fold_input(ecfp_tr, 32000)
    ecfp_va = du.fold_input(ecfp_va, 32000)
    return ecfp_tr, ic50_tr, ecfp_va, ic50_va
