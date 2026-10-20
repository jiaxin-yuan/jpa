"""Run SuTraN and ED-LSTM inference on val and test from the stored checkpoints -> results/seq2seq."""
import os
import pickle
import sys

import torch

import config as C

sys.path.insert(0, os.path.join(C.ROOT, "models", "seq2seq"))
S2S_OUT = os.path.join(C.RESULTS, "seq2seq")
TSS_INDEX = 0


def meta(ds):
    """Activity count, number of prefix categoricals and de-standardisation constants."""
    base = os.path.join(C.S2S_DATA, ds)

    def load(name):
        with open(os.path.join(base, f"{ds}_strict_{name}.pkl"), "rb") as f:
            return pickle.load(f)
    card, cat, mu, sd = load("cardin_dict"), load("cat_cols_dict"), \
        load("train_means_dict"), load("train_std_dict")
    return {"num_activities": card["concept:name"] + 2,
            "n_cat": len(cat["prefix_df"]),
            "ttne": [mu["timeLabel_df"][0], sd["timeLabel_df"][0]],
            "tsp": [mu["suffix_df"][1], sd["suffix_df"][1]],
            "tss": [mu["suffix_df"][0], sd["suffix_df"][0]],
            "rrt": [mu["timeLabel_df"][1], sd["timeLabel_df"][1]]}


def slice_nda(dataset, n_cat):
    """Keep activity and [ts_start, ts_prev] of the prefix (no-data-attribute variant)."""
    return (dataset[n_cat - 1], dataset[n_cat][:, :, TSS_INDEX:TSS_INDEX + 2]) \
        + dataset[n_cat + 1:]


def build(model, m, device):
    """Model with the benchmark's hyper-parameters, and its inference loop."""
    if model == "SuTraN":
        from SuTraN.SuTraN import SuTraN_no_context
        from SuTraN.inference_procedure import inference_loop
        net = SuTraN_no_context(num_activities=m["num_activities"], d_model=32,
                                num_prefix_encoder_layers=4, num_decoder_layers=4,
                                num_heads=8, d_ff=128, dropout=0.2,
                                remaining_runtime_head=True, layernorm_embeds=True,
                                outcome_bool=False)
    else:
        from LSTM_seq2seq.enc_dec_lstm import EncDecLSTM_no_context
        from LSTM_seq2seq.inference_procedure import inference_loop
        net = EncDecLSTM_no_context(num_activities=m["num_activities"], d_model=64,
                                    n_layers=4, dropout=0.2)
    return net.to(device), inference_loop


def run(ds, device):
    """Write <split>/ result tensors for both models on one log."""
    torch.manual_seed(0)
    m = meta(ds)
    for split in ("val", "test"):
        data = slice_nda(torch.load(os.path.join(C.S2S_DATA, ds, f"{split}_tensordataset.pt")),
                         m["n_cat"])
        for model in ("SuTraN", "ED-LSTM"):
            out = os.path.join(S2S_OUT, ds, model, split)
            os.makedirs(out, exist_ok=True)
            net, loop = build(model, m, device)
            ck = torch.load(os.path.join(C.S2S_CKPT, ds, f"{model}.pt"), map_location=device)
            net.load_state_dict(ck["model_state_dict"])
            net.eval()
            with torch.no_grad():
                if model == "SuTraN":
                    loop(net, data, True, False, 1, m["ttne"], m["tsp"], m["tss"], m["rrt"],
                         results_path=out, val_batch_size=2048)
                else:
                    loop(net, data, 1, m["ttne"], m["tsp"], m["tss"], m["rrt"],
                         results_path=out, val_batch_size=2048)
            print(f"{ds} {model} {split} -> {out}")


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for d in sys.argv[1:] or C.LOGS:
        run(d, dev)
