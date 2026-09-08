"""scGPT adapter (Cui et al. 2023, whole-human) — for REPRODUCING Igor's result with OUR
pipeline on HIS exact model. scGPT is a gene-token expression MLM (perturbation-responsive),
so the CRISPRi perturbation test is valid here (unlike UCE). Wraps scgpt.model.TransformerModel
(do NOT reimplement) + Igor's tokenisation (src/data/scgpt_dataset.py) and captures per-gene
residual by hooking transformer_encoder.layers[L].

KEY: load flash-trained whole-human weights into a STANDARD nn.TransformerEncoder by setting
use_fast_transformer=False — scgpt.utils.load_pretrained remaps Wqkv.->in_proj_ automatically,
so .layers[L] are hookable with no flash-attn dependency (CPU or GPU).

REQUIRES on the cluster:
  * the scGPT repo on PYTHONPATH (pip install scgpt, or repo path) for scgpt.model.TransformerModel,
    scgpt.tokenizer.GeneVocab, scgpt.utils.load_pretrained
  * checkpoint dir with best_model.pt + vocab.json + args.json (whole-human):
    pass ckpt_dir=<path>. Dims read from args.json (nlayers 12, d_model 512, nhead 8, n_bins 51).

VERIFY on H100 (expect 1-2 tweaks, like the uce/tgpt adapters):
  * GeneVocab API: vocab[token] / vocab.get_stoi() / get_itos() — adjust if the installed scgpt
    version differs. * binning: replicates scgpt.preprocess (per-cell quantile bins 1..n_bins-1);
    if Igor's preprocess log-normalises first, mirror that. * confirm forward returns and that the
    hook fires on transformer_encoder.layers (standard encoder).
"""
from __future__ import annotations
import json, os, sys
import numpy as np
from .base import Adapter


def _bin_row(values: np.ndarray, n_bins: int) -> np.ndarray:
    """scGPT per-cell binning of nonzero expression into 1..n_bins-1 (quantile digitize)."""
    if values.size == 0:
        return values
    bins = np.quantile(values, np.linspace(0, 1, n_bins - 1))
    # left digitize with tie handling, matching scgpt._digitize spirit
    idx = np.digitize(values, bins)
    return idx.astype(np.float32)


class ScGPTAdapter(Adapter):
    name = "scGPT"
    d_model = 512

    def __init__(self, ckpt_dir="external/scGPT_checkpoints/whole-human",
                 scgpt_repo=None, layers=None, max_seq_len=1200):
        self.ckpt_dir = ckpt_dir
        self.scgpt_repo = scgpt_repo
        self._layers_override = layers          # None -> depth_matched after load
        self.max_seq_len = max_seq_len

    @staticmethod
    def _stub_optional_deps():
        """scgpt/__init__ eagerly imports scbank(->datasets), tasks(->seaborn/networkx),
        trainer(->wandb) — none needed for TransformerModel. Stub any that are missing so the
        import succeeds without polluting the env (avoids datasets pulling a newer huggingface-hub)."""
        import types, importlib, importlib.machinery
        for name in ("datasets", "wandb", "seaborn", "networkx", "torch_geometric"):
            try:
                importlib.import_module(name)
            except Exception:
                m = types.ModuleType(name)
                m.__path__ = []
                m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)  # else find_spec() raises
                m.__getattr__ = lambda n: type(n, (), {})   # any `from mod import X` -> dummy class
                sys.modules[name] = m

    def load(self, device="cuda"):
        import torch
        if self.scgpt_repo:
            sys.path.insert(0, self.scgpt_repo)
        self._stub_optional_deps()
        from scgpt.model import TransformerModel
        from scgpt.tokenizer import GeneVocab
        from scgpt.utils import load_pretrained
        self.torch = torch; self.device = device

        args = json.load(open(os.path.join(self.ckpt_dir, "args.json")))
        vocab = GeneVocab.from_file(os.path.join(self.ckpt_dir, "vocab.json"))
        self.vocab = vocab
        self.pad_token = args.get("pad_token", "<pad>")
        for sp in (self.pad_token, "<pad>", "<cls>", "<eoc>"):
            if sp and sp not in vocab:
                vocab.append_token(sp)
        self.pad_id = vocab[self.pad_token]
        self.n_bins = int(args["n_bins"])
        self.d_model = int(args["embsize"])
        self.n_layers = int(args["nlayers"])
        from common.layers import depth_matched
        self.layers = tuple(self._layers_override) if self._layers_override else depth_matched(self.n_layers)

        model_args = dict(
            ntoken=len(vocab), d_model=int(args["embsize"]), nhead=int(args["nheads"]),
            d_hid=int(args["d_hid"]), nlayers=int(args["nlayers"]),
            nlayers_cls=int(args.get("n_layers_cls", 3)), n_cls=1, vocab=vocab,
            dropout=float(args.get("dropout", 0.2)), pad_token=self.pad_token,
            pad_value=int(args.get("pad_value", -2)), do_mvc=bool(args.get("MVC", False)),
            do_dab=False, use_batch_labels=False, domain_spec_batchnorm=False,
            input_emb_style=args.get("input_emb_style", "continuous"),
            n_input_bins=int(args["n_bins"]),
            cell_emb_style="avg-pool" if args.get("no_cls") else "cls",
            explicit_zero_prob=False, use_fast_transformer=False,   # standard encoder -> hookable, weights remapped
            fast_transformer_backend="flash", pre_norm=False)
        model = TransformerModel(**model_args)
        state = torch.load(os.path.join(self.ckpt_dir, "best_model.pt"), map_location="cpu")
        state = state.get("model", state.get("state_dict", state)) if isinstance(state, dict) else state
        load_pretrained(model, state, verbose=False)   # remaps Wqkv.->in_proj_ since use_fast_transformer=False
        self.model = model.eval().to(device)
        # disable TransformerEncoder's NestedTensor fast-path: in eval + padding-mask it converts
        # layer I/O to NestedTensor, which breaks per-layer hooks (.numpy() on nested fails)
        try:
            self.model.transformer_encoder.enable_nested_tensor = False
        except Exception:
            pass
        # gene symbol lookup (vocab index -> symbol)
        self.itos = np.array(vocab.get_itos())
        # the encoder layers to hook
        self.enc_layers = self.model.transformer_encoder.layers

    def _tokenise(self, xrow):
        """One cell -> (gene_ids, binned_values, gene_indices) for nonzero, in-vocab, expr-sorted genes."""
        import numpy as np
        v = xrow.toarray().ravel() if hasattr(xrow, "toarray") else np.asarray(xrow).ravel()
        nz = np.where(v > 0)[0]
        vals = v[nz]
        order = np.argsort(-vals)
        nz = nz[order]; vals = vals[order]
        # map to vocab ids via gene symbols; keep only in-vocab
        gids, keepidx, keepvals = [], [], []
        for gi, val in zip(nz, vals):
            g = self._var[gi]
            if g in self.vocab:
                gids.append(self.vocab[g]); keepidx.append(gi); keepvals.append(val)
        gids = np.array(gids[: self.max_seq_len], dtype=np.int64)
        keepidx = np.array(keepidx[: self.max_seq_len], dtype=np.int64)
        binned = _bin_row(np.array(keepvals[: self.max_seq_len], dtype=np.float32), self.n_bins)
        return gids, binned, keepidx

    def iter_activations(self, adata, batch_size=16):
        import torch, numpy as np
        self.processed_obs = adata.obs.reset_index(drop=True)
        self._var = np.array([str(x) for x in adata.var_names])
        caps = {}
        hooks = [self.enc_layers[L].register_forward_hook(
            (lambda L: (lambda m, i, o: caps.__setitem__(L, (o[0] if isinstance(o, tuple) else o).detach())))(L))
            for L in self.layers]
        cid = 0
        for s in range(0, adata.n_obs, batch_size):
            rows = [self._tokenise(adata.X[j]) for j in range(s, min(s + batch_size, adata.n_obs))]
            b = len(rows)
            L = max(1, max(len(r[0]) for r in rows))
            gid = np.full((b, L), self.pad_id, dtype=np.int64)
            val = np.zeros((b, L), dtype=np.float32)
            gidx = np.full((b, L), -1, dtype=np.int64)
            for i, (g, vv, gi) in enumerate(rows):
                gid[i, :len(g)] = g; val[i, :len(vv)] = vv; gidx[i, :len(gi)] = gi
            src = torch.as_tensor(gid, device=self.device)
            values = torch.as_tensor(val, device=self.device)
            mask = src == self.pad_id
            caps.clear()
            with torch.no_grad():
                self.model(src=src, values=values, src_key_padding_mask=mask)
            keep = (~mask.cpu().numpy())                       # [b, L] non-pad gene tokens
            syms, cids = [], []
            for i in range(b):
                pos = np.where(keep[i])[0]
                gi = gidx[i, pos]
                syms.append(np.array([str(self._var[k]).upper() if k >= 0 else "" for k in gi]))
                cids.append(np.full(len(pos), cid + i))
            syms = np.concatenate(syms); cids = np.concatenate(cids)
            acts = {}
            for Ly in self.layers:
                h = caps[Ly]
                if getattr(h, "is_nested", False):             # defensive: nested -> padded
                    h = h.to_padded_tensor(0.0)
                h = h.float().cpu().numpy()                    # [b, L, d]
                acts[Ly] = np.concatenate([h[i, np.where(keep[i])[0], :] for i in range(b)], 0)
            cid += b
            yield acts, syms, cids
        for hk in hooks:
            hk.remove()
