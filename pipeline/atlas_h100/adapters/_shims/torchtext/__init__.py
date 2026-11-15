"""Pure-Python stand-in for the part of torchtext that scGPT imports (torchtext.vocab.Vocab).

torchtext 0.17 is compiled against torch 2.2 and its C++ extension does not load under newer torch.
scGPT uses it only for the gene vocabulary (a token <-> index map), so adapters/scgpt.py puts this
package on sys.path when the real torchtext fails to import. Model weights and the forward pass do
not touch it.
"""
