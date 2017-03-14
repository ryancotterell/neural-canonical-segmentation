This is the code for the canonical segmenter presented in:
Katharina Kann, Ryan Cotterell and Hinrich Schütze. Neural Morphological Analysis: Encoding-Decoding Canonical Segments. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, Austin, USA, November 2016.

Usage:
- Train the encoder-decoder on the segmentation data.
- Test several times to sample from the defined distribution.
- Store the samples in a pickle file.
- Run the reranker using the provided script.