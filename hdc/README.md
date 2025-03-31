For a given audio sample:
- extract the features like ZCR, RMS, etc.
- Build a codebook with a basis vector for each of the features {ZCR, RMS ...}.
- Build a level hypervector (V) for converting the feature values to HVs.
- Encode the input as:
`audio_hv = ⊕(ZCR ⊙ V(0.12), RMS ⊙ V(0.83), ...)`

where ⊕ and ⊙ are bundling and binding operation.
