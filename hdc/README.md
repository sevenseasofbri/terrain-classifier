For a given audio sample:
- extract the features like ZCR, RMS, etc
- Build a codebook with a basis vector for each of the features {ZCR, RMS ...}
- Build a level hypervector (V) for converting the feature values to HVs
- Encode the input as:
`audio_hv = ⊕(ZCR ⊙ V(0.12), RMS ⊙ V(0.83), ...)`

where ⊕ and ⊙ are bundling and binding operation.

Files:
- `test.ipynb`: Working HDC code for training a HDC classifier and making inferences from audio features
- `train.py`: Generated version of above python notebook with numpy-based feature extractors
- `audio/`: Sample audio files for training and testing with 3 classes
- `inference.c`: Generate version of above notebook for making inferences of the recorded audio. Uses CMSIS DSP and assumes the recordings and codebooks are on the storage
