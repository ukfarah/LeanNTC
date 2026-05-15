## Quickstart

### Colab
1. Open `NTC.ipynb` in Colab.
2. Run cells top to bottom. The notebook will mount Google Drive, download the DTD dataset, train, and visualize results.

### Local
```bash
pip install torch torchvision numpy matplotlib pillow tqdm tensorboard requests
jupyter notebook NTC.ipynb
```
A CUDA GPU is recommended but not required. Training 10k iterations on a T4 takes a few minutes.

## Configuration

All hyperparameters live in the `Config` class:

| Setting | Default | Meaning |
|---|---|---|
| `BASE_RESOLUTION` | 16 | resolution of the coarsest grid |
| `NUM_LEVELS` | 4 | number of pyramid levels (each doubles resolution) |
| `LATENT_CHANNELS_LIST` | `[2,2,2,1]` | channels per level |
| `MLP_HIDDEN` | 48 | hidden width of the decoder |
| `BATCH_SIZE` | 32768 | UV samples per step |
| `NUM_ITERATIONS` | 10000 | total training steps |
| `LEARNING_RATE` | 0.01 | MLP learning rate (grids use 10× this) |
| `QUANT_BITS` | 8 | bit-depth for grid quantization |
| `QUANT_START_FRAC` / `END_FRAC` | 0.2 / 0.6 | progressive quantization schedule |

## Results

On a single DTD texture (256×256 RGB):

| Metric | Value |
|---|---|
| Parameters | ~30,019 |
| Compression ratio | ~23.9× vs. FP32 RGB |
| PSNR | ~44.95 dB |
| Training time (T4) | ~3 min for 10k iters |

## How it works

1. **Sample.** Pick random `(u, v)` coordinates and look up the ground-truth color via bilinear interpolation on the source texture.
2. **Encode.** Each pyramid grid is bilinearly sampled at `(u, v)` and the per-level feature vectors are concatenated into a 7-D latent.
3. **Decode.** The MLP maps the latent to RGB.
4. **Quantize.** After warmup, grid values are progressively quantized to 8 bits using a straight-through estimator so gradients can still flow.
5. **Train.** L2 loss between predicted and ground-truth color, Adam optimizer, StepLR schedule, gradient clipping.

## Roadmap / ideas

- Multi-texture training with a shared decoder
- Hash-grid encoding instead of dense grids
- Mipmap-aware sampling for filtered lookups
- Block-based decoding for GPU-friendly random access
- Per-texel importance weighting for better detail preservation

## License

MIT — feel free to reuse, modify, and extend.

## Acknowledgements

- DTD: *Describable Textures Dataset* (Cimpoi et al., 2014)
- The NVIDIA NTC paper for the overall framing of texture compression as a learned neural field
