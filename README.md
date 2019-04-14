# pytorch_group_sampler
PyTorch sampler for orthogonal and special orthogonal groups

This is meant to reproduce the functionality of `scipy.stats.`[`ortho_group.rvs()`, `special_ortho_group.rvs()`].

Useful for dimensions > 100s, in float32 on GPU. Can be ~25x faster for float32, 2x faster for float64. There is more error working in float32 than float64.

For dimensions not larger than a few 100, this will be slower.
