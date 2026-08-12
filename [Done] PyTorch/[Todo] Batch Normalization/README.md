## 5. Batch Normalization

Alright this one is classic. BatchNorm was the big deal for a long time (Ioffe & Szegedy 2015). The idea is basically: during training your activations can shift around a lot (internal covariate shift whatever), so we just normalize them using the mean and variance of the *current batch*.

For a feature across the batch:

mean = average of that feature over the batch  
var  = variance of that feature over the batch  

then:

x_hat = (x - mean) / sqrt(var + eps)

and finally:

out = gamma * x_hat + beta

So you have two learnable parameters per feature: gamma (scale) and beta (shift). That’s the full thing.

Important difference from LayerNorm / RMSNorm: BatchNorm looks *across the batch* for each feature, while LayerNorm/RMSNorm look *across the features* for each sample. That’s why BatchNorm is a bit annoying at inference time (you need running averages) and why it doesn’t play as nicely with small batches or sequence models.

- “batch” just means the group of samples you feed the model at the same time during one training step.

In real PyTorch you almost never write this yourself. You just do:

```python
bn = torch.nn.BatchNorm1d(num_features)
# or BatchNorm2d for images
```

and it handles the running mean/var for you during training vs eval.

That’s the core of it. Once you get this, LayerNorm and RMSNorm feel like simpler cousins that don’t depend on the batch size.
