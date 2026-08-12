# RMS Normalization (the one modern LLMs actually use)

Okay so you already know BatchNorm and LayerNorm right? Cool. Modern stuff like Llama, Mistral and pretty much every decent LLM these days just uses something simpler called RMSNorm. Paper is from Zhang & Sennrich 2019 if you care.

LayerNorm does this whole thing: subtract the mean, divide by std, then multiply by gamma and add beta. RMSNorm is like "eh, do we really need the mean subtraction?" and just throws that part away. Also no beta at all. Only gamma.

The formula is stupidly simple:

RMS(x) = sqrt( mean(x²) + eps )

then output = (x / RMS(x)) * gamma

That’s it. No centering. Fewer parameters, less memory, works great.

Here’s a pure python version of what you’d implement:

```python
def rms_norm(x, gamma, eps=1e-5):
    # x and gamma are lists of floats, same length
    mean_square = sum(val ** 2 for val in x) / len(x)
    rms = (mean_square + eps) ** 0.5
    return [round((val / rms) * g, 4) for val, g in zip(x, gamma)]
```

Quick check with the example they give:

```python
x = [1.0, 2.0, 3.0]
gamma = [1.0, 1.0, 1.0]
print(rms_norm(x, gamma))
# [0.4629, 0.9258, 1.3887]
```

Why does this work? Because for the values [1, 2, 3] the root mean square is basically sqrt((1+4+9)/3) = sqrt(14/3) ≈ 2.1602. Divide each number by that and you get those numbers. Since gamma is all 1s it stays the same.

In actual PyTorch you’d just use `torch.nn.RMSNorm` these days (or write your own with `torch.rsqrt` if you want to be fancy). But understanding the pure math version is good so you know what’s actually happening under the hood.

