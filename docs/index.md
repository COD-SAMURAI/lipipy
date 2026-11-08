# lipipy

Readable, verified-from-scratch implementations of machine learning, deep learning
and modern research architectures.

Every algorithm is **readable** (the math is in the docstring), **verified** (tested
for parity against scikit-learn or PyTorch) and **explained** (paired with a notebook).

```bash
pip install -e ".[dev]"
```

```python
from lipipy.ml import LinearRegression

model = LinearRegression().fit(X, y)
```
