# SIMP

Simple Integer Matched Partitions for quick clustering in high dimensions.

## Usage

The main entry point is the `cluster` method, which accepts a multidimensional numpy array and returns a list of cluster assignments equal to the length of the first dimension.

```python
    import numpy as np
    import simp

    data = np.random((10000, 512))
    cluster_indexes = simp.cluster(data)
```