# SIMP

Simple Integer Matched Partitions for quick clustering in high dimensions.

## Usage

The main entry point is the `cluster` method, which accepts a multidimensional numpy array and returns a list of cluster assignments for each index of the first dimension.

For example, if the `data` is shape `(100, 25)` then the output of `cluster` will be an array of shape `(100, )` with the first element being the partition that the first 25 length vector in the data is assigned to.

The following code snippit should be enough to get you started.

```python
    from collections import defaultdict
    import numpy as np
    import pandas as pd
    import simp

    data = np.random.normal(size=(10000, 512))
    cluster_indexes = simp.cluster(data)

    clusters = defaultdict(list)
    for j, i in enumerate(cluster_indexes):
        clusters[i].append(data[j])

    tidy = []
    for c in clusters.items():
        cluster_name = c[0]
        tidy += [(cluster_name, i) for i in c[1]]
    cluster_df = pd.DataFrame(tidy, columns=["clustered", "original"])
    cluster_df.to_csv("clustered.csv")
```
