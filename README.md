# AUM

Pytorch Library for Area Under the Margin (AUM) Ranking, as proposed in the paper:
[Identifying Mislabeled Data using the Area Under the Margin Ranking](https://arxiv.org/pdf/2001.10528.pdf)

## Install

`pip install -U AUM`

## Usage

Instantiate an AUMCalculator object:

```python
from aum import AUMCalculator

save_dir = '~/Desktop'
aum_calculator = AUMCalculator(save_dir, compressed=True)
```
Note: you can set `compressed` to `False` if you want to store the AUM metrics at every call to the update method. This will require considerably more space, however.

You can then update aum rankings on batches of data during training with:

```python
model.train()
for batch in loader:
    inputs, targets, sample_ids = batch

    logits = model(inputs)

    records = aum_calculator.update(logits, targets, sample_ids)

    ...
```

`records` is a dictionary mapping a sample_id to an `AUMRecord` containing the information below, including the AUM for the sample at this point in time.

```python
@dataclass
class AUMRecord:
    """
    Class for holding info around an aum update for a single sample
    """
    sample_id: Optional[int, str]
    num_measurements: int
    target_logit: int
    target_val: float
    other_logit: int
    other_val: float
    margin: float
    aum: float
```

And once you are done training, you can generate a csv of ranked samples with their aum scores with:

```python
aum_calculator.finalize()
```

If you have a dataset that does not return sample_ids, you can wrap it in `DatasetWithIndex`. The last element of the tuple returned for a given sample will be its sample_id.
```python
from aum import DatasetWithIndex
from torch.utils.data import Dataset

my_dataset = Dataset(...)
my_dataset_with_index = DatasetWithIndex(my_dataset)
```

## Example Outputs
Calling `finalize()` on an AUMCalculator will result in the creation of 1 or 2 csv files, depending if `compressed` was set to True or False.

If AUMCalculator was instantiated with `compressed = True`, you will find a csv file titled `aum_values.csv` in the following format:

| sample_id | aum    |
|-----------|--------|
| sample_1  | 1.205  |
| sample_3  | 1.145  |
| sample_2  | -3.785 |

If AUMCalculator was instantiated with `compressed = False`, you will find a csv file titled `full_aum_records.csv` in addition to the `aum_values.csv`. `full_aum_records.csv` is in the following format:

| sample_id | num_measurements | target_logit | target_val | other_logit | other_val | margin | aum    |
|-----------|------------------|--------------|------------|-------------|-----------|--------|--------|
| sample_1  | 1                | 0            | 3.74       | 10          | 2.48      | 1.26   | 1.26   |
| sample_1  | 2                | 0            | 4.59       | 10          | 3.44      | 1.15   | 1.205  |
| sample_2  | 1                | 1            | -0.09      | 0           | 3.11      | -3.20  | -3.02  |
| sample_2  | 2                | 1            | -1.12      | 0           | 3.25      | -4.37  | -3.785 |
| sample_3  | 1                | 6            | 3.39       | 10          | 1.62      | 1.77   | 1.77   |
| sample_3  | 2                | 6            | 2.63       | 2           | 2.11      | 0.52   | 1.145  |


## Replicate results from the paper
To replicate results, please refer to the [examples/paper_replication](examples/paper_replication) section.

## Example usage
For a more basic example of using the `AUMCalculator` and `DatasetWithIndex` in a training script, please refer to the [examples/cifar100](examples/cifar100) section.

## Cite
```sh
@article{pleiss2020identifying,
  title={Identifying Mislabeled Data using the Area Under the Margin Ranking},
  author={Geoff Pleiss and Tianyi Zhang and Ethan R. Elenberg and Kilian Q. Weinberger},
  journal={arXiv preprint arXiv:2001.10528},
  year={2020}
}
```