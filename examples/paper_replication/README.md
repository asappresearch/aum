# Paper Replication

## Requirements
- pytorch >=1.3
- torchvision >= 0.4
- numpy
- pandas
- tqdm
- aum
- fire (pip install fire)

## Datasets

We run experiments on 3 **small** datasets...
- cifar10
- cifar100
- tiny_imagenet

... and 3 **large** datasets
- webvision50
- clothing100k

Download and untar the file here for all 5 datasets:
https://drive.google.com/file/d/1rr2nvnnBMsbo1qcU3i3urJsDw86PJ9tR/view?usp=sharing

Alternatively, if you just want to run CIFAR10 and CIFAR100 you don't need to download anything

## Run the baseline models
These scripts produce the "Baseline" result in our tables.

```sh
# For `dataset=cifar10`, `dataset=cifar100`, or `dataset=tiny_imagenet`
./small_dataset_baseline.sh <path_to_data_dir> <dataset> <seed> <amount_of_noise> <noise_type>
# For `dataset=webvision50`, `dataset=clothing100k`
./large_dataset_baseline.sh <path_to_data_dir> <dataset> <seed>
```

The arguments:
- `<seed>` - set to something like `1`
- `<amount_of_noise>` - percentage of synthetic label noise to add (e.g. `0.2`). Set to `0` for no synthetic mislabeled data.
- `<noise_type>` - either `uniform` or `flip`.

Note that `./large_dataset_baseline` does not take an `<amount_of_noise>` or `<noise_type>` argument.

## Run the AUM models
These scripts produce the "AUM" result in our tables.

```sh
# For `dataset=cifar10`, `dataset=cifar100`, or `dataset=tiny_imagenet`
./small_dataset_aum.sh <path_to_data_dir> <dataset> <seed> <amount_of_noise> <noise_type>
# For `dataset=webvision50`, `dataset=clothing100k`
./large_dataset_aum.sh <path_to_data_dir> <dataset> <seed>
```
