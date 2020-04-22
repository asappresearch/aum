# CIFAR-100 Example
This is a simple example showing how to use the `AUMCalculator` and `DatasetWithIndex` in a training script. This script trains Resnet-34 on the CIFAR-100 dataset. At training completion, the aum artifacts will be located in the output directory. The samples with the lowest aum values are most likely mislabeled.

## Requirements
- pytorch >= 1.3
- torchvision >= 0.4
- numpy
- pandas
- aum
- tensorboard

## Usage
You can call the script as follows:

```sh
# for the compressed version of the AUMCalculator
python train.py

# For the uncompressed version of the AUMCalculator
python train.py --detailed-aum
```

The script will run without any specified arguments as all have defaults, but to see all available arguments:
```sh
python train.py --help
```
