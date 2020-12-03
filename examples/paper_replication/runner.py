import datetime
import logging
import os
import random
import shutil
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch

import fire
import tqdm
import util
from aum import AUMCalculator
from losses import losses
from models import models
from torchvision import datasets
from torchvision import models as tvmodels
from torchvision import transforms


class _Dataset(torch.utils.data.Dataset):
    """
    A wrapper around existing torch datasets to add purposefully mislabeled samplesa and threshold samples.

    :param :obj:`torch.utils.data.Dataset` base_dataset: Dataset to wrap
    :param :obj:`torch.LongTensor` indices: List of indices of base_dataset to include (used to create valid. sets)
    :param dict flip_dict: (optional) List mapping sample indices to their (incorrect) assigned label
    :param bool use_threshold_samples: (default False) Whether or not to add threshold samples to this datasets
    :param bool threshold_samples_set_idx: (default 1) Which set of threshold samples to use.
    """
    def __init__(self,
                 base_dataset,
                 indices=None,
                 flip_dict=None,
                 use_threshold_samples=False,
                 threshold_samples_set_idx=1):
        super().__init__()
        self.dataset = base_dataset
        self.flip_dict = flip_dict or {}
        self.indices = torch.arange(len(self.dataset)) if indices is None else indices

        # Create optional extra class (for threshold samples)
        self.use_threshold_samples = use_threshold_samples
        if use_threshold_samples:
            num_threshold_samples = len(self.indices) // (self.targets.max().item() + 1)
            start_index = (threshold_samples_set_idx - 1) * num_threshold_samples
            end_index = (threshold_samples_set_idx) * num_threshold_samples
            self.threshold_sample_indices = torch.randperm(len(self.indices))[start_index:end_index]

    @property
    def targets(self):
        """
        (Hidden) ground-truth labels
        """
        if not hasattr(self, "_target_memo"):
            try:
                self.__target_memo = torch.tensor(self.dataset.targets)[self.indices]
            except Exception:
                self.__target_memo = torch.tensor([target
                                                   for _, target in self.dataset])[self.indices]
        if torch.is_tensor(self.__target_memo):
            return self.__target_memo
        else:
            return torch.tensor(self.__target_memo)

    @property
    def assigned_targets(self):
        """
        (Potentially incorrect) assigned labels
        """
        if not hasattr(self, "_assigned_target_memo"):
            self._assigned_target_memo = self.targets.clone()

            # Change labels of mislabeled samples
            if self.flip_dict is not None:
                for i, idx in enumerate(self.indices.tolist()):
                    if idx in self.flip_dict.keys():
                        self._assigned_target_memo[i] = self.flip_dict[idx]

            # Change labels of threshold samples
            if self.use_threshold_samples:
                extra_class = (self.targets.max().item() + 1)
                self._assigned_target_memo[self.threshold_sample_indices] = extra_class
        return self._assigned_target_memo

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        input, _ = self.dataset[self.indices[index].item()]
        target = self.assigned_targets[index].item()
        res = input, target, index
        return res


class Runner(object):
    """
    Main module for running experiments. Can call `load`, `save`, `train`, `test`, etc.

    :param str data: Directory to load data from
    :param str save: Directory to save model/results
    :param str dataset: (cifar10, cifar100, tiny_imagenet, webvision50, clothing100k)

    :param int num_valid: (default 5000) What size validation set to use (comes from train set, indices determined by seed)
    :param int seed: (default 0) Random seed
    :param int split_seed: (default 0) Which random seed to use for creating trian/val split and for flipping random labels.
        If this arg is not supplied, the split_seed will come from the `seed` arg.

    :param float perc_mislabeled: (default 0.) How many samples will be intentionally mislabeled.
        Default is 0. - i.e. regular training without flipping any labels.
    :param str noise_type: (uniform, flip) Mislabeling noise model to use.

    :param bool use_threshold_samples: (default False) Whether to add indictaor samples
    :param bool threshold_samples_set_idx: (default 1) Which set of threshold samples to use (based on index)

    :param str loss_type: (default cross-entropy) Loss type
    :param bool oracle_training: (default False) If true, the network will be trained only on clean data
        (i.e. all training points with flipped labels will be discarded).

    :param str net_type: (resnet, densenet, wide_resnet) Which network to use.
    :param **model_args: Additional argumets to pass to the model
    """
    def __init__(self,
                 data,
                 save,
                 dataset="cifar10",
                 num_valid=5000,
                 seed=0,
                 split_seed=None,
                 noise_type="uniform",
                 perc_mislabeled=0.,
                 use_threshold_samples=False,
                 threshold_samples_set_idx=1,
                 loss_type="cross-entropy",
                 oracle_training=False,
                 net_type="resnet",
                 pretrained=False,
                 **model_args):
        if not os.path.exists(save):
            os.makedirs(save)
        if not os.path.isdir(save):
            raise Exception('%s is not a dir' % save)
        self.data = data
        self.savedir = save
        self.perc_mislabeled = perc_mislabeled
        self.noise_type = noise_type
        self.dataset = dataset
        self.net_type = net_type
        self.num_valid = num_valid
        self.use_threshold_samples = use_threshold_samples
        self.threshold_samples_set_idx = threshold_samples_set_idx
        self.split_seed = split_seed if split_seed is not None else seed
        self.seed = seed
        self.loss_func = losses[loss_type]
        self.oracle_training = oracle_training
        self.pretrained = pretrained

        # Seed
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        random.seed(0)

        # Logging
        self.timestring = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        logging.basicConfig(
            format='%(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(os.path.join(self.savedir, 'log-%s.log' % self.timestring)),
            ],
            level=logging.INFO,
        )
        logging.info('Data dir:\t%s' % data)
        logging.info('Save dir:\t%s\n' % save)

        # Make model
        self.num_classes = self.test_set.targets.max().item() + 1
        if use_threshold_samples:
            self.num_classes += 1
        self.num_data = len(self.train_set)
        logging.info(f"\nDataset: {self.dataset}")
        logging.info(f"Num train: {self.num_data}")
        logging.info(f"Num valid: {self.num_valid}")
        logging.info(f"Extra class: {self.use_threshold_samples}")
        logging.info(f"Num classes: {self.num_classes}")
        if self.perc_mislabeled:
            logging.info(f"Noise type: {self.noise_type}")
            logging.info(f"Flip perc: {self.perc_mislabeled}\n")
            if self.oracle_training:
                logging.info(f"Training with Oracle Only")

        # Model
        if self.dataset == "imagenet" or "webvision" in self.dataset or "clothing" in self.dataset:
            big_models = dict((key, val) for key, val in tvmodels.__dict__.items())
            self.model = big_models[self.net_type](pretrained=False, num_classes=self.num_classes)
            if self.pretrained:
                try:
                    self.model.load_state_dict(
                        big_models[self.net_type](pretrained=True).state_dict(), strict=False)
                except RuntimeError:
                    pass
            # Fix pooling issues
            if "inception" in self.net_type:
                self.avgpool_1a = torch.nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.model = models[self.net_type](
                num_classes=self.num_classes,
                initial_stride=(2 if "tiny" in self.dataset.lower() else 1),
                **model_args)
        logging.info(f"Model type: {self.net_type}")
        logging.info(f"Model args:")
        for key, val in model_args.items():
            logging.info(f" - {key}: {val}")
        logging.info(f"Loss type: {loss_type}")
        logging.info("")

    def _make_datasets(self):
        try:
            dataset_cls = getattr(datasets, self.dataset.upper())
            self.big_model = False
        except Exception:
            dataset_cls = datasets.ImageFolder
            if "tiny" in self.dataset.lower():
                self.big_model = False
            else:
                self.big_model = True

        # Get constants
        if dataset_cls == datasets.ImageFolder:
            tmp_set = dataset_cls(root=os.path.join(self.data, "train"))
        else:
            tmp_set = dataset_cls(root=self.data, train=True, download=True)
            if self.dataset.upper() == 'CIFAR10':
                tmp_set.targets = tmp_set.train_labels
        num_train = len(tmp_set) - self.num_valid
        num_valid = self.num_valid
        num_classes = int(max(tmp_set.targets)) + 1

        # Create train/valid split
        torch.manual_seed(self.split_seed)
        torch.cuda.manual_seed_all(self.split_seed)
        random.seed(self.split_seed)
        train_indices, valid_indices = torch.randperm(num_train + num_valid).split(
            [num_train, num_valid])

        # dataset indices flip
        flip_dict = {}
        if self.perc_mislabeled:
            # Generate noisy labels from random transitions
            transition_matrix = torch.eye(num_classes)
            if self.noise_type == "uniform":
                transition_matrix.mul_(1 - self.perc_mislabeled * (num_classes / (num_classes - 1)))
                transition_matrix.add_(self.perc_mislabeled / (num_classes - 1))
            elif self.noise_type == "flip":
                source_classes = torch.arange(num_classes)
                target_classes = (source_classes + 1).fmod(num_classes)
                transition_matrix.mul_(1 - self.perc_mislabeled)
                transition_matrix[source_classes, target_classes] = self.perc_mislabeled
            else:
                raise ValueError(f"Unknonwn noise type {self.noise}")
            true_targets = (torch.tensor(tmp_set.targets) if hasattr(tmp_set, "targets") else
                            torch.tensor([target for _, target in self]))
            transition_targets = torch.distributions.Categorical(
                probs=transition_matrix[true_targets, :]).sample()
            # Create a dictionary of transitions
            if not self.oracle_training:
                flip_indices = torch.nonzero(transition_targets != true_targets).squeeze(-1)
                flip_targets = transition_targets[flip_indices]
                for index, target in zip(flip_indices, flip_targets):
                    flip_dict[index.item()] = target.item()
            else:
                # In the oracle setting, don't add transitions
                oracle_indices = torch.nonzero(transition_targets == true_targets).squeeze(-1)
                train_indices = torch.from_numpy(
                    np.intersect1d(oracle_indices.numpy(), train_indices.numpy())).long()

        # Reset the seed for dataset/initializations
        torch.manual_seed(self.split_seed)
        torch.cuda.manual_seed_all(self.split_seed)
        random.seed(self.split_seed)

        # Define trainsforms
        if self.big_model:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            test_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(227 if "inception" in self.net_type else 224),
                transforms.ToTensor(),
                normalize,
            ])
            train_transforms = transforms.Compose([
                transforms.RandomResizedCrop(227 if "inception" in self.net_type else 224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif self.dataset == "tiny_imagenet":
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            test_transforms = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
            train_transforms = transforms.Compose([
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
                test_transforms,
            ])
        elif self.dataset == "cifar10":
            normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                             std=[0.2471, 0.2435, 0.2616])
            test_transforms = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
            train_transforms = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                test_transforms,
            ])
        elif self.dataset == "cifar100":
            normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                             std=[0.2471, 0.2435, 0.2616])
            test_transforms = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
            train_transforms = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                test_transforms,
            ])
        elif self.dataset == "mnist":
            normalize = transforms.Normalize(mean=(0.1307, ), std=(0.3081, ))
            test_transforms = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
            train_transforms = test_transforms
        else:
            raise ValueError(f"Unknown dataset {self.dataset}")

        # Get train set
        if dataset_cls == datasets.ImageFolder:
            self._train_set_memo = _Dataset(
                dataset_cls(
                    root=os.path.join(self.data, "train"),
                    transform=train_transforms,
                ),
                flip_dict=flip_dict,
                indices=train_indices,
                use_threshold_samples=self.use_threshold_samples,
                threshold_samples_set_idx=self.threshold_samples_set_idx,
            )
            if os.path.exists(os.path.join(self.data, "test")):
                self._valid_set_memo = _Dataset(
                    dataset_cls(root=os.path.join(self.data, "val"), transform=test_transforms))
                self._test_set_memo = _Dataset(
                    dataset_cls(root=os.path.join(self.data, "test"), transform=test_transforms))
            else:
                self._valid_set_memo = _Dataset(
                    dataset_cls(root=os.path.join(self.data, "train"), transform=test_transforms),
                    indices=valid_indices,
                ) if len(valid_indices) else None
                self._test_set_memo = _Dataset(
                    dataset_cls(root=os.path.join(self.data, "val"), transform=test_transforms))
        else:
            self._train_set_memo = _Dataset(
                dataset_cls(root=self.data, train=True, transform=train_transforms),
                flip_dict=flip_dict,
                indices=train_indices,
                use_threshold_samples=self.use_threshold_samples,
                threshold_samples_set_idx=self.threshold_samples_set_idx,
            )
            self._valid_set_memo = _Dataset(dataset_cls(
                root=self.data, train=True, transform=test_transforms),
                                            indices=valid_indices) if len(valid_indices) else None
            self._test_set_memo = _Dataset(
                dataset_cls(root=self.data, train=False, transform=test_transforms))

    @property
    def test_set(self):
        if not hasattr(self, "_test_set_memo"):
            self._make_datasets()
        return self._test_set_memo

    @property
    def train_set(self):
        if not hasattr(self, "_train_set_memo"):
            self._make_datasets()
        return self._train_set_memo

    @property
    def valid_set(self):
        if not hasattr(self, "_valid_set_memo"):
            self._make_datasets()
        return self._valid_set_memo

    def generate_aum_details(self, load=None):
        """
        Script for accumulating both aum values and other sample details at the end of training.
        It makes a dataframe that contains AUMs Clean for all samples
        The results are saved to the file `aum_details.csv` in the model folder.

        :param str load: (optional) If set to some value - it will assemble aum info from the model stored in the `load` folder.
            Otherwise - it will comptue aums from the runner's model.

        :return: self
        """

        load = load or self.savedir
        train_data = torch.load(os.path.join(load, "train_data.pth"))
        aum_data = pd.read_csv(os.path.join(load, "aum_values.csv"))

        # HACK: fix for old version of the code
        if "assigned_targets" not in train_data:
            train_data["assigned_targets"] = train_data["observed_targets"]

        true_targets = train_data["true_targets"]
        assigned_targets = train_data["assigned_targets"]
        is_threshold_sample = assigned_targets.gt(true_targets.max())
        label_flipped = torch.ne(true_targets, assigned_targets)

        # Where to store result
        result = {}

        # Add index of samples
        result["Index"] = torch.arange(train_data["assigned_targets"].size(-1))

        # Add label flipped info
        result["True Target"] = true_targets
        result["Observed Target"] = assigned_targets
        result["Label Flipped"] = label_flipped
        result["Is Threshold Sample"] = is_threshold_sample

        # Add AUM
        aum_data = aum_data.set_index('sample_id')
        aum_data = aum_data.reindex(list(range(train_data["assigned_targets"].size(-1))))
        aum_list = aum_data['aum'].to_list()
        result["AUM"] = torch.tensor(aum_list)

        # Add AUM "worse than random" (AUM_WTR) score
        # i.e. - is the AUM worse than 99% of threshold samples?
        if is_threshold_sample.sum().item():
            aum_wtr = torch.lt(
                result["AUM"].view(-1, 1),
                result["AUM"][is_threshold_sample].view(1, -1),
            ).float().mean(dim=-1).gt(0.01).float()
            result["AUM_WTR"] = aum_wtr
        else:
            result["AUM_WTR"] = torch.ones_like(result["AUM"])

        df = pd.DataFrame(result)
        df.set_index(
            ["Index", "True Target", "Observed Target", "Label Flipped", "Is Threshold Sample"],
            inplace=True)
        df.to_csv(os.path.join(load, "aum_details.csv"))
        return self

    def done(self):
        "Break out of the runner"
        return None

    def load(self, save=None, suffix=""):
        """
        Load a previously saved model state dict.

        :param str save: (optional) Which folder to load the saved model from.
            Will default to the current runner's save dir.
        :param str suffix: (optional) Which model file to load (e.g. "model.pth.last").
            By default will load "model.pth" which contains the early-stopped model.
        """
        save = save or self.savedir
        state_dict = torch.load(os.path.join(save, f"model.pth{suffix}"),
                                map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict, strict=False)
        return self

    def save(self, save=None, suffix=""):
        """
        Save the current state dict

        :param str save: (optional) Which folder to save the model to.
            Will default to the current runner's save dir.
        :param str suffix: (optional) A suffix to append to the save name.
        """
        save = save or self.savedir
        torch.save(self.model.state_dict(), os.path.join(save, f"model.pth{suffix}"))
        return self

    def subset(self, perc, aum_files=None):
        """
        Use only a subset of the training set
        If aum files are supplied, then drop samples with the lowest aum.
        Otherwise, drop samples at random.

        :param float perc: What percentage of the set to use
        :param str aum_files:
        """
        if aum_files is None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            random.seed(self.seed)
            order = torch.randperm(len(self.train_set))
        else:
            counts = torch.zeros(len(self.train_set))
            aums = torch.zeros(len(self.train_set))
            if isinstance(aum_files, str):
                aum_files = aum_files.split(",")
            for sub_aum_file in aum_files:
                aums_path = os.path.join(sub_aum_file, "aum_details.csv")
                if not os.path.exists(aums_path):
                    self.compute_aums(load=sub_aum_file)
                aums_data = pd.read_csv(aums_path).drop(
                    ["True Target", "Observed Target", "Label Flipped"], axis=1)
                counts += torch.tensor(~aums_data["Is Threshold Sample"].values).float()
                aums += torch.tensor(aums_data["AUM"].values *
                                     ~aums_data["Is Threshold Sample"].values).float()
            counts.clamp_min_(1)
            aums = aums.div_(counts)
            order = aums.argsort(descending=True)

        num_samples = int(len(self.train_set) * perc)
        self.train_set.indices = self.train_set.indices[order[:num_samples]]
        logging.info(f"Reducing training set from {len(order)} to {len(self.train_set)}")
        if aum_files is not None:
            logging.info(
                f"Average AUM: {aums[order[:num_samples]].mean().item()} (from {aums.mean().item()}"
            )
        return self

    def test(self,
             model=None,
             split="test",
             batch_size=512,
             dataset=None,
             epoch=None,
             num_workers=0):
        """
        Testing script
        """
        stats = ['error', 'top5_error', 'loss']
        meters = [util.AverageMeter() for _ in stats]
        result_class = util.result_class(stats)

        # Get model
        if model is None:
            model = self.model
            # Model on cuda
            if torch.cuda.is_available():
                model = model.cuda()
                if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                    model = torch.nn.DataParallel(model).cuda()

        # Get dataset/loader
        if dataset is None:
            try:
                dataset = getattr(self, f"{split}_set")
            except Exception:
                raise ValueError(f"Invalid split '{split}'")
        loader = tqdm.tqdm(torch.utils.data.DataLoader(dataset,
                                                       batch_size=batch_size,
                                                       shuffle=False,
                                                       num_workers=num_workers),
                           desc=split.title())

        # For storing results
        all_losses = []
        all_confs = []
        all_preds = []
        all_targets = []

        # Model on train mode
        model.eval()
        with torch.no_grad():
            for inputs, targets, indices in loader:
                # Get types right
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                # Calculate loss
                outputs = model(inputs)
                losses = self.loss_func(outputs, targets, reduction="none")
                confs, preds = outputs.topk(5, dim=-1, largest=True, sorted=True)
                is_correct = preds.eq(targets.unsqueeze(-1)).float()
                loss = losses.mean()
                error = 1 - is_correct[:, 0].mean()
                top5_error = 1 - is_correct.sum(dim=-1).mean()

                # measure and record stats
                batch_size = inputs.size(0)
                stat_vals = [error.item(), top5_error.item(), loss.item()]
                for stat_val, meter in zip(stat_vals, meters):
                    meter.update(stat_val, batch_size)

                # Record losses
                all_losses.append(losses.cpu())
                all_confs.append(confs[:, 0].cpu())
                all_preds.append(preds[:, 0].cpu())
                all_targets.append(targets.cpu())

                # log stats
                res = dict((name, f"{meter.val:.3f} ({meter.avg:.3f})")
                           for name, meter in zip(stats, meters))
                loader.set_postfix(**res)

        # Save the outputs
        pd.DataFrame({
            "Loss": torch.cat(all_losses).numpy(),
            "Prediction": torch.cat(all_preds).numpy(),
            "Confidence": torch.cat(all_confs).numpy(),
            "Label": torch.cat(all_targets).numpy(),
        }).to_csv(os.path.join(self.savedir, f"results_{split}.csv"), index_label="index")

        # Return summary statistics and outputs
        return result_class(*[meter.avg for meter in meters])

    def train_for_aum_computation(self,
                                  num_epochs=150,
                                  batch_size=64,
                                  lr=0.1,
                                  wd=1e-4,
                                  momentum=0.9,
                                  **kwargs):
        """
        Helper training script - this trains models that will be specifically used for AUL computations

        :param int num_epochs: (default 150) (This corresponds roughly to how
            many epochs a normal model is trained for before the lr drop.)
        :param int batch_size: (default 64) (The batch size is intentionally
            lower - this makes the network less likely to memorize.)
        :param float lr: Learning rate
        :param float wd: Weight decay
        :param float momentum: Momentum
        """
        return self.train(num_epochs=num_epochs,
                          batch_size=batch_size,
                          test_at_end=False,
                          lr=lr,
                          wd=wd,
                          momentum=momentum,
                          lr_drops=[],
                          **kwargs)

    def train(self,
              num_epochs=300,
              batch_size=256,
              test_at_end=True,
              lr=0.1,
              wd=1e-4,
              momentum=0.9,
              lr_drops=[0.5, 0.75],
              aum_wtr=False,
              rand_weight=False,
              **kwargs):
        """
        Training script

        :param int num_epochs: (default 300)
        :param int batch_size: (default 256)
        :param float lr: Learning rate
        :param float wd: Weight decay
        :param float momentum: Momentum
        :param list lr_drops: When to drop the learning rate (by a factor of 10) as a percentage of total training time.

        :param str aum_wtr: (optional) The path of the model/results directory to load AUM_WTR weights from.
        :param bool rand_weight (optional, default false): uses rectified normal random weighting if True.
        """
        # Model
        model = self.model
        if torch.cuda.is_available():
            model = model.cuda()
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model).cuda()

        # Optimizer
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=lr,
                                    weight_decay=wd,
                                    momentum=momentum,
                                    nesterov=True)
        milestones = [int(lr_drop * num_epochs) for lr_drop in (lr_drops or [])]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=milestones,
                                                         gamma=0.1)
        logging.info(f"\nOPTIMIZER:\n{optimizer}")
        logging.info(f"SCHEDULER:\n{scheduler.milestones}")

        # Initialize AUM caluclator object
        aum_calculator = AUMCalculator(save_dir=self.savedir, compressed=False)

        train_data = OrderedDict()
        train_data["train_indices"] = self.train_set.indices
        train_data["valid_indices"] = (self.valid_set.indices if self.valid_set is not None else
                                       torch.tensor([], dtype=torch.long))
        train_data["true_targets"] = self.train_set.targets
        train_data["assigned_targets"] = self.train_set.assigned_targets

        # Storage to log results
        results = []

        # Train model
        best_error = 1
        for epoch in range(num_epochs):
            train_results = self.train_epoch(model=model,
                                             optimizer=optimizer,
                                             epoch=epoch,
                                             num_epochs=num_epochs,
                                             batch_size=batch_size,
                                             aum_calculator=aum_calculator,
                                             aum_wtr=aum_wtr,
                                             rand_weight=rand_weight,
                                             **kwargs)
            if self.valid_set is not None:
                valid_results = self.test(model=model,
                                          split="valid",
                                          batch_size=batch_size,
                                          epoch=epoch,
                                          **kwargs)
            else:
                valid_results = self.test(model,
                                          split="test",
                                          batch_size=batch_size,
                                          epoch=epoch,
                                          **kwargs)
            scheduler.step()

            # Determine if model is the best
            if self.valid_set is not None:
                self.save()
            elif best_error > valid_results.error:
                best_error = valid_results.error
                logging.info('New best error: %.4f' % valid_results.error)
                self.save()

            # Log results
            logging.info(f"\nTraining {repr(train_results)}")
            logging.info(f"\nValidation {repr(valid_results)}")
            logging.info('')
            results.append(
                OrderedDict([("epoch", f"{epoch + 1:03d}"),
                             *[(f"train_{field}", val) for field, val in train_results.items()],
                             *[(f"valid_{field}", val) for field, val in valid_results.items()]]))
            pd.DataFrame(results).set_index("epoch").to_csv(
                os.path.join(self.savedir, "train_log.csv"))

            # Save metadata around train set (like which labels were flipped)
            torch.save(train_data, os.path.join(self.savedir, "train_data.pth"))

        # Once we're finished training calculate aum
        aum_calculator.finalize()

        # Maybe test (last epoch)
        if test_at_end and self.valid_set is not None:
            test_results = self.test(model=model, **kwargs)
            logging.info(f"\nTest (no early stopping) {repr(test_results)}")
            shutil.copyfile(os.path.join(self.savedir, "results_test.csv"),
                            os.path.join(self.savedir, "results_test_noearlystop.csv"))
            results.append(
                OrderedDict([(f"test_{field}", val) for field, val in test_results.items()]))
            pd.DataFrame(results).set_index("epoch").to_csv(
                os.path.join(self.savedir, "train_log.csv"))

        # Load best model
        self.save(suffix=".last")
        self.load()

        # Maybe test (best epoch)
        if test_at_end and self.valid_set is not None:
            test_results = self.test(model=model, **kwargs)
            logging.info(f"\nEarly Stopped Model Test {repr(test_results)}")
            results.append(
                OrderedDict([(f"test_best_{field}", val) for field, val in test_results.items()]))
        pd.DataFrame(results).set_index("epoch").to_csv(os.path.join(self.savedir, "train_log.csv"))

        return self

    def train_epoch(self,
                    model,
                    optimizer,
                    epoch,
                    num_epochs,
                    batch_size=256,
                    num_workers=0,
                    aum_calculator=None,
                    aum_wtr=False,
                    rand_weight=False):
        stats = ["error", "loss"]
        meters = [util.AverageMeter() for _ in stats]
        result_class = util.result_class(stats)

        # Weighting - set up from GMM
        # NOTE: This is only used when removing threshold samples
        # TODO: some of this probably needs to be changed?
        if aum_wtr:
            counts = torch.zeros(len(self.train_set))
            bad_probs = torch.zeros(len(self.train_set))
            if isinstance(aum_wtr, str):
                aum_wtr = aum_wtr.split(",")
            for sub_aum_wtr in aum_wtr:
                aums_path = os.path.join(sub_aum_wtr, "aum_details.csv")
                if not os.path.exists(aums_path):
                    self.generate_aum_details(load=sub_aum_wtr)
                aums_data = pd.read_csv(aums_path).drop(
                    ["True Target", "Observed Target", "Label Flipped"], axis=1)
                counts += torch.tensor(~aums_data["Is Threshold Sample"].values).float()
                bad_probs += torch.tensor(aums_data["AUM_WTR"].values *
                                          ~aums_data["Is Threshold Sample"].values).float()
            counts.clamp_min_(1)
            good_probs = (1 - bad_probs / counts).to(next(model.parameters()).dtype).ceil()
            if torch.cuda.is_available():
                good_probs = good_probs.cuda()
            logging.info(f"AUM WTR Score")
            logging.info(f"(Num samples removed: {good_probs.ne(1.).sum().item()})")
        elif rand_weight:
            logging.info("Rectified Normal Random Weighting")
        else:
            logging.info("Standard weighting")

        # Setup loader
        train_set = self.train_set
        loader = tqdm.tqdm(torch.utils.data.DataLoader(train_set,
                                                       batch_size=batch_size,
                                                       shuffle=True,
                                                       num_workers=num_workers),
                           desc=f"Train (Epoch {epoch + 1}/{num_epochs})")

        # Model on train mode
        model.train()
        for inputs, targets, indices in loader:
            optimizer.zero_grad()

            # Get types right
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()

            # Compute output and losses
            outputs = model(inputs)
            losses = self.loss_func(outputs, targets, reduction="none")
            preds = outputs.argmax(dim=-1)

            # Compute loss weights
            if aum_wtr:
                weights = good_probs[indices.to(good_probs.device)]
                weights = weights.div(weights.sum())
            elif rand_weight:
                weights = torch.randn(targets.size(), dtype=outputs.dtype,
                                      device=outputs.device).clamp_min_(0)
                weights = weights.div(weights.sum().clamp_min_(1e-10))
            else:
                weights = torch.ones(targets.size(), dtype=outputs.dtype,
                                     device=outputs.device).div_(targets.numel())

            # Backward through model
            loss = torch.dot(weights, losses)
            error = torch.ne(targets, preds).float().mean()
            loss.backward()

            # Update the model
            optimizer.step()

            # Update AUM values (after the first epoch due to variability of random initialization)
            if aum_calculator and epoch > 0:
                aum_calculator.update(logits=outputs.detach().cpu().half().float(),
                                      targets=targets.detach().cpu(),
                                      sample_ids=indices.tolist())

            # measure and record stats
            batch_size = outputs.size(0)
            stat_vals = [error.item(), loss.item()]
            for stat_val, meter in zip(stat_vals, meters):
                meter.update(stat_val, batch_size)

            # log stats
            res = dict(
                (name, f"{meter.val:.3f} ({meter.avg:.3f})") for name, meter in zip(stats, meters))
            loader.set_postfix(**res)

        # Return summary statistics
        return result_class(*[meter.avg for meter in meters])


if __name__ == "__main__":
    fire.Fire(Runner)
