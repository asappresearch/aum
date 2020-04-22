import math
import os
from collections import defaultdict, namedtuple
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import torch

sample_identifier = Union[int, str]


@dataclass
class AUMRecord:
    """
    Class for holding info around an aum update for a single sample
    """
    sample_id: sample_identifier
    num_measurements: int
    target_logit: int
    target_val: float
    other_logit: int
    other_val: float
    margin: float
    aum: float


class AUMCalculator():
    def __init__(self, save_dir: str, compressed: bool = True):
        """
        Intantiates the AUM object

        :param save_dir (str): Directory location of where to save out the final csv file(s)
            when calling `finalize`
        :param compressed (bool): Dictates how much information to store. If True, the object
            will only keep track of enough information to return the final AUM for each sample
            when `finalize` is called.
            If False, the object will keep track of the AUM value at each update call,
            storing an AUMRecord per sample per update call. This will also result in a
            `full_aum_records.csv` being saved out when calling `finalize`.
            Defaults to True.
        """
        self.save_dir = save_dir
        self.counts = defaultdict(int)
        self.sums = defaultdict(float)

        self.compressed = compressed
        if not compressed:
            self.records = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor,
               sample_ids: List[sample_identifier]) -> Dict[sample_identifier, AUMRecord]:
        """
        Updates the running totals and calculates the AUM values for the given samples

        :param logits (torch.Tensor): A 2 dimensional tensor where each row contains the logits
            for a given sample.
        :param targets (torch.Tensor): A 1 dimensional tensor containing the index of the target
            logit for a given sample.
        :param sample_ids (List[sample_identifier]): A list mapping each row of the logits & targets
            tensors to a sample id. This can be a list of ints or strings.

        :return (Dict[sample_identifier, AUMRecord]): A dictionary mapping each sample identifier
            to an AUMRecord. The AUMRecord contains the current AUM data for the given sample after
            this update step has been called.
        """

        target_values = logits.gather(1, targets.view(-1, 1)).squeeze()

        # mask out target values
        masked_logits = torch.scatter(logits, 1, targets.view(-1, 1), float('-inf'))
        other_logit_values, other_logit_index = masked_logits.max(1)
        other_logit_values = other_logit_values.squeeze()
        other_logit_index = other_logit_index.squeeze()

        margin_values = (target_values - other_logit_values).tolist()

        updated_aums = {}
        for i, (sample_id, margin) in enumerate(zip(sample_ids, margin_values)):
            self.counts[sample_id] += 1
            self.sums[sample_id] += margin

            record = AUMRecord(sample_id=sample_id,
                               num_measurements=self.counts[sample_id],
                               target_logit=targets[i].item(),
                               target_val=target_values[i].item(),
                               other_logit=other_logit_index[i].item(),
                               other_val=other_logit_values[i].item(),
                               margin=margin,
                               aum=self.sums[sample_id] / self.counts[sample_id])

            updated_aums[sample_id] = record
            if not self.compressed:
                self.records.append(record)

        return updated_aums

    def finalize(self, save_dir: Optional[str] = None) -> None:
        """
        Calculates AUM for each sample given the data gathered on each update call.
        Outputs a `aum_values.csv` file containing the final AUM values for each sample.
        If `self.compressed` set to False, this will also output a `full_aum_records.csv` file
        containing AUM values for each sample at each update call.

        :param save_dir (Optional[str]): Allows the ability to overwrite the original save
            directory that was set on instantiation of the AUM object. When set to None, the
            directory set on instantiation will be used. Defaults to None.
        """
        save_dir = save_dir or self.save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        results = [{
            'sample_id': sample_id,
            'aum': self.sums[sample_id] / self.counts[sample_id]
        } for sample_id in self.counts.keys()]

        result_df = pd.DataFrame(results).sort_values(by='aum', ascending=False)

        save_path = os.path.join(save_dir, 'aum_values.csv')
        result_df.to_csv(save_path, index=False)

        if not self.compressed:
            records_df = AUMCalculator.records_to_df(self.records)
            save_path = os.path.join(save_dir, 'full_aum_records.csv')
            records_df.to_csv(save_path, index=False)

    @staticmethod
    def records_to_df(records: List[AUMRecord]) -> pd.DataFrame:
        """
        Converts a list of AUMRecords to a dataframe, sorted by sample_id & num_measurements

        :param records (List[AUMRecord]): A list of AUMRecords

        :return (pd.DataFrame): a dataframe, sorted by sample_id & num_measurements
        """
        df = pd.DataFrame([asdict(record) for record in records])
        df.sort_values(by=['sample_id', 'num_measurements'], inplace=True)
        return df
