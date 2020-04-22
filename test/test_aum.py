import os
from dataclasses import asdict

import pandas as pd
import pytest
import torch

from aum import AUMCalculator, AUMRecord


@pytest.fixture(scope='module')
def aum_data():
    inputs = []
    outputs = []

    logits_1 = torch.tensor([[1., 2., 3.], [6., 5., 4.]])
    targets_1 = torch.tensor([1, 0])
    sample_ids_1 = ['a', 'b']
    inputs.append({'logits': logits_1, 'targets': targets_1, 'sample_ids': sample_ids_1})

    logits_2 = torch.tensor([[7., 8., 9.], [12., 11., 10.]])
    targets_2 = torch.tensor([2, 1])
    sample_ids_2 = ['b', 'c']
    inputs.append({'logits': logits_2, 'targets': targets_2, 'sample_ids': sample_ids_2})

    outputs.append({
        'a':
        AUMRecord(sample_id='a',
                  num_measurements=1,
                  target_logit=1,
                  target_val=2.,
                  other_logit=2,
                  other_val=3,
                  margin=-1.,
                  aum=-1.),
        'b':
        AUMRecord(sample_id='b',
                  num_measurements=1,
                  target_logit=0,
                  target_val=6.,
                  other_logit=1,
                  other_val=5.,
                  margin=1.,
                  aum=1.)
    })

    outputs.append({
        'b':
        AUMRecord(sample_id='b',
                  num_measurements=2,
                  target_logit=2,
                  target_val=9.,
                  other_logit=1,
                  other_val=8.,
                  margin=1.,
                  aum=1.),
        'c':
        AUMRecord(sample_id='c',
                  num_measurements=1,
                  target_logit=1,
                  target_val=11.,
                  other_logit=0,
                  other_val=12.,
                  margin=-1.,
                  aum=-1.)
    })

    return (inputs, outputs)


def test_aum_update(aum_data):
    inputs, outputs = aum_data
    aum_calculator = AUMCalculator(save_dir=None)

    expected_results = aum_calculator.update(inputs[0]['logits'], inputs[0]['targets'],
                                             inputs[0]['sample_ids'])
    assert expected_results == outputs[0]

    expected_results = aum_calculator.update(inputs[1]['logits'], inputs[1]['targets'],
                                             inputs[1]['sample_ids'])
    assert expected_results == outputs[1]


def test_aum_finalize(tmp_path, aum_data):
    inputs, outputs = aum_data
    save_dir = tmp_path.as_posix()
    aum_calculator = AUMCalculator(save_dir=save_dir, compressed=False)

    for data in inputs:
        aum_calculator.update(data['logits'], data['targets'], data['sample_ids'])

    aum_calculator.finalize()
    final_vals = pd.read_csv(os.path.join(save_dir, 'aum_values.csv'))
    detailed_vals = pd.read_csv(os.path.join(save_dir, 'full_aum_records.csv'))

    # Lets first verify detailed vals
    records = []
    for output in outputs:
        records.extend(output.values())

    expected_detailed_vals = pd.DataFrame([
        asdict(record) for record in records
    ]).sort_values(by=['sample_id', 'num_measurements']).reset_index(drop=True)
    assert detailed_vals.equals(expected_detailed_vals)

    # Now lets verfiy the final vals
    final_dict = {record.sample_id: record.aum for record in records}
    expected_final_vals = []
    for key, val in final_dict.items():
        expected_final_vals.append({'sample_id': key, 'aum': val})
    expected_final_vals = pd.DataFrame(expected_final_vals).sort_values(
        by='aum', ascending=False).reset_index(drop=True)

    assert final_vals.equals(expected_final_vals)
