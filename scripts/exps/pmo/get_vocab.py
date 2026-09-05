# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from tqdm import trange
import numpy as np
import pandas as pd
from tdc import Oracle
from genmol.utils.utils_chem import cut
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='data/zinc250k.csv')
    parser.add_argument('--output-dir', default='scripts/exps/pmo/vocab')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-fragments', type=int, default=10000)
    args = parser.parse_args()
    fragmentation_rng = random.Random(args.seed)

    props = ['albuterol_similarity',
             'amlodipine_mpo',
             'celecoxib_rediscovery',
             'deco_hop',
             'drd2',
             'fexofenadine_mpo',
             'gsk3b',
             'isomers_c7h8n2o2',
             'isomers_c9h10n2o2pf2cl',
             'jnk3',
             'median1',
             'median2',
             'mestranol_similarity',
             'osimertinib_mpo',
             'perindopril_mpo',
             'qed',
             'ranolazine_mpo',
             'scaffold_hop',
             'sitagliptin_mpo',
             'thiothixene_rediscovery',
             'troglitazone_rediscovery',
             'valsartan_smarts',
             'zaleplon_mpo']
    
    df = pd.read_csv(args.data_path)
    if df.empty:
        raise ValueError('input dataset is empty')
    if 'smiles' not in df:
        raise ValueError("input dataset must contain a 'smiles' column")

    # calculate properties
    for prop in props:
        if prop not in df:
            print(f'Calculating {prop}...')
            df[prop] = Oracle(prop)(df['smiles'].tolist())
            df.to_csv(args.data_path, index=False)
        if not np.isfinite(pd.to_numeric(df[prop], errors='coerce')).all():
            raise ValueError(f'Oracle column {prop!r} contains non-finite values')
    
    # construct vocabulary
    avg_num_frags = 0
    frag2cnt = defaultdict(int)
    frag2score = {prop: defaultdict(float) for prop in props}
    for i in trange(len(df)):
        frags = cut(df['smiles'].iloc[i], rng=fragmentation_rng)
        avg_num_frags += len(frags)
        for frag in frags:
            frag2cnt[frag] += 1
            for prop in props:
                frag2score[prop][frag] += df[prop].iloc[i]
    print(f'Average # of fragments: {avg_num_frags / len(df):.2f}')
    
    os.makedirs(args.output_dir, exist_ok=True)

    metadata = {
        'data_path': os.path.abspath(args.data_path),
        'data_sha256': sha256_file(args.data_path),
        'molecule_count': len(df),
        'fragmentation': 'three sampled single non-ring-bond cuts with replacement',
        'seed': args.seed,
        'max_fragments': args.max_fragments,
        'average_fragments_per_molecule': avg_num_frags / len(df),
        'oracle_global_means': {
            prop: float(df[prop].mean()) for prop in props
        },
    }

    for prop in props:
        rows = [
            {
                'frag': frag,
                'score': score_sum / frag2cnt[frag],
                'count': frag2cnt[frag],
                'score_sum': score_sum,
            }
            for frag, score_sum in frag2score[prop].items()
        ]
        vocab = pd.DataFrame(rows)
        vocab = vocab.sort_values(by=['score', 'frag'], ascending=[False, True])
        vocab = vocab.iloc[:args.max_fragments]
        vocab['size'] = vocab['frag'].apply(
            lambda frag: Chem.MolFromSmiles(frag).GetNumAtoms()
        )
        vocab.to_csv(os.path.join(args.output_dir, f'{prop}.csv'), index=False)

    with open(os.path.join(args.output_dir, 'vocab_metadata.json'), 'w') as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
