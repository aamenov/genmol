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


import hashlib
import os
import torch
import datasets
from datasets.distributed import split_dataset_by_node
from huggingface_hub import hf_hub_download
from safe.tokenizer import SAFETokenizer
from rdkit import RDLogger
from genmol.utils.bracket_safe_converter import safe2bracketsafe
RDLogger.DisableLog('rdApp.*')


ROOT_DIR = os.getcwd()
SAFE_GPT_REPO_ID = 'datamol-io/safe-gpt'
SAFE_GPT_TOKENIZER_REVISION = '3d5fa0988383e898d5ac5db7cd52bf715bc37061'
SAFE_GPT_TOKENIZER_SHA256 = (
    '0db5f4dbdc7e8ff759e98483759611a426e187ee7f3f0a91edc8800abe7bf140'
)
SAFE_GPT_DATASET_REVISION = 'b83175cd7394e7a4027478a35b2f9d1dda3ac62f'


def get_last_checkpoint(save_dir):
    if os.path.exists(save_dir):
        filenames = os.listdir(save_dir)
        if filenames:
            last_filename = sorted(filenames, key=lambda x: int(x[:-5]))[-1]
            return os.path.join(save_dir, last_filename)
    

def get_tokenizer():
    tokenizer_path = hf_hub_download(
        SAFE_GPT_REPO_ID,
        filename='tokenizer.json',
        revision=SAFE_GPT_TOKENIZER_REVISION,
    )
    with open(tokenizer_path, 'rb') as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    if digest != SAFE_GPT_TOKENIZER_SHA256:
        raise RuntimeError(
            f'Pinned SAFE tokenizer checksum mismatch: {digest} != '
            f'{SAFE_GPT_TOKENIZER_SHA256}'
        )
    tk = SAFETokenizer.from_pretrained(tokenizer_path).get_pretrained()
    tk.add_tokens(['<', '>'])   # for bracket_safe
    return tk


class Collator:
    """Tokenize SAFE strings from hosted or user-defined dataset schemas."""

    TEXT_COLUMNS = ('safe', 'input')

    def __init__(self, config):
        self.tokenizer = get_tokenizer()
        self.max_length = config.model.max_position_embeddings
        self.use_bracket_safe = config.training.get('use_bracket_safe')

    @classmethod
    def _read_safe(cls, example):
        for column in cls.TEXT_COLUMNS:
            value = example.get(column)
            if isinstance(value, str) and value:
                return value
        raise KeyError(
            f"Expected one non-empty SAFE string in {cls.TEXT_COLUMNS}; "
            f"received columns {sorted(example)}."
        )

    def __call__(self, examples):
        safe_strings = [self._read_safe(example) for example in examples]
        if self.use_bracket_safe:
            safe_strings = [safe2bracketsafe(value) for value in safe_strings]

        batch = self.tokenizer(
            safe_strings,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        batch.pop('token_type_ids', None)
        return batch
    

class UserDataset(datasets.Dataset):
    def __init__(self, data_path):
        with open(data_path) as f:
            self.safe_list = f.readlines()
        self.safe_list = [s.strip('\n') for s in self.safe_list]
        
    def __len__(self):
        return len(self.safe_list)

    def __getitem__(self, index):
        return {'input': self.safe_list[index]}
    

def _validate_distributed_identity(streaming_rank, streaming_world_size):
    """Validate the process identity used to shard hosted streaming data."""

    if (
        isinstance(streaming_rank, bool)
        or not isinstance(streaming_rank, int)
        or isinstance(streaming_world_size, bool)
        or not isinstance(streaming_world_size, int)
        or streaming_world_size <= 0
        or not 0 <= streaming_rank < streaming_world_size
    ):
        raise ValueError(
            "distributed data identity requires integer 0 <= streaming_rank < "
            "streaming_world_size"
        )
    return streaming_rank, streaming_world_size


def _shard_hosted_stream(
    dataset,
    *,
    streaming_rank=None,
    streaming_world_size=None,
):
    """Give each distributed rank a disjoint slice of the pinned stream.

    Lightning cannot inject a ``DistributedSampler`` for an iterable dataset.
    Without this explicit split, every DDP rank replays the same hosted stream
    and the configured global batch overstates the number of unique examples.
    """

    if streaming_rank is None and streaming_world_size is None:
        return dataset
    if streaming_rank is None or streaming_world_size is None:
        raise ValueError(
            "streaming_rank and streaming_world_size must be provided together"
        )
    streaming_rank, streaming_world_size = _validate_distributed_identity(
        streaming_rank, streaming_world_size
    )
    if streaming_world_size == 1:
        return dataset
    return split_dataset_by_node(
        dataset,
        rank=streaming_rank,
        world_size=streaming_world_size,
    )


def get_dataloader(
    config,
    *,
    streaming_rank=None,
    streaming_world_size=None,
):
    if config.data == 'safe':
        dataset = datasets.load_dataset(
            SAFE_GPT_REPO_ID,
            revision=SAFE_GPT_DATASET_REVISION,
            streaming=True,
            split='train',
        )
        dataset = _shard_hosted_stream(
            dataset,
            streaming_rank=streaming_rank,
            streaming_world_size=streaming_world_size,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=config.loader.batch_size,
            collate_fn=Collator(config),
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
            shuffle=False,  # streaming
            persistent_workers=config.loader.num_workers > 0)

    # User-defined dataset
    return torch.utils.data.DataLoader(
        UserDataset(config.data),
        batch_size=config.loader.batch_size,
        collate_fn=Collator(config),
        num_workers=config.loader.num_workers,
        pin_memory=config.loader.pin_memory,
        shuffle=True,
        persistent_workers=config.loader.num_workers > 0)
