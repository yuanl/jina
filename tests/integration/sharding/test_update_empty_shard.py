import os
import random
import string
from pathlib import Path

import numpy as np
import pytest

from jina import Document, Flow

random.seed(0)
np.random.seed(0)

cur_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def config(tmpdir):
    os.environ['JINA_SHARDING_DIR'] = str(tmpdir)
    yield
    del os.environ['JINA_SHARDING_DIR']


def random_docs(start, end, embed_dim=10):
    for j in range(start, end):
        d = Document()
        d.id = f'{j:0>16}'
        d.tags['id'] = j
        d.text = ''.join(random.choice(string.ascii_lowercase) for _ in range(10)).encode('utf8')
        d.embedding = np.random.random([embed_dim])
        yield d


# fails TODO fix related issue
def test_update_empty_shard(config):
    yaml_file = 'index_kv_simple.yml'
    index_name = 'kvidx'

    with Flow().add(
            uses=os.path.join(cur_dir, 'yaml', yaml_file),
            shards=2,
            separated_workspace=True,
    ) as index_flow:
        index_flow.index(input_fn=random_docs(0, 1))

    with Flow().add(
            uses=os.path.join(cur_dir, 'yaml', yaml_file),
            shards=2,
            separated_workspace=True,
            polling='all',
    ) as update_flow:
        update_flow.update(input_fn=random_docs(0, 1))

    path = Path(os.environ['JINA_SHARDING_DIR'])
    index_files = list(path.glob(f'*/{index_name}.bin'))
    assert len(index_files) == 1
