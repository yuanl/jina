import pytest

from jina import Flow, AsyncFlow
from jina.executors import BaseExecutor
from jina.executors.crafters import BaseCrafter
from jina.executors.encoders import BaseEncoder
from jina.executors.indexers import BaseIndexer
from jina.flow import BaseFlow
from jina.jaml import JAML


@pytest.mark.parametrize('f_load_fn', [JAML.load, Flow.load_config])
@pytest.mark.parametrize('f_cls', [Flow, AsyncFlow, BaseFlow])
def test_flow_resolver_from_jaml(f_load_fn, f_cls):
    y = f'''
{f_cls.__name__}:
    version: '1'
    pods:
        - name: hello
    '''
    f = f_load_fn(y)

    assert isinstance(f, f_cls)
    assert f.num_pods == 1


@pytest.mark.parametrize('f_load_fn', [JAML.load, BaseExecutor.load_config])
@pytest.mark.parametrize('f_cls', [BaseEncoder, BaseCrafter, BaseIndexer])
def test_compound_executor_resolver(f_load_fn, f_cls):
    y = f'''
{f_cls.__name__}:
    metas:
      name: my-{f_cls.__name__}  # a customized name
'''
    f = f_load_fn(y)

    assert isinstance(f, f_cls)
