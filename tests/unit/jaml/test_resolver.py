import pytest

from jina import Flow, AsyncFlow
from jina.drivers import BaseDriver
from jina.executors.crafters import BaseCrafter
from jina.executors.encoders import BaseEncoder
from jina.executors.indexers import BaseIndexer, CompoundIndexer
from jina.executors.rankers import BaseRanker
from jina.flow import BaseFlow
from jina.jaml import JAML


@pytest.mark.parametrize('f_load_fn', [JAML.load])
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


@pytest.mark.parametrize('f_load_fn', [JAML.load])
@pytest.mark.parametrize('f_cls', [BaseEncoder, BaseCrafter, BaseIndexer, BaseRanker])
def test_executor_resolver(f_load_fn, f_cls):
    y = f'''
{f_cls.__name__}:
    metas:
      name: my-{f_cls.__name__}  # a customized name
'''
    f = f_load_fn(y)

    assert isinstance(f, f_cls)


@pytest.mark.parametrize('f_load_fn', [JAML.load])
@pytest.mark.parametrize('f_cls', [BaseDriver])
def test_driver_resolver(f_load_fn, f_cls):
    y = f'''
{f_cls.__name__}:
    metas:
      name: my-{f_cls.__name__}  # a customized name
'''
    f = f_load_fn(y)

    assert isinstance(f, f_cls)


def test_empty_encoder_resolver():
    y = '''
BaseEncoder: {} 
'''
    f = JAML.load(y)

    assert isinstance(f, BaseEncoder)

def test_nest_encoder_resolver():
    y = '''
- BaseEncoder: {}
- BaseEncoder: {} 
'''
    f = JAML.load(y)

    assert isinstance(f, list)
    assert isinstance(f[0], BaseEncoder)
