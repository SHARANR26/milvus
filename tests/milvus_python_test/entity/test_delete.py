import time
import random
import pdb
import copy
import threading
import logging
from multiprocessing import Pool, Process
import pytest
from utils import *


dim = 128
segment_size = 10
collection_id = "test_delete"
DELETE_TIMEOUT = 60
tag = "1970-01-01"
nb = 6000
field_name = "float_vector"
default_index_name = "insert_index"
entity = gen_entities(1)
raw_vector, binary_entity = gen_binary_entities(1)
entities = gen_entities(nb)
raw_vectors, binary_entities = gen_binary_entities(nb)
default_single_query = {
    "bool": {
        "must": [
            {"vector": {field_name: {"topk": 10, "query": gen_vectors(1, dim), "params": {"nprobe": 10}}}}
        ]
    }
}

def query_with_index(index_name):
    query = copy.deepcopy(default_single_query)
    query["bool"]["must"][0]["vector"]["params"].update({"index_name": default_index_name})
    return query


class TestDeleteBase:
    """
    ******************************************************************
      The following cases are used to test `delete_entity_by_id` function
    ******************************************************************
    """
    @pytest.fixture(
        scope="function",
        params=gen_simple_index()
    )
    def get_simple_index(self, request, connect):
        if str(connect._cmd("mode")) == "GPU":
            if not request.param["index_type"] not in ivf():
                pytest.skip("Only support index_type: idmap/ivf")
        if str(connect._cmd("mode")) == "CPU":
            if request.param["index_type"] in index_cpu_not_support():
                pytest.skip("CPU not support index_type: ivf_sq8h")
        return request.param

    @pytest.fixture(
        scope="function",
        params=[
            1,
            6000
        ],
    )
    def insert_count(self, request):
        yield request.param

    def test_delete_entity_id_not_exised(self, connect, collection):
        '''
        target: test delete entity, params entity_id not existed
        method: add entity and delete
        expected: status DELETED
        '''
        ids = connect.insert(collection, entity)
        connect.flush([collection])
        status = connect.delete_entity_by_id(collection, [0])
        assert status

    # TODO
    def test_delete_empty_collection(self, connect, collection):
        '''
        target: test delete entity, params collection_name not existed
        method: add entity and delete
        expected: status DELETED
        '''
        status = connect.delete_entity_by_id(collection, [0])
        assert status

    def test_delete_entity_collection_not_existed(self, connect, collection):
        '''
        target: test delete entity, params collection_name not existed
        method: add entity and delete
        expected: error raised
        '''
        collection_new = gen_unique_str()
        with pytest.raises(Exception) as e:
            status = connect.delete_entity_by_id(collection_new, [0])

    def test_delete_entity_collection_not_existed(self, connect, collection):
        '''
        target: test delete entity, params collection_name not existed
        method: add entity and delete
        expected: error raised
        '''
        ids = connect.insert(collection, entity)
        connect.flush([collection])
        collection_new = gen_unique_str()
        with pytest.raises(Exception) as e:
            status = connect.delete_entity_by_id(collection_new, [0])

    # TODO:
    def test_insert_delete(self, connect, collection, insert_count):
        '''
        target: test delete entity
        method: add entities and delete
        expected: no error raised
        '''
        entities = gen_entities(insert_count)
        ids = connect.insert(collection, entities)
        connect.flush([collection])
        delete_ids = [ids[0]]
        status = connect.delete_entity_by_id(collection, delete_ids)
        assert status

    def test_insert_delete_A(self, connect, collection):
        '''
        target: test delete entity
        method: add entities and delete one in collection, and one not in collection
        expected: no error raised
        '''
        ids = connect.insert(collection, entities)
        connect.flush([collection])
        delete_ids = [ids[0], 1]
        status = connect.delete_entity_by_id(collection, delete_ids)
        assert status
        connect.flush([collection])
        res_count = connect.count_entities(collection)
        assert res_count == nb - 1

    def test_insert_delete_B(self, connect, collection):
        '''
        target: test delete entity
        method: add entities with the same ids, and delete the id in collection
        expected: no error raised, all entities deleted
        '''
        ids = [1 for i in range(nb)]
        res_ids = connect.insert(collection, entities, ids)
        connect.flush([collection])
        delete_ids = [1]
        status = connect.delete_entity_by_id(collection, delete_ids)
        assert status
        connect.flush([collection])
        res_count = connect.count_entities(collection)
        assert res_count == 0

    def test_delete_exceed_limit(self, connect, collection):
        '''
        target: test delete entity
        method: add one entity and delete two ids
        expected: error raised
        '''        
        ids = connect.insert(collection, entity)
        connect.flush([collection])
        delete_ids = [ids[0], 1]
        status = connect.delete_entity_by_id(collection, delete_ids)
        connect.flush([collection])
        res_count = connect.count_entities(collection)
        assert res_count == 0

    # TODO
    def test_flush_after_delete(self, connect, collection):
        '''
        target: test delete entity
        method: add entities and delete, then flush
        expected: entity deleted and no error raised
        '''
        ids = connect.insert(collection, entities)
        connect.flush([collection])
        delete_ids = [ids[0], ids[-1]]
        status = connect.delete_entity_by_id(collection, delete_ids)
        assert status
        connect.flush([collection])
        res_count = connect.count_entities(collection)
        assert res_count == nb - len(delete_ids)

    # TODO
    def test_flush_after_delete_ip(self, connect, ip_collection):
        '''
        target: test delete entity
        method: add entities and delete, then flush
        expected: entity deleted and no error raised
        '''
        ids = connect.insert(ip_collection, entities)
        connect.flush([ip_collection])
        delete_ids = [ids[0], ids[-1]]
        status = connect.delete_entity_by_id(ip_collection, delete_ids)
        assert status
        connect.flush([ip_collection])
        res_count = connect.count_entities(ip_collection)
        assert res_count == nb - len(delete_ids)

    # TODO
    def test_flush_after_delete_jac(self, connect, jac_collection):
        '''
        target: test delete entity
        method: add entities and delete, then flush
        expected: entity deleted and no error raised
        '''
        ids = connect.insert(jac_collection, binary_entities)
        connect.flush([jac_collection])
        delete_ids = [ids[0], ids[-1]]
        status = connect.delete_entity_by_id(jac_collection, delete_ids)
        assert status
        connect.flush([jac_collection])
        res_count = connect.count_entities(jac_collection)
        assert res_count == nb - len(delete_ids)

    # TODO
    def test_insert_same_ids_after_delete(self, connect, collection):
        '''
        method: add entities and delete
        expected: status DELETED
        '''
        insert_ids = [i for i in range(nb)]
        ids = connect.insert(collection, entities, insert_ids)
        connect.flush([collection])
        delete_ids = [ids[0], ids[-1]]
        status = connect.delete_entity_by_id(collection, delete_ids)
        assert status
        new_ids = connect.insert(collection, entity, [ids[0]])
        assert new_ids == [ids[0]]
        connect.flush([collection])
        res_count = connect.count_entities(collection)
        assert res_count == nb - 1

    # TODO
    @pytest.mark.level(2)
    def test_insert_same_ids_after_delete_ip(self, connect, ip_collection):
        '''
        method: add entities and delete
        expected: status DELETED
        '''
        insert_ids = [i for i in range(nb)]
        ids = connect.insert(ip_collection, entities, insert_ids)
        connect.flush([ip_collection])
        delete_ids = [ids[0], ids[-1]]
        status = connect.delete_entity_by_id(ip_collection, delete_ids)
        assert status
        new_ids = connect.insert(ip_collection, entity, [ids[0]])
        assert new_ids == [ids[0]]
        connect.flush([ip_collection])
        res_count = connect.count_entities(ip_collection)
        assert res_count == nb - 1

    # TODO
    def test_insert_same_ids_after_delete_jac(self, connect, jac_collection):
        '''
        method: add entities, with the same id and delete the ids
        expected: status DELETED, all id deleted
        '''
        insert_ids = [i for i in range(nb)]
        ids = connect.insert(jac_collection, binary_entities, insert_ids)
        connect.flush([jac_collection])
        delete_ids = [ids[0], ids[-1]]
        status = connect.delete_entity_by_id(jac_collection, delete_ids)
        assert status
        new_ids = connect.insert(jac_collection, binary_entity, [ids[0]])
        assert new_ids == [ids[0]]
        connect.flush([jac_collection])
        res_count = connect.count_entities(jac_collection)
        assert res_count == nb - 1

    # TODO:
    def test_search_after_delete(self, connect, collection):
        '''
        target: test delete entity
        method: add entities and delete, then search
        expected: entity deleted and no error raised
        '''
        ids = connect.insert(collection, entities)
        connect.flush([collection])
        delete_ids = [ids[0], ids[-1]]
        status = connect.delete_entity_by_id(collection, delete_ids)
        assert status
        query = copy.deepcopy(default_single_query)
        query["bool"]["must"][0]["vector"][field_name]["query"] = [entity[-1]["values"][0], entities[-1]["values"][0], entities[-1]["values"][-1]]
        res = connect.search(collection, query)
        logging.getLogger().debug(res)
        assert len(res) == len(query["bool"]["must"][0]["vector"][field_name]["query"])
        assert res[0]._distances[0] > epsilon
        assert res[1]._distances[0] < epsilon
        assert res[2]._distances[0] < epsilon

    # TODO
    def test_create_index_after_delete(self, connect, collection, get_simple_index):
        '''
        method: add entitys and delete, then create index
        expected: vectors deleted, index created
        '''
        ids = connect.insert(collection, entities)
        connect.flush([collection])
        delete_ids = [ids[0], ids[-1]]
        status = connect.delete_entity_by_id(collection, delete_ids)
        connect.create_index(collection, field_name, default_index_name, get_simple_index)
        # assert index info

    # TODO
    def test_delete_multiable_times(self, connect, collection):
        '''
        method: add entities and delete id serveral times
        expected: entities deleted
        '''
        ids = connect.insert(collection, entities)
        connect.flush([collection])
        delete_ids = [ids[0], ids[-1]]
        status = connect.delete_entity_by_id(collection, delete_ids)
        assert status
        connect.flush([collection])
        for i in range(10):
            status = connect.delete_entity_by_id(collection, delete_ids)
            assert status

    # TODO
    def test_index_insert_batch_delete_get(self, connect, collection, get_simple_index):
        '''
        method: create index, insert entities, and delete
        expected: entities deleted
        '''
        connect.create_index(collection, field_name, default_index_name, get_simple_index)
        ids = connect.insert(collection, entities)
        connect.flush([collection])
        delete_ids = [ids[0], ids[-1]]
        status = connect.delete_entity_by_id(collection, delete_ids)
        assert status
        connect.flush([collection])
        res_count = connect.count_entities(collection)
        assert res_count == nb - len(delete_ids)
        res_get = connect.get_entity_by_id(collection, delete_ids)
        assert res_get[0] is None

    # TODO
    def test_index_insert_single_delete_get(self, connect, collection, get_simple_index):
        '''
        method: create index, insert entities, and delete
        expected: entities deleted
        '''
        ids = [i for i in range(nb)]
        connect.create_index(collection, field_name, default_index_name, get_simple_index)
        for i in range(nb):
            connect.insert(collection, entity, [ids[i]])
        connect.flush([collection])
        delete_ids = [ids[0], ids[-1]]
        status = connect.delete_entity_by_id(collection, delete_ids)
        assert status
        connect.flush([collection])
        res_count = connect.count_entities(collection)
        assert res_count == nb - len(delete_ids)

    """
    ******************************************************************
      The following cases are used to test `delete_entity_by_id` function, with tags
    ******************************************************************
    """
    # TODO:
    def test_insert_tag_delete(self, connect, collection):
        '''
        method: add entitys with given tag, delete entities with the return ids
        expected: entities deleted
        '''
        connect.create_partition(collection, tag)
        ids = connect.insert(collection, entities, partition_tag=tag)
        connect.flush([collection])
        delete_ids = [ids[0], ids[-1]]
        status = connect.delete_entity_by_id(collection, delete_ids)
        assert status

    # TODO:
    def test_insert_default_tag_delete(self, connect, collection):
        '''
        method: add entitys, delete entities with the return ids
        expected: entities deleted
        '''
        connect.create_partition(collection, tag)
        ids = connect.insert(collection, entities)
        connect.flush([collection])
        delete_ids = [ids[0], ids[-1]]
        status = connect.delete_entity_by_id(collection, delete_ids)
        assert status

    # TODO:
    def test_insert_tags_delete(self, connect, collection):
        '''
        method: add entitys with given two tags, delete entities with the return ids
        expected: entities deleted
        '''
        tag_new = "tag_new"
        connect.create_partition(collection, tag)
        connect.create_partition(collection, tag_new)
        ids = connect.insert(collection, entities, partition_tag=tag)
        ids_new = connect.insert(collection, entities, partition_tag=tag_new)
        connect.flush([collection])
        delete_ids = [ids[0], ids_new[0]]
        status = connect.delete_entity_by_id(collection, delete_ids)
        assert status
        connect.flush([collection])
        res_count = connect.count_entities(collection)
        assert res_count == 2 * (nb - 1)

    # TODO:
    def test_insert_tags_index_delete(self, connect, collection, get_simple_index):
        '''
        method: add entitys with given tag, create index, delete entities with the return ids
        expected: entities deleted
        '''
        tag_new = "tag_new"
        connect.create_partition(collection, tag)
        connect.create_partition(collection, tag_new)
        ids = connect.insert(collection, entities, partition_tag=tag)
        ids_new = connect.insert(collection, entities, partition_tag=tag_new)
        connect.flush([collection])
        connect.create_index(collection, field_name, default_index_name, get_simple_index)
        delete_ids = [ids[0], ids_new[0]]
        status = connect.delete_entity_by_id(collection, delete_ids)
        assert status
        connect.flush([collection])
        res_count = connect.count_entities(collection)
        assert res_count == 2 * (nb - 1)


class TestDeleteInvalid(object):

    """
    Test adding vectors with invalid vectors
    """
    @pytest.fixture(
        scope="function",
        params=gen_invalid_ints()
    )
    def gen_entity_id(self, request):
        yield request.param

    @pytest.fixture(
        scope="function",
        params=gen_invalid_strs()
    )
    def get_collection_name(self, request):
        yield request.param

    @pytest.mark.level(1)
    def test_delete_entity_id_invalid(self, connect, collection, gen_entity_id):
        invalid_id = gen_entity_id
        with pytest.raises(Exception) as e:
            status = connect.delete_entity_by_id(collection, [invalid_id])

    @pytest.mark.level(2)
    def test_delete_entity_ids_invalid(self, connect, collection, gen_entity_id):
        invalid_id = gen_entity_id
        with pytest.raises(Exception) as e:
            status = connect.delete_entity_by_id(collection, [1, invalid_id])

    @pytest.mark.level(2)
    def test_delete_entity_with_invalid_collection_name(self, connect, get_collection_name):
        collection_name = get_collection_name
        with pytest.raises(Exception) as e:
            status = connect.delete_entity_by_id(collection_name, [1])

    # def test_insert_same_ids_after_delete_jac(self, connect, jac_collection):
    #     '''
    #     method: add entities and delete
    #     expected: status DELETED
    #     '''
    #     insert_ids = [i for i in range(nb)]
    #     ids = connect.insert(jac_collection, binary_entities, insert_ids)
    #     connect.flush([jac_collection])
    #     delete_ids = [ids[0], ids[-1]]
    #     with pytest.raises(Exception) as e:
    #         status = connect.delete_entity_by_id(jac_collection, delete_ids)