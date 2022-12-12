package model

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus-proto/go-api/commonpb"
	"github.com/milvus-io/milvus/internal/proto/datapb"
)

var (
	indexID     int64 = 1
	indexName         = "idx"
	indexParams       = []*commonpb.KeyValuePair{
		{
			Key:   "field110-i1",
			Value: "field110-v1",
		},
	}

	indexModel = &Index{
		IndexID:     indexID,
		IndexName:   indexName,
		IndexParams: indexParams,
		IsDeleted:   true,
		CreateTime:  1,
	}

	indexPb = &datapb.FieldIndex{
		IndexInfo: &datapb.IndexInfo{
			CollectionID: colID,
			FieldID:      fieldID,
			IndexName:    indexName,
			IndexID:      indexID,
			TypeParams:   typeParams,
			IndexParams:  indexParams,
		},
		Deleted:    true,
		CreateTime: 1,
	}
)

func TestMarshalIndexModel(t *testing.T) {
	ret := MarshalIndexModel(indexModel)
	assert.Equal(t, indexPb.IndexInfo.IndexID, ret.IndexInfo.IndexID)
	assert.Nil(t, MarshalIndexModel(nil))
}

func TestUnmarshalIndexModel(t *testing.T) {
	ret := UnmarshalIndexModel(indexPb)
	assert.Equal(t, indexModel.IndexID, ret.IndexID)
	assert.Nil(t, UnmarshalIndexModel(nil))
}
