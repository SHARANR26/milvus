package datanode

import (
	"fmt"
	"strconv"
	"sync"

	"github.com/zilliztech/milvus-distributed/internal/errors"
)

type collectionReplica interface {

	// collection
	getCollectionNum() int
	addCollection(collectionID UniqueID, schemaBlob string) error
	removeCollection(collectionID UniqueID) error
	getCollectionByID(collectionID UniqueID) (*Collection, error)
	getCollectionByName(collectionName string) (*Collection, error)
	hasCollection(collectionID UniqueID) bool
	getSegmentStatistics() // GOOSE TODO
}

type collectionReplicaImpl struct {
	mu          sync.RWMutex
	collections []*Collection
}

//----------------------------------------------------------------------------------------------------- collection
func (colReplica *collectionReplicaImpl) getSegmentStatistics() {
	// GOOSE TODO
}

func (colReplica *collectionReplicaImpl) getCollectionNum() int {
	colReplica.mu.RLock()
	defer colReplica.mu.RUnlock()

	return len(colReplica.collections)
}

func (colReplica *collectionReplicaImpl) addCollection(collectionID UniqueID, schemaBlob string) error {
	colReplica.mu.Lock()
	defer colReplica.mu.Unlock()

	var newCollection = newCollection(collectionID, schemaBlob)
	colReplica.collections = append(colReplica.collections, newCollection)
	fmt.Println("yyy, create collection: ", newCollection.Name())

	return nil
}

func (colReplica *collectionReplicaImpl) removeCollection(collectionID UniqueID) error {
	fmt.Println("drop collection:", collectionID)
	colReplica.mu.Lock()
	defer colReplica.mu.Unlock()

	tmpCollections := make([]*Collection, 0)
	for _, col := range colReplica.collections {
		if col.ID() != collectionID {
			tmpCollections = append(tmpCollections, col)
		} else {
			fmt.Println("yyy, drop collection name: ", col.Name())
		}
	}
	colReplica.collections = tmpCollections
	return nil
}

func (colReplica *collectionReplicaImpl) getCollectionByID(collectionID UniqueID) (*Collection, error) {
	colReplica.mu.RLock()
	defer colReplica.mu.RUnlock()

	for _, collection := range colReplica.collections {
		if collection.ID() == collectionID {
			return collection, nil
		}
	}

	return nil, errors.New("cannot find collection, id = " + strconv.FormatInt(collectionID, 10))
}

func (colReplica *collectionReplicaImpl) getCollectionByName(collectionName string) (*Collection, error) {
	colReplica.mu.RLock()
	defer colReplica.mu.RUnlock()

	for _, collection := range colReplica.collections {
		if collection.Name() == collectionName {
			return collection, nil
		}
	}

	return nil, errors.New("Cannot found collection: " + collectionName)
}

func (colReplica *collectionReplicaImpl) hasCollection(collectionID UniqueID) bool {
	colReplica.mu.RLock()
	defer colReplica.mu.RUnlock()

	for _, col := range colReplica.collections {
		if col.ID() == collectionID {
			return true
		}
	}
	return false
}
