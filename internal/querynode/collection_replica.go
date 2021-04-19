package querynode

/*

#cgo CFLAGS: -I${SRCDIR}/../core/output/include

#cgo LDFLAGS: -L${SRCDIR}/../core/output/lib -lmilvus_segcore -Wl,-rpath=${SRCDIR}/../core/output/lib

#include "segcore/collection_c.h"
#include "segcore/segment_c.h"

*/
import "C"
import (
	"fmt"
	"log"
	"strconv"
	"sync"

	"github.com/zilliztech/milvus-distributed/internal/errors"
	"github.com/zilliztech/milvus-distributed/internal/proto/etcdpb"
	"github.com/zilliztech/milvus-distributed/internal/proto/internalpb2"
	"github.com/zilliztech/milvus-distributed/internal/proto/schemapb"
)

/*
 * collectionReplica contains a in-memory local copy of persistent collections.
 * In common cases, the system has multiple query nodes. Data of a collection will be
 * distributed across all the available query nodes, and each query node's collectionReplica
 * will maintain its own share (only part of the collection).
 * Every replica tracks a value called tSafe which is the maximum timestamp that the replica
 * is up-to-date.
 */
type collectionReplica interface {
	getTSafe() tSafe

	// collection
	getCollectionNum() int
	addCollection(collectionID UniqueID, schema *schemapb.CollectionSchema) error
	removeCollection(collectionID UniqueID) error
	getCollectionByID(collectionID UniqueID) (*Collection, error)
	hasCollection(collectionID UniqueID) bool
	getVecFieldsByCollectionID(collectionID UniqueID) ([]int64, error)

	// partition
	// TODO: remove collection ID, add a `map[partitionID]partition` to replica implement
	getPartitionNum(collectionID UniqueID) (int, error)
	addPartition(collectionID UniqueID, partitionID UniqueID) error
	removePartition(collectionID UniqueID, partitionID UniqueID) error
	addPartitionsByCollectionMeta(colMeta *etcdpb.CollectionInfo) error
	removePartitionsByCollectionMeta(colMeta *etcdpb.CollectionInfo) error
	getPartitionByID(collectionID UniqueID, partitionID UniqueID) (*Partition, error)
	hasPartition(collectionID UniqueID, partitionID UniqueID) bool
	enablePartitionDM(collectionID UniqueID, partitionID UniqueID) error
	disablePartitionDM(collectionID UniqueID, partitionID UniqueID) error
	getEnablePartitionDM(collectionID UniqueID, partitionID UniqueID) (bool, error)

	// segment
	getSegmentNum() int
	getSegmentStatistics() []*internalpb2.SegmentStats
	addSegment(segmentID UniqueID, partitionID UniqueID, collectionID UniqueID, segType segmentType) error
	removeSegment(segmentID UniqueID) error
	getSegmentByID(segmentID UniqueID) (*Segment, error)
	hasSegment(segmentID UniqueID) bool
	getSealedSegments() ([]UniqueID, []UniqueID)
	replaceGrowingSegmentBySealedSegment(segment *Segment) error

	freeAll()
}

type collectionReplicaImpl struct {
	tSafe tSafe

	mu          sync.RWMutex // guards collections and segments
	collections []*Collection
	segments    map[UniqueID]*Segment
}

//----------------------------------------------------------------------------------------------------- tSafe
func (colReplica *collectionReplicaImpl) getTSafe() tSafe {
	return colReplica.tSafe
}

//----------------------------------------------------------------------------------------------------- collection
func (colReplica *collectionReplicaImpl) getCollectionNum() int {
	colReplica.mu.RLock()
	defer colReplica.mu.RUnlock()

	return len(colReplica.collections)
}

func (colReplica *collectionReplicaImpl) addCollection(collectionID UniqueID, schema *schemapb.CollectionSchema) error {
	colReplica.mu.Lock()
	defer colReplica.mu.Unlock()

	var newCollection = newCollection(collectionID, schema)
	colReplica.collections = append(colReplica.collections, newCollection)

	return nil
}

func (colReplica *collectionReplicaImpl) removeCollection(collectionID UniqueID) error {
	colReplica.mu.Lock()
	defer colReplica.mu.Unlock()

	collection, err := colReplica.getCollectionByIDPrivate(collectionID)
	if err != nil {
		return err
	}

	deleteCollection(collection)

	tmpCollections := make([]*Collection, 0)
	for _, col := range colReplica.collections {
		if col.ID() == collectionID {
			for _, p := range *col.Partitions() {
				for _, s := range *p.Segments() {
					deleteSegment(colReplica.segments[s.ID()])
					delete(colReplica.segments, s.ID())
				}
			}
		} else {
			tmpCollections = append(tmpCollections, col)
		}
	}

	colReplica.collections = tmpCollections
	return nil
}

func (colReplica *collectionReplicaImpl) getCollectionByID(collectionID UniqueID) (*Collection, error) {
	colReplica.mu.RLock()
	defer colReplica.mu.RUnlock()

	return colReplica.getCollectionByIDPrivate(collectionID)
}

func (colReplica *collectionReplicaImpl) getCollectionByIDPrivate(collectionID UniqueID) (*Collection, error) {
	for _, collection := range colReplica.collections {
		if collection.ID() == collectionID {
			return collection, nil
		}
	}

	return nil, errors.New("cannot find collection, id = " + strconv.FormatInt(collectionID, 10))
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

func (colReplica *collectionReplicaImpl) getVecFieldsByCollectionID(collectionID UniqueID) ([]int64, error) {
	colReplica.mu.RLock()
	defer colReplica.mu.RUnlock()

	col, err := colReplica.getCollectionByIDPrivate(collectionID)
	if err != nil {
		return nil, err
	}

	vecFields := make([]int64, 0)
	for _, field := range col.Schema().Fields {
		if field.DataType == schemapb.DataType_VECTOR_BINARY || field.DataType == schemapb.DataType_VECTOR_FLOAT {
			vecFields = append(vecFields, field.FieldID)
		}
	}

	if len(vecFields) <= 0 {
		return nil, errors.New("no vector field in segment " + strconv.FormatInt(collectionID, 10))
	}

	return vecFields, nil
}

//----------------------------------------------------------------------------------------------------- partition
func (colReplica *collectionReplicaImpl) getPartitionNum(collectionID UniqueID) (int, error) {
	colReplica.mu.RLock()
	defer colReplica.mu.RUnlock()

	collection, err := colReplica.getCollectionByIDPrivate(collectionID)
	if err != nil {
		return -1, err
	}

	return len(collection.partitions), nil
}

func (colReplica *collectionReplicaImpl) addPartition(collectionID UniqueID, partitionID UniqueID) error {
	colReplica.mu.Lock()
	defer colReplica.mu.Unlock()

	collection, err := colReplica.getCollectionByIDPrivate(collectionID)
	if err != nil {
		return err
	}

	var newPartition = newPartition(partitionID)

	*collection.Partitions() = append(*collection.Partitions(), newPartition)
	return nil
}

func (colReplica *collectionReplicaImpl) removePartition(collectionID UniqueID, partitionID UniqueID) error {
	colReplica.mu.Lock()
	defer colReplica.mu.Unlock()

	return colReplica.removePartitionPrivate(collectionID, partitionID)
}

func (colReplica *collectionReplicaImpl) removePartitionPrivate(collectionID UniqueID, partitionID UniqueID) error {
	collection, err := colReplica.getCollectionByIDPrivate(collectionID)
	if err != nil {
		return err
	}

	var tmpPartitions = make([]*Partition, 0)
	for _, p := range *collection.Partitions() {
		if p.ID() == partitionID {
			for _, s := range *p.Segments() {
				deleteSegment(colReplica.segments[s.ID()])
				delete(colReplica.segments, s.ID())
			}
		} else {
			tmpPartitions = append(tmpPartitions, p)
		}
	}

	*collection.Partitions() = tmpPartitions
	return nil
}

// deprecated
func (colReplica *collectionReplicaImpl) addPartitionsByCollectionMeta(colMeta *etcdpb.CollectionInfo) error {
	if !colReplica.hasCollection(colMeta.ID) {
		err := errors.New("Cannot find collection, id = " + strconv.FormatInt(colMeta.ID, 10))
		return err
	}
	pToAdd := make([]UniqueID, 0)
	for _, partitionID := range colMeta.PartitionIDs {
		if !colReplica.hasPartition(colMeta.ID, partitionID) {
			pToAdd = append(pToAdd, partitionID)
		}
	}

	for _, id := range pToAdd {
		err := colReplica.addPartition(colMeta.ID, id)
		if err != nil {
			log.Println(err)
		}
		fmt.Println("add partition: ", id)
	}

	return nil
}

func (colReplica *collectionReplicaImpl) removePartitionsByCollectionMeta(colMeta *etcdpb.CollectionInfo) error {
	colReplica.mu.Lock()
	defer colReplica.mu.Unlock()

	col, err := colReplica.getCollectionByIDPrivate(colMeta.ID)
	if err != nil {
		return err
	}

	pToDel := make([]UniqueID, 0)
	for _, partition := range col.partitions {
		hasPartition := false
		for _, id := range colMeta.PartitionIDs {
			if partition.ID() == id {
				hasPartition = true
			}
		}
		if !hasPartition {
			pToDel = append(pToDel, partition.ID())
		}
	}

	for _, id := range pToDel {
		err := colReplica.removePartitionPrivate(col.ID(), id)
		if err != nil {
			log.Println(err)
		}
		fmt.Println("delete partition: ", id)
	}

	return nil
}

func (colReplica *collectionReplicaImpl) getPartitionByID(collectionID UniqueID, partitionID UniqueID) (*Partition, error) {
	colReplica.mu.RLock()
	defer colReplica.mu.RUnlock()

	return colReplica.getPartitionByIDPrivate(collectionID, partitionID)
}

func (colReplica *collectionReplicaImpl) getPartitionByIDPrivate(collectionID UniqueID, partitionID UniqueID) (*Partition, error) {
	collection, err := colReplica.getCollectionByIDPrivate(collectionID)
	if err != nil {
		return nil, err
	}

	for _, p := range *collection.Partitions() {
		if p.ID() == partitionID {
			return p, nil
		}
	}

	return nil, errors.New("cannot find partition, id = " + strconv.FormatInt(partitionID, 10))
}

func (colReplica *collectionReplicaImpl) hasPartition(collectionID UniqueID, partitionID UniqueID) bool {
	colReplica.mu.RLock()
	defer colReplica.mu.RUnlock()

	collection, err := colReplica.getCollectionByIDPrivate(collectionID)
	if err != nil {
		log.Println(err)
		return false
	}

	for _, p := range *collection.Partitions() {
		if p.ID() == partitionID {
			return true
		}
	}

	return false
}

func (colReplica *collectionReplicaImpl) enablePartitionDM(collectionID UniqueID, partitionID UniqueID) error {
	colReplica.mu.Lock()
	defer colReplica.mu.Unlock()

	partition, err := colReplica.getPartitionByIDPrivate(collectionID, partitionID)
	if err != nil {
		return err
	}

	partition.enableDM = true
	return nil
}

func (colReplica *collectionReplicaImpl) disablePartitionDM(collectionID UniqueID, partitionID UniqueID) error {
	colReplica.mu.Lock()
	defer colReplica.mu.Unlock()

	partition, err := colReplica.getPartitionByIDPrivate(collectionID, partitionID)
	if err != nil {
		return err
	}

	partition.enableDM = false
	return nil
}

func (colReplica *collectionReplicaImpl) getEnablePartitionDM(collectionID UniqueID, partitionID UniqueID) (bool, error) {
	colReplica.mu.Lock()
	defer colReplica.mu.Unlock()

	partition, err := colReplica.getPartitionByIDPrivate(collectionID, partitionID)
	if err != nil {
		return false, err
	}
	return partition.enableDM, nil
}

//----------------------------------------------------------------------------------------------------- segment
func (colReplica *collectionReplicaImpl) getSegmentNum() int {
	colReplica.mu.RLock()
	defer colReplica.mu.RUnlock()

	return len(colReplica.segments)
}

func (colReplica *collectionReplicaImpl) getSegmentStatistics() []*internalpb2.SegmentStats {
	colReplica.mu.RLock()
	defer colReplica.mu.RUnlock()

	var statisticData = make([]*internalpb2.SegmentStats, 0)

	for segmentID, segment := range colReplica.segments {
		currentMemSize := segment.getMemSize()
		segment.lastMemSize = currentMemSize
		segmentNumOfRows := segment.getRowCount()

		stat := internalpb2.SegmentStats{
			SegmentID:        segmentID,
			MemorySize:       currentMemSize,
			NumRows:          segmentNumOfRows,
			RecentlyModified: segment.getRecentlyModified(),
		}

		statisticData = append(statisticData, &stat)
		segment.setRecentlyModified(false)
	}

	return statisticData
}

func (colReplica *collectionReplicaImpl) addSegment(segmentID UniqueID, partitionID UniqueID, collectionID UniqueID, segType segmentType) error {
	colReplica.mu.Lock()
	defer colReplica.mu.Unlock()

	collection, err := colReplica.getCollectionByIDPrivate(collectionID)
	if err != nil {
		return err
	}

	partition, err2 := colReplica.getPartitionByIDPrivate(collectionID, partitionID)
	if err2 != nil {
		return err2
	}

	var newSegment = newSegment(collection, segmentID, partitionID, collectionID, segType)

	colReplica.segments[segmentID] = newSegment
	*partition.Segments() = append(*partition.Segments(), newSegment)

	return nil
}

func (colReplica *collectionReplicaImpl) removeSegment(segmentID UniqueID) error {
	colReplica.mu.Lock()
	defer colReplica.mu.Unlock()

	return colReplica.removeSegmentPrivate(segmentID)
}

func (colReplica *collectionReplicaImpl) removeSegmentPrivate(segmentID UniqueID) error {
	var targetPartition *Partition
	var segmentIndex = -1

	for _, col := range colReplica.collections {
		for _, p := range *col.Partitions() {
			for i, s := range *p.Segments() {
				if s.ID() == segmentID {
					targetPartition = p
					segmentIndex = i
					deleteSegment(colReplica.segments[s.ID()])
				}
			}
		}
	}

	delete(colReplica.segments, segmentID)

	if targetPartition != nil && segmentIndex > 0 {
		targetPartition.segments = append(targetPartition.segments[:segmentIndex], targetPartition.segments[segmentIndex+1:]...)
	}

	return nil
}

func (colReplica *collectionReplicaImpl) getSegmentByID(segmentID UniqueID) (*Segment, error) {
	colReplica.mu.RLock()
	defer colReplica.mu.RUnlock()

	return colReplica.getSegmentByIDPrivate(segmentID)
}

func (colReplica *collectionReplicaImpl) getSegmentByIDPrivate(segmentID UniqueID) (*Segment, error) {
	targetSegment, ok := colReplica.segments[segmentID]

	if !ok {
		return nil, errors.New("cannot found segment with id = " + strconv.FormatInt(segmentID, 10))
	}

	return targetSegment, nil
}

func (colReplica *collectionReplicaImpl) hasSegment(segmentID UniqueID) bool {
	colReplica.mu.RLock()
	defer colReplica.mu.RUnlock()

	_, ok := colReplica.segments[segmentID]

	return ok
}

func (colReplica *collectionReplicaImpl) getSealedSegments() ([]UniqueID, []UniqueID) {
	colReplica.mu.RLock()
	defer colReplica.mu.RUnlock()

	collectionIDs := make([]UniqueID, 0)
	segmentIDs := make([]UniqueID, 0)
	for k, v := range colReplica.segments {
		if v.getType() == segTypeSealed {
			collectionIDs = append(collectionIDs, v.collectionID)
			segmentIDs = append(segmentIDs, k)
		}
	}

	return collectionIDs, segmentIDs
}

func (colReplica *collectionReplicaImpl) replaceGrowingSegmentBySealedSegment(segment *Segment) error {
	colReplica.mu.Lock()
	defer colReplica.mu.Unlock()
	targetSegment, ok := colReplica.segments[segment.ID()]
	if ok {
		if targetSegment.segmentType != segTypeGrowing {
			return nil
		}
		deleteSegment(targetSegment)
		targetSegment = segment
	} else {
		// add segment
		targetPartition, err := colReplica.getPartitionByIDPrivate(segment.collectionID, segment.partitionID)
		if err != nil {
			return err
		}
		targetPartition.segments = append(targetPartition.segments, segment)
		colReplica.segments[segment.ID()] = segment
	}
	return nil
}

//-----------------------------------------------------------------------------------------------------
func (colReplica *collectionReplicaImpl) freeAll() {
	colReplica.mu.Lock()
	defer colReplica.mu.Unlock()

	for _, seg := range colReplica.segments {
		deleteSegment(seg)
	}
	for _, col := range colReplica.collections {
		deleteCollection(col)
	}

	colReplica.segments = make(map[UniqueID]*Segment)
	colReplica.collections = make([]*Collection, 0)
}
