package master

import (
	"github.com/zilliztech/milvus-distributed/internal/errors"
	"github.com/zilliztech/milvus-distributed/internal/msgstream"
	"github.com/zilliztech/milvus-distributed/internal/proto/internalpb"
)

type StatsProcessor struct {
	metaTable    *metaTable
	runTimeStats *RuntimeStats

	segmentThreshold       float64
	segmentThresholdFactor float64
	globalTSOAllocator     func() (Timestamp, error)
}

func (processor *StatsProcessor) ProcessQueryNodeStats(msgPack *msgstream.MsgPack) error {
	for _, msg := range msgPack.Msgs {
		statsMsg, ok := msg.(*msgstream.QueryNodeStatsMsg)
		if !ok {
			return errors.Errorf("Type of message is not QueryNodeSegStatsMsg")
		}

		for _, segStat := range statsMsg.GetSegStats() {
			if err := processor.processSegmentStat(segStat); err != nil {
				return err
			}
		}

		for _, fieldStat := range statsMsg.GetFieldStats() {
			if err := processor.processFieldStat(statsMsg.PeerID, fieldStat); err != nil {
				return err
			}
		}

	}

	return nil
}

func (processor *StatsProcessor) processSegmentStat(segStats *internalpb.SegmentStats) error {
	if !segStats.GetRecentlyModified() {
		return nil
	}

	segID := segStats.GetSegmentID()
	segMeta, err := processor.metaTable.GetSegmentByID(segID)
	if err != nil {
		return err
	}

	segMeta.NumRows = segStats.NumRows
	segMeta.MemSize = segStats.MemorySize

	return processor.metaTable.UpdateSegment(segMeta)
}

func (processor *StatsProcessor) processFieldStat(peerID int64, fieldStats *internalpb.FieldStats) error {
	collID := fieldStats.CollectionID
	fieldID := fieldStats.FieldID

	for _, stat := range fieldStats.IndexStats {
		fieldStats := &FieldRuntimeStats{
			peerID:               peerID,
			indexParams:          stat.IndexParams,
			numOfRelatedSegments: stat.NumRelatedSegments,
		}

		if err := processor.runTimeStats.UpdateFieldStat(collID, fieldID, fieldStats); err != nil {
			return err
		}
	}
	return nil
}

func NewStatsProcessor(mt *metaTable, runTimeStats *RuntimeStats, globalTSOAllocator func() (Timestamp, error)) *StatsProcessor {
	return &StatsProcessor{
		metaTable:              mt,
		runTimeStats:           runTimeStats,
		segmentThreshold:       Params.SegmentSize * 1024 * 1024,
		segmentThresholdFactor: Params.SegmentSizeFactor,
		globalTSOAllocator:     globalTSOAllocator,
	}
}
