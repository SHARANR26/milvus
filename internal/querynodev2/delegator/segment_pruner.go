package delegator

import (
	"context"
	"fmt"
	"sort"
	"strconv"

	"github.com/golang/protobuf/proto"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/planpb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/util/clustering"
	"github.com/milvus-io/milvus/internal/util/exprutil"
	"github.com/milvus-io/milvus/internal/util/typeutil"
	"github.com/milvus-io/milvus/pkg/common"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/metrics"
	"github.com/milvus-io/milvus/pkg/util/distance"
	"github.com/milvus-io/milvus/pkg/util/funcutil"
	"github.com/milvus-io/milvus/pkg/util/merr"
	"github.com/milvus-io/milvus/pkg/util/paramtable"
)

type PruneInfo struct {
	filterRatio float64
}

func PruneSegments(ctx context.Context,
	partitionStats map[UniqueID]*storage.PartitionStatsSnapshot,
	searchReq *internalpb.SearchRequest,
	queryReq *internalpb.RetrieveRequest,
	schema *schemapb.CollectionSchema,
	sealedSegments []SnapshotItem,
	info PruneInfo,
) {
	log := log.Ctx(ctx)
	// 1. calculate filtered segments
	filteredSegments := make(map[UniqueID]struct{}, 0)
	clusteringKeyField := typeutil.GetClusteringKeyField(schema.Fields)
	if clusteringKeyField == nil {
		return
	}
	if searchReq != nil {
		// parse searched vectors
		var vectorsHolder commonpb.PlaceholderGroup
		err := proto.Unmarshal(searchReq.GetPlaceholderGroup(), &vectorsHolder)
		if err != nil || len(vectorsHolder.GetPlaceholders()) == 0 {
			return
		}
		vectorsBytes := vectorsHolder.GetPlaceholders()[0].GetValues()
		// parse dim
		dimStr, err := funcutil.GetAttrByKeyFromRepeatedKV(common.DimKey, clusteringKeyField.GetTypeParams())
		if err != nil {
			return
		}
		dimValue, err := strconv.ParseInt(dimStr, 10, 64)
		if err != nil {
			return
		}
		for _, partID := range searchReq.GetPartitionIDs() {
			partStats := partitionStats[partID]
			FilterSegmentsByVector(partStats, searchReq, vectorsBytes, dimValue, clusteringKeyField, filteredSegments, info.filterRatio)
		}
	} else if queryReq != nil {
		// 0. parse expr from plan
		plan := planpb.PlanNode{}
		err := proto.Unmarshal(queryReq.GetSerializedExprPlan(), &plan)
		if err != nil {
			log.Error("failed to unmarshall serialized expr from bytes, failed the operation")
			return
		}
		expr, err := exprutil.ParseExprFromPlan(&plan)
		if err != nil {
			log.Error("failed to parse expr from plan, failed the operation")
			return
		}
		targetRanges, matchALL := exprutil.ParseRanges(expr, exprutil.ClusteringKey)
		if matchALL || targetRanges == nil {
			return
		}
		for _, partID := range queryReq.GetPartitionIDs() {
			partStats := partitionStats[partID]
			FilterSegmentsOnScalarField(partStats, targetRanges, clusteringKeyField, filteredSegments)
		}
	}

	// 2. remove filtered segments from sealed segment list
	if len(filteredSegments) > 0 {
		totalSegNum := 0
		for idx, item := range sealedSegments {
			newSegments := make([]SegmentEntry, 0)
			totalSegNum += len(item.Segments)
			for _, segment := range item.Segments {
				if _, ok := filteredSegments[segment.SegmentID]; !ok {
					newSegments = append(newSegments, segment)
				}
			}
			item.Segments = newSegments
			sealedSegments[idx] = item
		}
		filterRate := float64(len(filteredSegments) / totalSegNum)
		log.RatedInfo(30, "Pruned segment for search/query",
			zap.Int("filtered_segment_num[excluded]", len(filteredSegments)),
			zap.Int("total_segment_num", totalSegNum),
			zap.Float64("filtered_rate", filterRate),
		)
		opType := metrics.SearchLabel
		if queryReq != nil {
			opType = metrics.QueryLabel
		}
		metrics.QueryNodeSegmentPrunedRate.WithLabelValues(fmt.Sprint(paramtable.GetNodeID()), opType).
			Set(filterRate)
	}
}

type segmentDisStruct struct {
	segmentID UniqueID
	distance  float32
	rows      int // for keep track of sufficiency of topK
}

func FilterSegmentsByVector(partitionStats *storage.PartitionStatsSnapshot,
	searchReq *internalpb.SearchRequest,
	vectorBytes [][]byte,
	dim int64,
	keyField *schemapb.FieldSchema,
	filteredSegments map[UniqueID]struct{},
	filterRatio float64,
) {
	// 1. calculate vectors' distances
	neededSegments := make(map[UniqueID]struct{})
	for _, vecBytes := range vectorBytes {
		segmentsToSearch := make([]segmentDisStruct, 0)
		for segId, segStats := range partitionStats.SegmentStats {
			// here, we do not skip needed segments required by former query vector
			// meaning that repeated calculation will be carried and the larger the nq is
			// the more segments have to be included and prune effect will decline
			// 1. calculate distances from centroids
			for _, fieldStat := range segStats.FieldStats {
				if fieldStat.FieldID == keyField.GetFieldID() {
					if fieldStat.Centroids == nil || len(fieldStat.Centroids) == 0 {
						neededSegments[segId] = struct{}{}
						break
					}
					var dis []float32
					var disErr error
					switch keyField.GetDataType() {
					case schemapb.DataType_FloatVector:
						dis, disErr = clustering.CalcVectorDistance(dim, keyField.GetDataType(),
							vecBytes, fieldStat.Centroids[0].GetValue().([]float32), searchReq.GetMetricType())
					default:
						neededSegments[segId] = struct{}{}
						disErr = merr.WrapErrParameterInvalid(schemapb.DataType_FloatVector, keyField.GetDataType(),
							"Currently, pruning by cluster only support float_vector type")
					}
					// currently, we only support float vector and only one center one segment
					if disErr != nil {
						neededSegments[segId] = struct{}{}
						break
					}
					segmentsToSearch = append(segmentsToSearch, segmentDisStruct{
						segmentID: segId,
						distance:  dis[0],
						rows:      segStats.NumRows,
					})
					break
				}
			}
		}
		// 2. sort the distances
		switch searchReq.GetMetricType() {
		case distance.L2:
			sort.SliceStable(segmentsToSearch, func(i, j int) bool {
				return segmentsToSearch[i].distance < segmentsToSearch[j].distance
			})
		case distance.IP, distance.COSINE:
			sort.SliceStable(segmentsToSearch, func(i, j int) bool {
				return segmentsToSearch[i].distance > segmentsToSearch[j].distance
			})
		}

		// 3. filtered non-target segments
		segmentCount := len(segmentsToSearch)
		targetSegNum := int(float64(segmentCount) * filterRatio)
		optimizedRowCount := 0
		// set the last n - targetSegNum as being filtered
		for i := 0; i < segmentCount; i++ {
			optimizedRowCount += segmentsToSearch[i].rows
			neededSegments[segmentsToSearch[i].segmentID] = struct{}{}
			if int64(optimizedRowCount) >= searchReq.GetTopk() && i >= targetSegNum {
				break
			}
		}
	}

	// 3. set not needed segments as removed
	for segId := range partitionStats.SegmentStats {
		if _, ok := neededSegments[segId]; !ok {
			filteredSegments[segId] = struct{}{}
		}
	}
}

func FilterSegmentsOnScalarField(partitionStats *storage.PartitionStatsSnapshot,
	targetRanges []*exprutil.PlanRange,
	keyField *schemapb.FieldSchema,
	filteredSegments map[UniqueID]struct{},
) {
	// 1. try to filter segments
	overlap := func(min storage.ScalarFieldValue, max storage.ScalarFieldValue) bool {
		for _, tRange := range targetRanges {
			switch keyField.DataType {
			case schemapb.DataType_Int8, schemapb.DataType_Int16, schemapb.DataType_Int32, schemapb.DataType_Int64:
				targetRange := tRange.ToIntRange()
				statRange := exprutil.NewIntRange(min.GetValue().(int64), max.GetValue().(int64), true, true)
				return exprutil.IntRangeOverlap(targetRange, statRange)
			case schemapb.DataType_String, schemapb.DataType_VarChar:
				targetRange := tRange.ToStrRange()
				statRange := exprutil.NewStrRange(min.GetValue().(string), max.GetValue().(string), true, true)
				return exprutil.StrRangeOverlap(targetRange, statRange)
			}
		}
		return false
	}
	for segID, segStats := range partitionStats.SegmentStats {
		for _, fieldStat := range segStats.FieldStats {
			if keyField.FieldID == fieldStat.FieldID && !overlap(fieldStat.Min, fieldStat.Max) {
				filteredSegments[segID] = struct{}{}
			}
		}
	}
}
