// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package segments

import (
	"context"
	"fmt"
	"sync"
	"time"

	"go.uber.org/atomic"
	"go.uber.org/zap"
	"golang.org/x/sync/errgroup"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus/internal/querynodev2/segments/metricsutil"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/metrics"
	"github.com/milvus-io/milvus/pkg/util/paramtable"
	"github.com/milvus-io/milvus/pkg/util/timerecord"
)

// searchOnSegments performs search on listed segments
// all segment ids are validated before calling this function
func searchSegments(ctx context.Context, mgr *Manager, segments []Segment, segType SegmentType, searchReq *SearchRequest) ([]*SearchResult, error) {
	var (
		// results variables
		resultCh = make(chan *SearchResult, len(segments))
		errs     = make([]error, len(segments))
		wg       sync.WaitGroup

		// For log only
		mu                   sync.Mutex
		segmentsWithoutIndex []int64
	)

	searchLabel := metrics.SealedSegmentLabel
	if segType == commonpb.SegmentState_Growing {
		searchLabel = metrics.GrowingSegmentLabel
	}

	searcher := func(s Segment) error {
		// record search time
		tr := timerecord.NewTimeRecorder("searchOnSegments")
		searchResult, err := s.Search(ctx, searchReq)
		resultCh <- searchResult
		if err != nil {
			return err
		}
		// update metrics
		elapsed := tr.ElapseSpan().Milliseconds()
		metrics.QueryNodeSQSegmentLatency.WithLabelValues(fmt.Sprint(paramtable.GetNodeID()),
			metrics.SearchLabel, searchLabel).Observe(float64(elapsed))
		metrics.QueryNodeSegmentSearchLatencyPerVector.WithLabelValues(fmt.Sprint(paramtable.GetNodeID()),
			metrics.SearchLabel, searchLabel).Observe(float64(elapsed) / float64(searchReq.getNumOfQuery()))
		return nil
	}

	// calling segment search in goroutines
	for i, segment := range segments {
		wg.Add(1)
		go func(seg Segment, i int) {
			defer wg.Done()
			if !seg.ExistIndex(searchReq.searchFieldID) {
				mu.Lock()
				segmentsWithoutIndex = append(segmentsWithoutIndex, seg.ID())
				mu.Unlock()
			}
			var err error
			accessRecord := metricsutil.NewSearchSegmentAccessRecord(getSegmentMetricLabel(seg))
			defer func() {
				accessRecord.Finish(err)
			}()
			if seg.IsLazyLoad() {
				var missing bool
				missing, err = mgr.DiskCache.Do(seg.ID(), searcher)
				if missing {
					accessRecord.CacheMissing()
				}
			} else {
				err = searcher(seg)
			}
			if err != nil {
				errs[i] = err
			}
		}(segment, i)
	}
	wg.Wait()
	close(resultCh)

	searchResults := make([]*SearchResult, 0, len(segments))
	for result := range resultCh {
		searchResults = append(searchResults, result)
	}

	for _, err := range errs {
		if err != nil {
			DeleteSearchResults(searchResults)
			return nil, err
		}
	}

	if len(segmentsWithoutIndex) > 0 {
		log.Ctx(ctx).Debug("search growing/sealed segments without indexes", zap.Int64s("segmentIDs", segmentsWithoutIndex))
	}

	return searchResults, nil
}

// searchSegmentsStreamly performs search on listed segments in a stream mode instead of a batch mode
// all segment ids are validated before calling this function
func searchSegmentsStreamly(ctx context.Context,
	mgr *Manager,
	segments []Segment,
	searchReq *SearchRequest,
	streamReduce func(result *SearchResult) error,
) error {
	searchLabel := metrics.SealedSegmentLabel
	var sumReduceDuration atomic.Duration
	eg, egCtx := errgroup.WithContext(ctx)
	searcher := func(seg Segment) error {
		var searchResult *SearchResult
		defer func() {
			select {
			case <-egCtx.Done():
				DeleteSearchResults([]*SearchResult{searchResult})
			default:
			}
		}()

		// record search time
		tr := timerecord.NewTimeRecorder("searchOnSegments")
		searchResult, searchErr := seg.Search(ctx, searchReq)
		searchDuration := tr.RecordSpan()
		if searchErr != nil {
			return searchErr
		}
		reducedErr := streamReduce(searchResult)
		reduceDuration := tr.RecordSpan()
		if reducedErr != nil {
			return reducedErr
		}
		sumReduceDuration.Add(reduceDuration)
		// update metrics
		metrics.QueryNodeSQSegmentLatency.WithLabelValues(fmt.Sprint(paramtable.GetNodeID()),
			metrics.SearchLabel, searchLabel).Observe(float64(searchDuration))
		metrics.QueryNodeSegmentSearchLatencyPerVector.WithLabelValues(fmt.Sprint(paramtable.GetNodeID()),
			metrics.SearchLabel, searchLabel).Observe(float64(searchDuration) / float64(searchReq.getNumOfQuery()))
		return nil
	}

	for _, seg := range segments {
		segCopy := seg
		eg.Go(func() error {
			var err error
			if segCopy.IsLazyLoad() {
				var timeout time.Duration
				_, err := mgr.DiskCache.DoWait(segCopy.ID(), timeout, searcher)
				if err != nil {
					log.Error("failed to do search for disk cache", zap.Int64("seg_id", segCopy.ID()), zap.Error(err))
					return err
				}
			} else {
				err = searcher(segCopy)
			}
			return err
		})
	}
	err := eg.Wait()
	if err != nil {
		return err
	}
	metrics.QueryNodeReduceLatency.WithLabelValues(fmt.Sprint(paramtable.GetNodeID()),
		metrics.SearchLabel,
		metrics.ReduceSegments,
		metrics.StreamReduce).Observe(float64(sumReduceDuration.Load().Milliseconds()))
	log.Debug("stream reduce sum duration:", zap.Duration("duration", sumReduceDuration.Load()))
	return nil
}

// search will search on the historical segments the target segments in historical.
// if segIDs is not specified, it will search on all the historical segments speficied by partIDs.
// if segIDs is specified, it will only search on the segments specified by the segIDs.
// if partIDs is empty, it means all the partitions of the loaded collection or all the partitions loaded.
func SearchHistorical(ctx context.Context, manager *Manager, searchReq *SearchRequest, collID int64, partIDs []int64, segIDs []int64) ([]*SearchResult, []Segment, error) {
	segments, err := validateOnHistorical(ctx, manager, collID, partIDs, segIDs)
	if err != nil {
		return nil, nil, err
	}
	searchResults, err := searchSegments(ctx, manager, segments, SegmentTypeSealed, searchReq)
	return searchResults, segments, err
}

// searchStreaming will search all the target segments in streaming
// if partIDs is empty, it means all the partitions of the loaded collection or all the partitions loaded.
func SearchStreaming(ctx context.Context, manager *Manager, searchReq *SearchRequest, collID int64, partIDs []int64, segIDs []int64) ([]*SearchResult, []Segment, error) {
	segments, err := validateOnStream(ctx, manager, collID, partIDs, segIDs)
	if err != nil {
		return nil, nil, err
	}
	searchResults, err := searchSegments(ctx, manager, segments, SegmentTypeGrowing, searchReq)
	return searchResults, segments, err
}

func SearchHistoricalStreamly(ctx context.Context, manager *Manager, searchReq *SearchRequest,
	collID int64, partIDs []int64, segIDs []int64, streamReduce func(result *SearchResult) error,
) ([]Segment, error) {
	segments, err := validateOnHistorical(ctx, manager, collID, partIDs, segIDs)
	if err != nil {
		return segments, err
	}
	err = searchSegmentsStreamly(ctx, manager, segments, searchReq, streamReduce)
	if err != nil {
		return segments, err
	}
	return segments, nil
}
