package proxy

import (
	"context"
	"fmt"
	"strconv"

	"github.com/cockroachdb/errors"
	"github.com/golang/protobuf/proto"
	"go.opentelemetry.io/otel"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus/internal/parser/planparserv2"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/util/funcutil"
	"github.com/milvus-io/milvus/pkg/util/merr"
	"github.com/milvus-io/milvus/pkg/util/paramtable"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
)

func initSearchRequest(ctx context.Context, t *searchTask) error {
	ctx, sp := otel.Tracer(typeutil.ProxyRole).Start(ctx, "init search request")
	defer sp.End()

	t.Base.MsgType = commonpb.MsgType_Search
	t.Base.SourceID = paramtable.GetNodeID()

	collectionName := t.request.CollectionName
	t.collectionName = collectionName
	collID, err := globalMetaCache.GetCollectionID(ctx, t.request.GetDbName(), collectionName)
	if err != nil { // err is not nil if collection not exists
		return err
	}

	log := log.Ctx(ctx).With(zap.Int64("collID", collID), zap.String("collName", collectionName))

	t.SearchRequest.DbID = 0 // todo
	t.SearchRequest.CollectionID = collID
	t.schema, err = globalMetaCache.GetCollectionSchema(ctx, t.request.GetDbName(), collectionName)
	if err != nil {
		log.Warn("get collection schema failed", zap.Error(err))
		return err
	}

	// fetch search_growing from search param
	var ignoreGrowing bool
	for i, kv := range t.request.GetSearchParams() {
		if kv.GetKey() == IgnoreGrowingKey {
			ignoreGrowing, err = strconv.ParseBool(kv.GetValue())
			if err != nil {
				return errors.New("parse search growing failed")
			}
			t.request.SearchParams = append(t.request.GetSearchParams()[:i], t.request.GetSearchParams()[i+1:]...)
			break
		}
	}
	t.SearchRequest.IgnoreGrowing = ignoreGrowing

	// Manually update nq if not set.
	nq, err := getNq(t.request)
	if err != nil {
		log.Warn("failed to get nq", zap.Error(err))
		return err
	}
	// Check if nq is valid:
	// https://milvus.io/docs/limitations.md
	if err := validateNQLimit(nq); err != nil {
		return fmt.Errorf("%s [%d] is invalid, %w", NQKey, nq, err)
	}
	t.SearchRequest.Nq = nq
	log = log.With(zap.Int64("nq", nq))

	outputFieldIDs, err := getOutputFieldIDs(t.schema, t.request.GetOutputFields())
	if err != nil {
		log.Warn("fail to get output field ids", zap.Error(err))
		return err
	}
	t.SearchRequest.OutputFieldsId = outputFieldIDs

	partitionNames := t.request.GetPartitionNames()
	if t.request.GetDslType() == commonpb.DslType_BoolExprV1 {
		annsField, err := funcutil.GetAttrByKeyFromRepeatedKV(AnnsFieldKey, t.request.GetSearchParams())
		if err != nil || len(annsField) == 0 {
			vecFields := typeutil.GetVectorFieldSchemas(t.schema.CollectionSchema)
			if len(vecFields) == 0 {
				return errors.New(AnnsFieldKey + " not found in schema")
			}

			if enableMultipleVectorFields && len(vecFields) > 1 {
				return errors.New("multiple anns_fields exist, please specify a anns_field in search_params")
			}

			annsField = vecFields[0].Name
		}
		queryInfo, offset, err := parseSearchInfo(t.request.GetSearchParams(), t.schema.CollectionSchema)
		if err != nil {
			return err
		}
		if queryInfo.GroupByFieldId != 0 {
			t.SearchRequest.IgnoreGrowing = true
			// for group by operation, currently, we ignore growing segments
		}
		t.offset = offset

		plan, err := planparserv2.CreateSearchPlan(t.schema.CollectionSchema, t.request.Dsl, annsField, queryInfo)
		if err != nil {
			log.Warn("failed to create query plan", zap.Error(err),
				zap.String("dsl", t.request.Dsl), // may be very large if large term passed.
				zap.String("anns field", annsField), zap.Any("query info", queryInfo))
			return merr.WrapErrParameterInvalidMsg("failed to create query plan: %v", err)
		}
		log.Debug("create query plan",
			zap.String("dsl", t.request.Dsl), // may be very large if large term passed.
			zap.String("anns field", annsField), zap.Any("query info", queryInfo))

		if t.partitionKeyMode {
			expr, err := ParseExprFromPlan(plan)
			if err != nil {
				log.Warn("failed to parse expr", zap.Error(err))
				return err
			}
			partitionKeys := ParsePartitionKeys(expr)
			hashedPartitionNames, err := assignPartitionKeys(ctx, t.request.GetDbName(), collectionName, partitionKeys)
			if err != nil {
				log.Warn("failed to assign partition keys", zap.Error(err))
				return err
			}

			partitionNames = append(partitionNames, hashedPartitionNames...)
		}

		plan.OutputFieldIds = outputFieldIDs

		t.SearchRequest.Topk = queryInfo.GetTopk()
		t.SearchRequest.MetricType = queryInfo.GetMetricType()
		t.SearchRequest.DslType = commonpb.DslType_BoolExprV1

		estimateSize, err := t.estimateResultSize(nq, t.SearchRequest.Topk)
		if err != nil {
			log.Warn("failed to estimate result size", zap.Error(err))
			return err
		}
		if estimateSize >= requeryThreshold {
			t.requery = true
			plan.OutputFieldIds = nil
		}

		t.SearchRequest.SerializedExprPlan, err = proto.Marshal(plan)
		if err != nil {
			return err
		}

		log.Debug("Proxy::searchTask::PreExecute",
			zap.Int64s("plan.OutputFieldIds", plan.GetOutputFieldIds()),
			zap.Stringer("plan", plan)) // may be very large if large term passed.
	}

	// translate partition name to partition ids. Use regex-pattern to match partition name.
	t.SearchRequest.PartitionIDs, err = getPartitionIDs(ctx, t.request.GetDbName(), collectionName, partitionNames)
	if err != nil {
		log.Warn("failed to get partition ids", zap.Error(err))
		return err
	}

	t.SearchRequest.PlaceholderGroup = t.request.PlaceholderGroup

	// Set username of this search request for feature like task scheduling.
	if username, _ := GetCurUserFromContext(ctx); username != "" {
		t.SearchRequest.Username = username
	}

	return nil
}
