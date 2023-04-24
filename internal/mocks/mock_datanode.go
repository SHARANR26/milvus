// Code generated by mockery v2.16.0. DO NOT EDIT.

package mocks

import (
	context "context"

	commonpb "github.com/milvus-io/milvus-proto/go-api/commonpb"
	clientv3 "go.etcd.io/etcd/client/v3"

	datapb "github.com/milvus-io/milvus/internal/proto/datapb"

	internalpb "github.com/milvus-io/milvus/internal/proto/internalpb"

	milvuspb "github.com/milvus-io/milvus-proto/go-api/milvuspb"

	mock "github.com/stretchr/testify/mock"

	types "github.com/milvus-io/milvus/internal/types"
)

// DataNodeNode is an autogenerated mock type for the DataNodeComponent type
type DataNodeNode struct {
	mock.Mock
}

type DataNodeNode_Expecter struct {
	mock *mock.Mock
}

func (_m *DataNodeNode) EXPECT() *DataNodeNode_Expecter {
	return &DataNodeNode_Expecter{mock: &_m.Mock}
}

// AddImportSegment provides a mock function with given fields: ctx, req
func (_m *DataNodeNode) AddImportSegment(ctx context.Context, req *datapb.AddImportSegmentRequest) (*datapb.AddImportSegmentResponse, error) {
	ret := _m.Called(ctx, req)

	var r0 *datapb.AddImportSegmentResponse
	if rf, ok := ret.Get(0).(func(context.Context, *datapb.AddImportSegmentRequest) *datapb.AddImportSegmentResponse); ok {
		r0 = rf(ctx, req)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*datapb.AddImportSegmentResponse)
		}
	}

	var r1 error
	if rf, ok := ret.Get(1).(func(context.Context, *datapb.AddImportSegmentRequest) error); ok {
		r1 = rf(ctx, req)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// DataNodeNode_AddImportSegment_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'AddImportSegment'
type DataNodeNode_AddImportSegment_Call struct {
	*mock.Call
}

// AddImportSegment is a helper method to define mock.On call
//   - ctx context.Context
//   - req *datapb.AddImportSegmentRequest
func (_e *DataNodeNode_Expecter) AddImportSegment(ctx interface{}, req interface{}) *DataNodeNode_AddImportSegment_Call {
	return &DataNodeNode_AddImportSegment_Call{Call: _e.mock.On("AddImportSegment", ctx, req)}
}

func (_c *DataNodeNode_AddImportSegment_Call) Run(run func(ctx context.Context, req *datapb.AddImportSegmentRequest)) *DataNodeNode_AddImportSegment_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(*datapb.AddImportSegmentRequest))
	})
	return _c
}

func (_c *DataNodeNode_AddImportSegment_Call) Return(_a0 *datapb.AddImportSegmentResponse, _a1 error) *DataNodeNode_AddImportSegment_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

// Compaction provides a mock function with given fields: ctx, req
func (_m *DataNodeNode) Compaction(ctx context.Context, req *datapb.CompactionPlan) (*commonpb.Status, error) {
	ret := _m.Called(ctx, req)

	var r0 *commonpb.Status
	if rf, ok := ret.Get(0).(func(context.Context, *datapb.CompactionPlan) *commonpb.Status); ok {
		r0 = rf(ctx, req)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*commonpb.Status)
		}
	}

	var r1 error
	if rf, ok := ret.Get(1).(func(context.Context, *datapb.CompactionPlan) error); ok {
		r1 = rf(ctx, req)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// DataNodeNode_Compaction_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Compaction'
type DataNodeNode_Compaction_Call struct {
	*mock.Call
}

// Compaction is a helper method to define mock.On call
//   - ctx context.Context
//   - req *datapb.CompactionPlan
func (_e *DataNodeNode_Expecter) Compaction(ctx interface{}, req interface{}) *DataNodeNode_Compaction_Call {
	return &DataNodeNode_Compaction_Call{Call: _e.mock.On("Compaction", ctx, req)}
}

func (_c *DataNodeNode_Compaction_Call) Run(run func(ctx context.Context, req *datapb.CompactionPlan)) *DataNodeNode_Compaction_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(*datapb.CompactionPlan))
	})
	return _c
}

func (_c *DataNodeNode_Compaction_Call) Return(_a0 *commonpb.Status, _a1 error) *DataNodeNode_Compaction_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

// FlushSegments provides a mock function with given fields: ctx, req
func (_m *DataNodeNode) FlushSegments(ctx context.Context, req *datapb.FlushSegmentsRequest) (*commonpb.Status, error) {
	ret := _m.Called(ctx, req)

	var r0 *commonpb.Status
	if rf, ok := ret.Get(0).(func(context.Context, *datapb.FlushSegmentsRequest) *commonpb.Status); ok {
		r0 = rf(ctx, req)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*commonpb.Status)
		}
	}

	var r1 error
	if rf, ok := ret.Get(1).(func(context.Context, *datapb.FlushSegmentsRequest) error); ok {
		r1 = rf(ctx, req)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// DataNodeNode_FlushSegments_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'FlushSegments'
type DataNodeNode_FlushSegments_Call struct {
	*mock.Call
}

// FlushSegments is a helper method to define mock.On call
//   - ctx context.Context
//   - req *datapb.FlushSegmentsRequest
func (_e *DataNodeNode_Expecter) FlushSegments(ctx interface{}, req interface{}) *DataNodeNode_FlushSegments_Call {
	return &DataNodeNode_FlushSegments_Call{Call: _e.mock.On("FlushSegments", ctx, req)}
}

func (_c *DataNodeNode_FlushSegments_Call) Run(run func(ctx context.Context, req *datapb.FlushSegmentsRequest)) *DataNodeNode_FlushSegments_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(*datapb.FlushSegmentsRequest))
	})
	return _c
}

func (_c *DataNodeNode_FlushSegments_Call) Return(_a0 *commonpb.Status, _a1 error) *DataNodeNode_FlushSegments_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

// GetAddress provides a mock function with given fields:
func (_m *DataNodeNode) GetAddress() string {
	ret := _m.Called()

	var r0 string
	if rf, ok := ret.Get(0).(func() string); ok {
		r0 = rf()
	} else {
		r0 = ret.Get(0).(string)
	}

	return r0
}

// DataNodeNode_GetAddress_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetAddress'
type DataNodeNode_GetAddress_Call struct {
	*mock.Call
}

// GetAddress is a helper method to define mock.On call
func (_e *DataNodeNode_Expecter) GetAddress() *DataNodeNode_GetAddress_Call {
	return &DataNodeNode_GetAddress_Call{Call: _e.mock.On("GetAddress")}
}

func (_c *DataNodeNode_GetAddress_Call) Run(run func()) *DataNodeNode_GetAddress_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *DataNodeNode_GetAddress_Call) Return(_a0 string) *DataNodeNode_GetAddress_Call {
	_c.Call.Return(_a0)
	return _c
}

// GetCompactionState provides a mock function with given fields: ctx, req
func (_m *DataNodeNode) GetCompactionState(ctx context.Context, req *datapb.CompactionStateRequest) (*datapb.CompactionStateResponse, error) {
	ret := _m.Called(ctx, req)

	var r0 *datapb.CompactionStateResponse
	if rf, ok := ret.Get(0).(func(context.Context, *datapb.CompactionStateRequest) *datapb.CompactionStateResponse); ok {
		r0 = rf(ctx, req)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*datapb.CompactionStateResponse)
		}
	}

	var r1 error
	if rf, ok := ret.Get(1).(func(context.Context, *datapb.CompactionStateRequest) error); ok {
		r1 = rf(ctx, req)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// DataNodeNode_GetCompactionState_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetCompactionState'
type DataNodeNode_GetCompactionState_Call struct {
	*mock.Call
}

// GetCompactionState is a helper method to define mock.On call
//   - ctx context.Context
//   - req *datapb.CompactionStateRequest
func (_e *DataNodeNode_Expecter) GetCompactionState(ctx interface{}, req interface{}) *DataNodeNode_GetCompactionState_Call {
	return &DataNodeNode_GetCompactionState_Call{Call: _e.mock.On("GetCompactionState", ctx, req)}
}

func (_c *DataNodeNode_GetCompactionState_Call) Run(run func(ctx context.Context, req *datapb.CompactionStateRequest)) *DataNodeNode_GetCompactionState_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(*datapb.CompactionStateRequest))
	})
	return _c
}

func (_c *DataNodeNode_GetCompactionState_Call) Return(_a0 *datapb.CompactionStateResponse, _a1 error) *DataNodeNode_GetCompactionState_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

// GetComponentStates provides a mock function with given fields: ctx
func (_m *DataNodeNode) GetComponentStates(ctx context.Context) (*milvuspb.ComponentStates, error) {
	ret := _m.Called(ctx)

	var r0 *milvuspb.ComponentStates
	if rf, ok := ret.Get(0).(func(context.Context) *milvuspb.ComponentStates); ok {
		r0 = rf(ctx)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*milvuspb.ComponentStates)
		}
	}

	var r1 error
	if rf, ok := ret.Get(1).(func(context.Context) error); ok {
		r1 = rf(ctx)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// DataNodeNode_GetComponentStates_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetComponentStates'
type DataNodeNode_GetComponentStates_Call struct {
	*mock.Call
}

// GetComponentStates is a helper method to define mock.On call
//   - ctx context.Context
func (_e *DataNodeNode_Expecter) GetComponentStates(ctx interface{}) *DataNodeNode_GetComponentStates_Call {
	return &DataNodeNode_GetComponentStates_Call{Call: _e.mock.On("GetComponentStates", ctx)}
}

func (_c *DataNodeNode_GetComponentStates_Call) Run(run func(ctx context.Context)) *DataNodeNode_GetComponentStates_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context))
	})
	return _c
}

func (_c *DataNodeNode_GetComponentStates_Call) Return(_a0 *milvuspb.ComponentStates, _a1 error) *DataNodeNode_GetComponentStates_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

// GetMetrics provides a mock function with given fields: ctx, req
func (_m *DataNodeNode) GetMetrics(ctx context.Context, req *milvuspb.GetMetricsRequest) (*milvuspb.GetMetricsResponse, error) {
	ret := _m.Called(ctx, req)

	var r0 *milvuspb.GetMetricsResponse
	if rf, ok := ret.Get(0).(func(context.Context, *milvuspb.GetMetricsRequest) *milvuspb.GetMetricsResponse); ok {
		r0 = rf(ctx, req)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*milvuspb.GetMetricsResponse)
		}
	}

	var r1 error
	if rf, ok := ret.Get(1).(func(context.Context, *milvuspb.GetMetricsRequest) error); ok {
		r1 = rf(ctx, req)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// DataNodeNode_GetMetrics_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetMetrics'
type DataNodeNode_GetMetrics_Call struct {
	*mock.Call
}

// GetMetrics is a helper method to define mock.On call
//   - ctx context.Context
//   - req *milvuspb.GetMetricsRequest
func (_e *DataNodeNode_Expecter) GetMetrics(ctx interface{}, req interface{}) *DataNodeNode_GetMetrics_Call {
	return &DataNodeNode_GetMetrics_Call{Call: _e.mock.On("GetMetrics", ctx, req)}
}

func (_c *DataNodeNode_GetMetrics_Call) Run(run func(ctx context.Context, req *milvuspb.GetMetricsRequest)) *DataNodeNode_GetMetrics_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(*milvuspb.GetMetricsRequest))
	})
	return _c
}

func (_c *DataNodeNode_GetMetrics_Call) Return(_a0 *milvuspb.GetMetricsResponse, _a1 error) *DataNodeNode_GetMetrics_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

// GetStateCode provides a mock function with given fields:
func (_m *DataNodeNode) GetStateCode() commonpb.StateCode {
	ret := _m.Called()

	var r0 commonpb.StateCode
	if rf, ok := ret.Get(0).(func() commonpb.StateCode); ok {
		r0 = rf()
	} else {
		r0 = ret.Get(0).(commonpb.StateCode)
	}

	return r0
}

// DataNodeNode_GetStateCode_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetStateCode'
type DataNodeNode_GetStateCode_Call struct {
	*mock.Call
}

// GetStateCode is a helper method to define mock.On call
func (_e *DataNodeNode_Expecter) GetStateCode() *DataNodeNode_GetStateCode_Call {
	return &DataNodeNode_GetStateCode_Call{Call: _e.mock.On("GetStateCode")}
}

func (_c *DataNodeNode_GetStateCode_Call) Run(run func()) *DataNodeNode_GetStateCode_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *DataNodeNode_GetStateCode_Call) Return(_a0 commonpb.StateCode) *DataNodeNode_GetStateCode_Call {
	_c.Call.Return(_a0)
	return _c
}

// GetStatisticsChannel provides a mock function with given fields: ctx
func (_m *DataNodeNode) GetStatisticsChannel(ctx context.Context) (*milvuspb.StringResponse, error) {
	ret := _m.Called(ctx)

	var r0 *milvuspb.StringResponse
	if rf, ok := ret.Get(0).(func(context.Context) *milvuspb.StringResponse); ok {
		r0 = rf(ctx)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*milvuspb.StringResponse)
		}
	}

	var r1 error
	if rf, ok := ret.Get(1).(func(context.Context) error); ok {
		r1 = rf(ctx)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// DataNodeNode_GetStatisticsChannel_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetStatisticsChannel'
type DataNodeNode_GetStatisticsChannel_Call struct {
	*mock.Call
}

// GetStatisticsChannel is a helper method to define mock.On call
//   - ctx context.Context
func (_e *DataNodeNode_Expecter) GetStatisticsChannel(ctx interface{}) *DataNodeNode_GetStatisticsChannel_Call {
	return &DataNodeNode_GetStatisticsChannel_Call{Call: _e.mock.On("GetStatisticsChannel", ctx)}
}

func (_c *DataNodeNode_GetStatisticsChannel_Call) Run(run func(ctx context.Context)) *DataNodeNode_GetStatisticsChannel_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context))
	})
	return _c
}

func (_c *DataNodeNode_GetStatisticsChannel_Call) Return(_a0 *milvuspb.StringResponse, _a1 error) *DataNodeNode_GetStatisticsChannel_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

// Import provides a mock function with given fields: ctx, req
func (_m *DataNodeNode) Import(ctx context.Context, req *datapb.ImportTaskRequest) (*commonpb.Status, error) {
	ret := _m.Called(ctx, req)

	var r0 *commonpb.Status
	if rf, ok := ret.Get(0).(func(context.Context, *datapb.ImportTaskRequest) *commonpb.Status); ok {
		r0 = rf(ctx, req)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*commonpb.Status)
		}
	}

	var r1 error
	if rf, ok := ret.Get(1).(func(context.Context, *datapb.ImportTaskRequest) error); ok {
		r1 = rf(ctx, req)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// DataNodeNode_Import_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Import'
type DataNodeNode_Import_Call struct {
	*mock.Call
}

// Import is a helper method to define mock.On call
//   - ctx context.Context
//   - req *datapb.ImportTaskRequest
func (_e *DataNodeNode_Expecter) Import(ctx interface{}, req interface{}) *DataNodeNode_Import_Call {
	return &DataNodeNode_Import_Call{Call: _e.mock.On("Import", ctx, req)}
}

func (_c *DataNodeNode_Import_Call) Run(run func(ctx context.Context, req *datapb.ImportTaskRequest)) *DataNodeNode_Import_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(*datapb.ImportTaskRequest))
	})
	return _c
}

func (_c *DataNodeNode_Import_Call) Return(_a0 *commonpb.Status, _a1 error) *DataNodeNode_Import_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

// Init provides a mock function with given fields:
func (_m *DataNodeNode) Init() error {
	ret := _m.Called()

	var r0 error
	if rf, ok := ret.Get(0).(func() error); ok {
		r0 = rf()
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// DataNodeNode_Init_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Init'
type DataNodeNode_Init_Call struct {
	*mock.Call
}

// Init is a helper method to define mock.On call
func (_e *DataNodeNode_Expecter) Init() *DataNodeNode_Init_Call {
	return &DataNodeNode_Init_Call{Call: _e.mock.On("Init")}
}

func (_c *DataNodeNode_Init_Call) Run(run func()) *DataNodeNode_Init_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *DataNodeNode_Init_Call) Return(_a0 error) *DataNodeNode_Init_Call {
	_c.Call.Return(_a0)
	return _c
}

// Register provides a mock function with given fields:
func (_m *DataNodeNode) Register() error {
	ret := _m.Called()

	var r0 error
	if rf, ok := ret.Get(0).(func() error); ok {
		r0 = rf()
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// DataNodeNode_Register_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Register'
type DataNodeNode_Register_Call struct {
	*mock.Call
}

// Register is a helper method to define mock.On call
func (_e *DataNodeNode_Expecter) Register() *DataNodeNode_Register_Call {
	return &DataNodeNode_Register_Call{Call: _e.mock.On("Register")}
}

func (_c *DataNodeNode_Register_Call) Run(run func()) *DataNodeNode_Register_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *DataNodeNode_Register_Call) Return(_a0 error) *DataNodeNode_Register_Call {
	_c.Call.Return(_a0)
	return _c
}

// ResendSegmentStats provides a mock function with given fields: ctx, req
func (_m *DataNodeNode) ResendSegmentStats(ctx context.Context, req *datapb.ResendSegmentStatsRequest) (*datapb.ResendSegmentStatsResponse, error) {
	ret := _m.Called(ctx, req)

	var r0 *datapb.ResendSegmentStatsResponse
	if rf, ok := ret.Get(0).(func(context.Context, *datapb.ResendSegmentStatsRequest) *datapb.ResendSegmentStatsResponse); ok {
		r0 = rf(ctx, req)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*datapb.ResendSegmentStatsResponse)
		}
	}

	var r1 error
	if rf, ok := ret.Get(1).(func(context.Context, *datapb.ResendSegmentStatsRequest) error); ok {
		r1 = rf(ctx, req)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// DataNodeNode_ResendSegmentStats_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'ResendSegmentStats'
type DataNodeNode_ResendSegmentStats_Call struct {
	*mock.Call
}

// ResendSegmentStats is a helper method to define mock.On call
//   - ctx context.Context
//   - req *datapb.ResendSegmentStatsRequest
func (_e *DataNodeNode_Expecter) ResendSegmentStats(ctx interface{}, req interface{}) *DataNodeNode_ResendSegmentStats_Call {
	return &DataNodeNode_ResendSegmentStats_Call{Call: _e.mock.On("ResendSegmentStats", ctx, req)}
}

func (_c *DataNodeNode_ResendSegmentStats_Call) Run(run func(ctx context.Context, req *datapb.ResendSegmentStatsRequest)) *DataNodeNode_ResendSegmentStats_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(*datapb.ResendSegmentStatsRequest))
	})
	return _c
}

func (_c *DataNodeNode_ResendSegmentStats_Call) Return(_a0 *datapb.ResendSegmentStatsResponse, _a1 error) *DataNodeNode_ResendSegmentStats_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

// SetAddress provides a mock function with given fields: address
func (_m *DataNodeNode) SetAddress(address string) {
	_m.Called(address)
}

// DataNodeNode_SetAddress_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'SetAddress'
type DataNodeNode_SetAddress_Call struct {
	*mock.Call
}

// SetAddress is a helper method to define mock.On call
//   - address string
func (_e *DataNodeNode_Expecter) SetAddress(address interface{}) *DataNodeNode_SetAddress_Call {
	return &DataNodeNode_SetAddress_Call{Call: _e.mock.On("SetAddress", address)}
}

func (_c *DataNodeNode_SetAddress_Call) Run(run func(address string)) *DataNodeNode_SetAddress_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(string))
	})
	return _c
}

func (_c *DataNodeNode_SetAddress_Call) Return() *DataNodeNode_SetAddress_Call {
	_c.Call.Return()
	return _c
}

// SetDataCoord provides a mock function with given fields: dataCoord
func (_m *DataNodeNode) SetDataCoord(dataCoord types.DataCoord) error {
	ret := _m.Called(dataCoord)

	var r0 error
	if rf, ok := ret.Get(0).(func(types.DataCoord) error); ok {
		r0 = rf(dataCoord)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// DataNodeNode_SetDataCoord_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'SetDataCoord'
type DataNodeNode_SetDataCoord_Call struct {
	*mock.Call
}

// SetDataCoord is a helper method to define mock.On call
//   - dataCoord types.DataCoord
func (_e *DataNodeNode_Expecter) SetDataCoord(dataCoord interface{}) *DataNodeNode_SetDataCoord_Call {
	return &DataNodeNode_SetDataCoord_Call{Call: _e.mock.On("SetDataCoord", dataCoord)}
}

func (_c *DataNodeNode_SetDataCoord_Call) Run(run func(dataCoord types.DataCoord)) *DataNodeNode_SetDataCoord_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(types.DataCoord))
	})
	return _c
}

func (_c *DataNodeNode_SetDataCoord_Call) Return(_a0 error) *DataNodeNode_SetDataCoord_Call {
	_c.Call.Return(_a0)
	return _c
}

// SetEtcdClient provides a mock function with given fields: etcdClient
func (_m *DataNodeNode) SetEtcdClient(etcdClient *clientv3.Client) {
	_m.Called(etcdClient)
}

// DataNodeNode_SetEtcdClient_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'SetEtcdClient'
type DataNodeNode_SetEtcdClient_Call struct {
	*mock.Call
}

// SetEtcdClient is a helper method to define mock.On call
//   - etcdClient *clientv3.Client
func (_e *DataNodeNode_Expecter) SetEtcdClient(etcdClient interface{}) *DataNodeNode_SetEtcdClient_Call {
	return &DataNodeNode_SetEtcdClient_Call{Call: _e.mock.On("SetEtcdClient", etcdClient)}
}

func (_c *DataNodeNode_SetEtcdClient_Call) Run(run func(etcdClient *clientv3.Client)) *DataNodeNode_SetEtcdClient_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(*clientv3.Client))
	})
	return _c
}

func (_c *DataNodeNode_SetEtcdClient_Call) Return() *DataNodeNode_SetEtcdClient_Call {
	_c.Call.Return()
	return _c
}

// SetRootCoord provides a mock function with given fields: rootCoord
func (_m *DataNodeNode) SetRootCoord(rootCoord types.RootCoord) error {
	ret := _m.Called(rootCoord)

	var r0 error
	if rf, ok := ret.Get(0).(func(types.RootCoord) error); ok {
		r0 = rf(rootCoord)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// DataNodeNode_SetRootCoord_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'SetRootCoord'
type DataNodeNode_SetRootCoord_Call struct {
	*mock.Call
}

// SetRootCoord is a helper method to define mock.On call
//   - rootCoord types.RootCoord
func (_e *DataNodeNode_Expecter) SetRootCoord(rootCoord interface{}) *DataNodeNode_SetRootCoord_Call {
	return &DataNodeNode_SetRootCoord_Call{Call: _e.mock.On("SetRootCoord", rootCoord)}
}

func (_c *DataNodeNode_SetRootCoord_Call) Run(run func(rootCoord types.RootCoord)) *DataNodeNode_SetRootCoord_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(types.RootCoord))
	})
	return _c
}

func (_c *DataNodeNode_SetRootCoord_Call) Return(_a0 error) *DataNodeNode_SetRootCoord_Call {
	_c.Call.Return(_a0)
	return _c
}

// ShowConfigurations provides a mock function with given fields: ctx, req
func (_m *DataNodeNode) ShowConfigurations(ctx context.Context, req *internalpb.ShowConfigurationsRequest) (*internalpb.ShowConfigurationsResponse, error) {
	ret := _m.Called(ctx, req)

	var r0 *internalpb.ShowConfigurationsResponse
	if rf, ok := ret.Get(0).(func(context.Context, *internalpb.ShowConfigurationsRequest) *internalpb.ShowConfigurationsResponse); ok {
		r0 = rf(ctx, req)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*internalpb.ShowConfigurationsResponse)
		}
	}

	var r1 error
	if rf, ok := ret.Get(1).(func(context.Context, *internalpb.ShowConfigurationsRequest) error); ok {
		r1 = rf(ctx, req)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// DataNodeNode_ShowConfigurations_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'ShowConfigurations'
type DataNodeNode_ShowConfigurations_Call struct {
	*mock.Call
}

// ShowConfigurations is a helper method to define mock.On call
//   - ctx context.Context
//   - req *internalpb.ShowConfigurationsRequest
func (_e *DataNodeNode_Expecter) ShowConfigurations(ctx interface{}, req interface{}) *DataNodeNode_ShowConfigurations_Call {
	return &DataNodeNode_ShowConfigurations_Call{Call: _e.mock.On("ShowConfigurations", ctx, req)}
}

func (_c *DataNodeNode_ShowConfigurations_Call) Run(run func(ctx context.Context, req *internalpb.ShowConfigurationsRequest)) *DataNodeNode_ShowConfigurations_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(*internalpb.ShowConfigurationsRequest))
	})
	return _c
}

func (_c *DataNodeNode_ShowConfigurations_Call) Return(_a0 *internalpb.ShowConfigurationsResponse, _a1 error) *DataNodeNode_ShowConfigurations_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

// Start provides a mock function with given fields:
func (_m *DataNodeNode) Start() error {
	ret := _m.Called()

	var r0 error
	if rf, ok := ret.Get(0).(func() error); ok {
		r0 = rf()
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// DataNodeNode_Start_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Start'
type DataNodeNode_Start_Call struct {
	*mock.Call
}

// Start is a helper method to define mock.On call
func (_e *DataNodeNode_Expecter) Start() *DataNodeNode_Start_Call {
	return &DataNodeNode_Start_Call{Call: _e.mock.On("Start")}
}

func (_c *DataNodeNode_Start_Call) Run(run func()) *DataNodeNode_Start_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *DataNodeNode_Start_Call) Return(_a0 error) *DataNodeNode_Start_Call {
	_c.Call.Return(_a0)
	return _c
}

// Stop provides a mock function with given fields:
func (_m *DataNodeNode) Stop() error {
	ret := _m.Called()

	var r0 error
	if rf, ok := ret.Get(0).(func() error); ok {
		r0 = rf()
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// DataNodeNode_Stop_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Stop'
type DataNodeNode_Stop_Call struct {
	*mock.Call
}

// Stop is a helper method to define mock.On call
func (_e *DataNodeNode_Expecter) Stop() *DataNodeNode_Stop_Call {
	return &DataNodeNode_Stop_Call{Call: _e.mock.On("Stop")}
}

func (_c *DataNodeNode_Stop_Call) Run(run func()) *DataNodeNode_Stop_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *DataNodeNode_Stop_Call) Return(_a0 error) *DataNodeNode_Stop_Call {
	_c.Call.Return(_a0)
	return _c
}

// SyncSegments provides a mock function with given fields: ctx, req
func (_m *DataNodeNode) SyncSegments(ctx context.Context, req *datapb.SyncSegmentsRequest) (*commonpb.Status, error) {
	ret := _m.Called(ctx, req)

	var r0 *commonpb.Status
	if rf, ok := ret.Get(0).(func(context.Context, *datapb.SyncSegmentsRequest) *commonpb.Status); ok {
		r0 = rf(ctx, req)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*commonpb.Status)
		}
	}

	var r1 error
	if rf, ok := ret.Get(1).(func(context.Context, *datapb.SyncSegmentsRequest) error); ok {
		r1 = rf(ctx, req)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// DataNodeNode_SyncSegments_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'SyncSegments'
type DataNodeNode_SyncSegments_Call struct {
	*mock.Call
}

// SyncSegments is a helper method to define mock.On call
//   - ctx context.Context
//   - req *datapb.SyncSegmentsRequest
func (_e *DataNodeNode_Expecter) SyncSegments(ctx interface{}, req interface{}) *DataNodeNode_SyncSegments_Call {
	return &DataNodeNode_SyncSegments_Call{Call: _e.mock.On("SyncSegments", ctx, req)}
}

func (_c *DataNodeNode_SyncSegments_Call) Run(run func(ctx context.Context, req *datapb.SyncSegmentsRequest)) *DataNodeNode_SyncSegments_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(*datapb.SyncSegmentsRequest))
	})
	return _c
}

func (_c *DataNodeNode_SyncSegments_Call) Return(_a0 *commonpb.Status, _a1 error) *DataNodeNode_SyncSegments_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

// UpdateStateCode provides a mock function with given fields: stateCode
func (_m *DataNodeNode) UpdateStateCode(stateCode commonpb.StateCode) {
	_m.Called(stateCode)
}

// DataNodeNode_UpdateStateCode_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'UpdateStateCode'
type DataNodeNode_UpdateStateCode_Call struct {
	*mock.Call
}

// UpdateStateCode is a helper method to define mock.On call
//   - stateCode commonpb.StateCode
func (_e *DataNodeNode_Expecter) UpdateStateCode(stateCode interface{}) *DataNodeNode_UpdateStateCode_Call {
	return &DataNodeNode_UpdateStateCode_Call{Call: _e.mock.On("UpdateStateCode", stateCode)}
}

func (_c *DataNodeNode_UpdateStateCode_Call) Run(run func(stateCode commonpb.StateCode)) *DataNodeNode_UpdateStateCode_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(commonpb.StateCode))
	})
	return _c
}

func (_c *DataNodeNode_UpdateStateCode_Call) Return() *DataNodeNode_UpdateStateCode_Call {
	_c.Call.Return()
	return _c
}

// WatchDmChannels provides a mock function with given fields: ctx, req
func (_m *DataNodeNode) WatchDmChannels(ctx context.Context, req *datapb.WatchDmChannelsRequest) (*commonpb.Status, error) {
	ret := _m.Called(ctx, req)

	var r0 *commonpb.Status
	if rf, ok := ret.Get(0).(func(context.Context, *datapb.WatchDmChannelsRequest) *commonpb.Status); ok {
		r0 = rf(ctx, req)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*commonpb.Status)
		}
	}

	var r1 error
	if rf, ok := ret.Get(1).(func(context.Context, *datapb.WatchDmChannelsRequest) error); ok {
		r1 = rf(ctx, req)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// DataNodeNode_WatchDmChannels_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'WatchDmChannels'
type DataNodeNode_WatchDmChannels_Call struct {
	*mock.Call
}

// WatchDmChannels is a helper method to define mock.On call
//   - ctx context.Context
//   - req *datapb.WatchDmChannelsRequest
func (_e *DataNodeNode_Expecter) WatchDmChannels(ctx interface{}, req interface{}) *DataNodeNode_WatchDmChannels_Call {
	return &DataNodeNode_WatchDmChannels_Call{Call: _e.mock.On("WatchDmChannels", ctx, req)}
}

func (_c *DataNodeNode_WatchDmChannels_Call) Run(run func(ctx context.Context, req *datapb.WatchDmChannelsRequest)) *DataNodeNode_WatchDmChannels_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(*datapb.WatchDmChannelsRequest))
	})
	return _c
}

func (_c *DataNodeNode_WatchDmChannels_Call) Return(_a0 *commonpb.Status, _a1 error) *DataNodeNode_WatchDmChannels_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

type mockConstructorTestingTNewDataNodeNode interface {
	mock.TestingT
	Cleanup(func())
}

// NewDataNodeNode creates a new instance of DataNodeNode. It also registers a testing interface on the mock and a cleanup function to assert the mocks expectations.
func NewDataNodeNode(t mockConstructorTestingTNewDataNodeNode) *DataNodeNode {
	mock := &DataNodeNode{}
	mock.Mock.Test(t)

	t.Cleanup(func() { mock.AssertExpectations(t) })

	return mock
}
