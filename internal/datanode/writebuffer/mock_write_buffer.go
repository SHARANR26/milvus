// Code generated by mockery v2.32.4. DO NOT EDIT.

package writebuffer

import (
	context "context"

	msgpb "github.com/milvus-io/milvus-proto/go-api/v2/msgpb"
	mock "github.com/stretchr/testify/mock"

	msgstream "github.com/milvus-io/milvus/pkg/mq/msgstream"
)

// MockWriteBuffer is an autogenerated mock type for the WriteBuffer type
type MockWriteBuffer struct {
	mock.Mock
}

type MockWriteBuffer_Expecter struct {
	mock *mock.Mock
}

func (_m *MockWriteBuffer) EXPECT() *MockWriteBuffer_Expecter {
	return &MockWriteBuffer_Expecter{mock: &_m.Mock}
}

// BufferData provides a mock function with given fields: insertMsgs, deleteMsgs, startPos, endPos
func (_m *MockWriteBuffer) BufferData(insertMsgs []*msgstream.InsertMsg, deleteMsgs []*msgstream.DeleteMsg, startPos *msgpb.MsgPosition, endPos *msgpb.MsgPosition) error {
	ret := _m.Called(insertMsgs, deleteMsgs, startPos, endPos)

	var r0 error
	if rf, ok := ret.Get(0).(func([]*msgstream.InsertMsg, []*msgstream.DeleteMsg, *msgpb.MsgPosition, *msgpb.MsgPosition) error); ok {
		r0 = rf(insertMsgs, deleteMsgs, startPos, endPos)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockWriteBuffer_BufferData_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'BufferData'
type MockWriteBuffer_BufferData_Call struct {
	*mock.Call
}

// BufferData is a helper method to define mock.On call
//   - insertMsgs []*msgstream.InsertMsg
//   - deleteMsgs []*msgstream.DeleteMsg
//   - startPos *msgpb.MsgPosition
//   - endPos *msgpb.MsgPosition
func (_e *MockWriteBuffer_Expecter) BufferData(insertMsgs interface{}, deleteMsgs interface{}, startPos interface{}, endPos interface{}) *MockWriteBuffer_BufferData_Call {
	return &MockWriteBuffer_BufferData_Call{Call: _e.mock.On("BufferData", insertMsgs, deleteMsgs, startPos, endPos)}
}

func (_c *MockWriteBuffer_BufferData_Call) Run(run func(insertMsgs []*msgstream.InsertMsg, deleteMsgs []*msgstream.DeleteMsg, startPos *msgpb.MsgPosition, endPos *msgpb.MsgPosition)) *MockWriteBuffer_BufferData_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].([]*msgstream.InsertMsg), args[1].([]*msgstream.DeleteMsg), args[2].(*msgpb.MsgPosition), args[3].(*msgpb.MsgPosition))
	})
	return _c
}

func (_c *MockWriteBuffer_BufferData_Call) Return(_a0 error) *MockWriteBuffer_BufferData_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockWriteBuffer_BufferData_Call) RunAndReturn(run func([]*msgstream.InsertMsg, []*msgstream.DeleteMsg, *msgpb.MsgPosition, *msgpb.MsgPosition) error) *MockWriteBuffer_BufferData_Call {
	_c.Call.Return(run)
	return _c
}

// Close provides a mock function with given fields: drop
func (_m *MockWriteBuffer) Close(drop bool) {
	_m.Called(drop)
}

// MockWriteBuffer_Close_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Close'
type MockWriteBuffer_Close_Call struct {
	*mock.Call
}

// Close is a helper method to define mock.On call
//   - drop bool
func (_e *MockWriteBuffer_Expecter) Close(drop interface{}) *MockWriteBuffer_Close_Call {
	return &MockWriteBuffer_Close_Call{Call: _e.mock.On("Close", drop)}
}

func (_c *MockWriteBuffer_Close_Call) Run(run func(drop bool)) *MockWriteBuffer_Close_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(bool))
	})
	return _c
}

func (_c *MockWriteBuffer_Close_Call) Return() *MockWriteBuffer_Close_Call {
	_c.Call.Return()
	return _c
}

func (_c *MockWriteBuffer_Close_Call) RunAndReturn(run func(bool)) *MockWriteBuffer_Close_Call {
	_c.Call.Return(run)
	return _c
}

// FlushSegments provides a mock function with given fields: ctx, segmentIDs
func (_m *MockWriteBuffer) FlushSegments(ctx context.Context, segmentIDs []int64) error {
	ret := _m.Called(ctx, segmentIDs)

	var r0 error
	if rf, ok := ret.Get(0).(func(context.Context, []int64) error); ok {
		r0 = rf(ctx, segmentIDs)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockWriteBuffer_FlushSegments_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'FlushSegments'
type MockWriteBuffer_FlushSegments_Call struct {
	*mock.Call
}

// FlushSegments is a helper method to define mock.On call
//   - ctx context.Context
//   - segmentIDs []int64
func (_e *MockWriteBuffer_Expecter) FlushSegments(ctx interface{}, segmentIDs interface{}) *MockWriteBuffer_FlushSegments_Call {
	return &MockWriteBuffer_FlushSegments_Call{Call: _e.mock.On("FlushSegments", ctx, segmentIDs)}
}

func (_c *MockWriteBuffer_FlushSegments_Call) Run(run func(ctx context.Context, segmentIDs []int64)) *MockWriteBuffer_FlushSegments_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].([]int64))
	})
	return _c
}

func (_c *MockWriteBuffer_FlushSegments_Call) Return(_a0 error) *MockWriteBuffer_FlushSegments_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockWriteBuffer_FlushSegments_Call) RunAndReturn(run func(context.Context, []int64) error) *MockWriteBuffer_FlushSegments_Call {
	_c.Call.Return(run)
	return _c
}

// GetFlushTimestamp provides a mock function with given fields:
func (_m *MockWriteBuffer) GetFlushTimestamp() uint64 {
	ret := _m.Called()

	var r0 uint64
	if rf, ok := ret.Get(0).(func() uint64); ok {
		r0 = rf()
	} else {
		r0 = ret.Get(0).(uint64)
	}

	return r0
}

// MockWriteBuffer_GetFlushTimestamp_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetFlushTimestamp'
type MockWriteBuffer_GetFlushTimestamp_Call struct {
	*mock.Call
}

// GetFlushTimestamp is a helper method to define mock.On call
func (_e *MockWriteBuffer_Expecter) GetFlushTimestamp() *MockWriteBuffer_GetFlushTimestamp_Call {
	return &MockWriteBuffer_GetFlushTimestamp_Call{Call: _e.mock.On("GetFlushTimestamp")}
}

func (_c *MockWriteBuffer_GetFlushTimestamp_Call) Run(run func()) *MockWriteBuffer_GetFlushTimestamp_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockWriteBuffer_GetFlushTimestamp_Call) Return(_a0 uint64) *MockWriteBuffer_GetFlushTimestamp_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockWriteBuffer_GetFlushTimestamp_Call) RunAndReturn(run func() uint64) *MockWriteBuffer_GetFlushTimestamp_Call {
	_c.Call.Return(run)
	return _c
}

// HasSegment provides a mock function with given fields: segmentID
func (_m *MockWriteBuffer) HasSegment(segmentID int64) bool {
	ret := _m.Called(segmentID)

	var r0 bool
	if rf, ok := ret.Get(0).(func(int64) bool); ok {
		r0 = rf(segmentID)
	} else {
		r0 = ret.Get(0).(bool)
	}

	return r0
}

// MockWriteBuffer_HasSegment_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'HasSegment'
type MockWriteBuffer_HasSegment_Call struct {
	*mock.Call
}

// HasSegment is a helper method to define mock.On call
//   - segmentID int64
func (_e *MockWriteBuffer_Expecter) HasSegment(segmentID interface{}) *MockWriteBuffer_HasSegment_Call {
	return &MockWriteBuffer_HasSegment_Call{Call: _e.mock.On("HasSegment", segmentID)}
}

func (_c *MockWriteBuffer_HasSegment_Call) Run(run func(segmentID int64)) *MockWriteBuffer_HasSegment_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(int64))
	})
	return _c
}

func (_c *MockWriteBuffer_HasSegment_Call) Return(_a0 bool) *MockWriteBuffer_HasSegment_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockWriteBuffer_HasSegment_Call) RunAndReturn(run func(int64) bool) *MockWriteBuffer_HasSegment_Call {
	_c.Call.Return(run)
	return _c
}

// MinCheckpoint provides a mock function with given fields:
func (_m *MockWriteBuffer) MinCheckpoint() *msgpb.MsgPosition {
	ret := _m.Called()

	var r0 *msgpb.MsgPosition
	if rf, ok := ret.Get(0).(func() *msgpb.MsgPosition); ok {
		r0 = rf()
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*msgpb.MsgPosition)
		}
	}

	return r0
}

// MockWriteBuffer_MinCheckpoint_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'MinCheckpoint'
type MockWriteBuffer_MinCheckpoint_Call struct {
	*mock.Call
}

// MinCheckpoint is a helper method to define mock.On call
func (_e *MockWriteBuffer_Expecter) MinCheckpoint() *MockWriteBuffer_MinCheckpoint_Call {
	return &MockWriteBuffer_MinCheckpoint_Call{Call: _e.mock.On("MinCheckpoint")}
}

func (_c *MockWriteBuffer_MinCheckpoint_Call) Run(run func()) *MockWriteBuffer_MinCheckpoint_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockWriteBuffer_MinCheckpoint_Call) Return(_a0 *msgpb.MsgPosition) *MockWriteBuffer_MinCheckpoint_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockWriteBuffer_MinCheckpoint_Call) RunAndReturn(run func() *msgpb.MsgPosition) *MockWriteBuffer_MinCheckpoint_Call {
	_c.Call.Return(run)
	return _c
}

// SetFlushTimestamp provides a mock function with given fields: flushTs
func (_m *MockWriteBuffer) SetFlushTimestamp(flushTs uint64) {
	_m.Called(flushTs)
}

// MockWriteBuffer_SetFlushTimestamp_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'SetFlushTimestamp'
type MockWriteBuffer_SetFlushTimestamp_Call struct {
	*mock.Call
}

// SetFlushTimestamp is a helper method to define mock.On call
//   - flushTs uint64
func (_e *MockWriteBuffer_Expecter) SetFlushTimestamp(flushTs interface{}) *MockWriteBuffer_SetFlushTimestamp_Call {
	return &MockWriteBuffer_SetFlushTimestamp_Call{Call: _e.mock.On("SetFlushTimestamp", flushTs)}
}

func (_c *MockWriteBuffer_SetFlushTimestamp_Call) Run(run func(flushTs uint64)) *MockWriteBuffer_SetFlushTimestamp_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(uint64))
	})
	return _c
}

func (_c *MockWriteBuffer_SetFlushTimestamp_Call) Return() *MockWriteBuffer_SetFlushTimestamp_Call {
	_c.Call.Return()
	return _c
}

func (_c *MockWriteBuffer_SetFlushTimestamp_Call) RunAndReturn(run func(uint64)) *MockWriteBuffer_SetFlushTimestamp_Call {
	_c.Call.Return(run)
	return _c
}

// NewMockWriteBuffer creates a new instance of MockWriteBuffer. It also registers a testing interface on the mock and a cleanup function to assert the mocks expectations.
// The first argument is typically a *testing.T value.
func NewMockWriteBuffer(t interface {
	mock.TestingT
	Cleanup(func())
}) *MockWriteBuffer {
	mock := &MockWriteBuffer{}
	mock.Mock.Test(t)

	t.Cleanup(func() { mock.AssertExpectations(t) })

	return mock
}
