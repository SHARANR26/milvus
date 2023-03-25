// Code generated by mockery v2.16.0. DO NOT EDIT.

package balance

import (
	meta "github.com/milvus-io/milvus/internal/querycoordv2/meta"
	mock "github.com/stretchr/testify/mock"
)

// MockBalancer is an autogenerated mock type for the Balance type
type MockBalancer struct {
	mock.Mock
}

type MockBalancer_Expecter struct {
	mock *mock.Mock
}

func (_m *MockBalancer) EXPECT() *MockBalancer_Expecter {
	return &MockBalancer_Expecter{mock: &_m.Mock}
}

// AssignChannel provides a mock function with given fields: channels, nodes
func (_m *MockBalancer) AssignChannel(channels []*meta.DmChannel, nodes []int64) []ChannelAssignPlan {
	ret := _m.Called(channels, nodes)

	var r0 []ChannelAssignPlan
	if rf, ok := ret.Get(0).(func([]*meta.DmChannel, []int64) []ChannelAssignPlan); ok {
		r0 = rf(channels, nodes)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).([]ChannelAssignPlan)
		}
	}

	return r0
}

// MockBalancer_AssignChannel_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'AssignChannel'
type MockBalancer_AssignChannel_Call struct {
	*mock.Call
}

// AssignChannel is a helper method to define mock.On call
//  - channels []*meta.DmChannel
//  - nodes []int64
func (_e *MockBalancer_Expecter) AssignChannel(channels interface{}, nodes interface{}) *MockBalancer_AssignChannel_Call {
	return &MockBalancer_AssignChannel_Call{Call: _e.mock.On("AssignChannel", channels, nodes)}
}

func (_c *MockBalancer_AssignChannel_Call) Run(run func(channels []*meta.DmChannel, nodes []int64)) *MockBalancer_AssignChannel_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].([]*meta.DmChannel), args[1].([]int64))
	})
	return _c
}

func (_c *MockBalancer_AssignChannel_Call) Return(_a0 []ChannelAssignPlan) *MockBalancer_AssignChannel_Call {
	_c.Call.Return(_a0)
	return _c
}

// AssignSegment provides a mock function with given fields: collectionID, segments, nodes
func (_m *MockBalancer) AssignSegment(collectionID int64, segments []*meta.Segment, nodes []int64) []SegmentAssignPlan {
	ret := _m.Called(collectionID, segments, nodes)

	var r0 []SegmentAssignPlan
	if rf, ok := ret.Get(0).(func(int64, []*meta.Segment, []int64) []SegmentAssignPlan); ok {
		r0 = rf(collectionID, segments, nodes)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).([]SegmentAssignPlan)
		}
	}

	return r0
}

// MockBalancer_AssignSegment_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'AssignSegment'
type MockBalancer_AssignSegment_Call struct {
	*mock.Call
}

// AssignSegment is a helper method to define mock.On call
//   - collectionID int64
//   - segments []*meta.Segment
//   - nodes []int64
func (_e *MockBalancer_Expecter) AssignSegment(collectionID interface{}, segments interface{}, nodes interface{}) *MockBalancer_AssignSegment_Call {
	return &MockBalancer_AssignSegment_Call{Call: _e.mock.On("AssignSegment", collectionID, segments, nodes)}
}

func (_c *MockBalancer_AssignSegment_Call) Run(run func(collectionID int64, segments []*meta.Segment, nodes []int64)) *MockBalancer_AssignSegment_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(int64), args[1].([]*meta.Segment), args[2].([]int64))
	})
	return _c
}

func (_c *MockBalancer_AssignSegment_Call) Return(_a0 []SegmentAssignPlan) *MockBalancer_AssignSegment_Call {
	_c.Call.Return(_a0)
	return _c
}

// Balance provides a mock function with given fields:
func (_m *MockBalancer) Balance() ([]SegmentAssignPlan, []ChannelAssignPlan) {
	ret := _m.Called()

	var r0 []SegmentAssignPlan
	if rf, ok := ret.Get(0).(func() []SegmentAssignPlan); ok {
		r0 = rf()
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).([]SegmentAssignPlan)
		}
	}

	var r1 []ChannelAssignPlan
	if rf, ok := ret.Get(1).(func() []ChannelAssignPlan); ok {
		r1 = rf()
	} else {
		if ret.Get(1) != nil {
			r1 = ret.Get(1).([]ChannelAssignPlan)
		}
	}

	return r0, r1
}

// MockBalancer_Balance_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Balance'
type MockBalancer_Balance_Call struct {
	*mock.Call
}

// Balance is a helper method to define mock.On call
func (_e *MockBalancer_Expecter) Balance() *MockBalancer_Balance_Call {
	return &MockBalancer_Balance_Call{Call: _e.mock.On("Balance")}
}

func (_c *MockBalancer_Balance_Call) Run(run func()) *MockBalancer_Balance_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockBalancer_Balance_Call) Return(_a0 []SegmentAssignPlan, _a1 []ChannelAssignPlan) *MockBalancer_Balance_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

type mockConstructorTestingTNewMockBalancer interface {
	mock.TestingT
	Cleanup(func())
}

// NewMockBalancer creates a new instance of MockBalancer. It also registers a testing interface on the mock and a cleanup function to assert the mocks expectations.
func NewMockBalancer(t mockConstructorTestingTNewMockBalancer) *MockBalancer {
	mock := &MockBalancer{}
	mock.Mock.Test(t)

	t.Cleanup(func() { mock.AssertExpectations(t) })

	return mock
}
