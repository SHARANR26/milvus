// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

package rootcoord

import (
	"fmt"
	"math"
	"sync"

	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/metrics"
	"github.com/milvus-io/milvus/internal/msgstream"
	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/util/sessionutil"
	"github.com/milvus-io/milvus/internal/util/timerecord"
	"github.com/milvus-io/milvus/internal/util/tsoutil"
	"github.com/milvus-io/milvus/internal/util/typeutil"
	"go.uber.org/zap"
)

type timetickSync struct {
	core          *Core
	lock          sync.Mutex
	proxyTimeTick map[typeutil.UniqueID]*channelTimeTickMsg
	sendChan      chan map[typeutil.UniqueID]*channelTimeTickMsg

	// record ddl timetick info
	ddlLock  sync.RWMutex
	ddlMinTs typeutil.Timestamp
	ddlTsSet map[typeutil.Timestamp]struct{}
}

type channelTimeTickMsg struct {
	in       *internalpb.ChannelTimeTickMsg
	timeTick map[string]typeutil.Timestamp
}

func newChannelTimeTickMsg(in *internalpb.ChannelTimeTickMsg) *channelTimeTickMsg {
	msg := &channelTimeTickMsg{
		in:       in,
		timeTick: make(map[string]typeutil.Timestamp),
	}
	for idx := range in.ChannelNames {
		msg.timeTick[in.ChannelNames[idx]] = in.Timestamps[idx]
	}
	return msg
}

func (c *channelTimeTickMsg) getTimetick(channelName string) typeutil.Timestamp {
	tt, ok := c.timeTick[channelName]
	if ok {
		return tt
	}
	return c.in.DefaultTimestamp
}

func newTimeTickSync(core *Core) *timetickSync {
	return &timetickSync{
		lock:          sync.Mutex{},
		core:          core,
		proxyTimeTick: make(map[typeutil.UniqueID]*channelTimeTickMsg),
		sendChan:      make(chan map[typeutil.UniqueID]*channelTimeTickMsg, 16),

		ddlLock:  sync.RWMutex{},
		ddlMinTs: typeutil.Timestamp(math.MaxUint64),
		ddlTsSet: make(map[typeutil.Timestamp]struct{}),
	}
}

// sendToChannel send all channels' timetick to sendChan
// lock is needed by the invoker
func (t *timetickSync) sendToChannel() {
	if len(t.proxyTimeTick) == 0 {
		return
	}
	for _, v := range t.proxyTimeTick {
		if v == nil {
			return
		}
	}
	// clear proxyTimeTick and send a clone
	ptt := make(map[typeutil.UniqueID]*channelTimeTickMsg)
	for k, v := range t.proxyTimeTick {
		ptt[k] = v
		t.proxyTimeTick[k] = nil
	}
	t.sendChan <- ptt
}

// AddDmlTimeTick add ts into ddlTimetickInfos[sourceID],
// can be used to tell if DDL operation is in process.
func (t *timetickSync) AddDdlTimeTick(ts typeutil.Timestamp, reason string) {
	t.ddlLock.Lock()
	defer t.ddlLock.Unlock()

	if ts < t.ddlMinTs {
		t.ddlMinTs = ts
	}
	t.ddlTsSet[ts] = struct{}{}

	log.Debug("add ddl timetick", zap.Uint64("minTs", t.ddlMinTs), zap.Uint64("ts", ts),
		zap.Int("len(ddlTsSet)", len(t.ddlTsSet)), zap.String("reason", reason))
}

// RemoveDdlTimeTick is invoked in UpdateTimeTick.
// It clears the ts generated by AddDdlTimeTick, indicates DDL operation finished.
func (t *timetickSync) RemoveDdlTimeTick(ts typeutil.Timestamp, reason string) {
	t.ddlLock.Lock()
	defer t.ddlLock.Unlock()

	delete(t.ddlTsSet, ts)
	log.Debug("remove ddl timetick", zap.Uint64("ts", ts), zap.Int("len(ddlTsSet)", len(t.ddlTsSet)),
		zap.String("reason", reason))
	if len(t.ddlTsSet) == 0 {
		t.ddlMinTs = typeutil.Timestamp(math.MaxUint64)
	} else if t.ddlMinTs == ts {
		// re-calculate minTs
		minTs := typeutil.Timestamp(math.MaxUint64)
		for tt := range t.ddlTsSet {
			if tt < minTs {
				minTs = tt
			}
		}
		t.ddlMinTs = minTs
		log.Debug("update ddl minTs", zap.Any("minTs", minTs))
	}
}

func (t *timetickSync) GetDdlMinTimeTick() typeutil.Timestamp {
	t.ddlLock.Lock()
	defer t.ddlLock.Unlock()

	return t.ddlMinTs
}

// UpdateTimeTick check msg validation and send it to local channel
func (t *timetickSync) UpdateTimeTick(in *internalpb.ChannelTimeTickMsg, reason string) error {
	t.lock.Lock()
	defer t.lock.Unlock()
	if len(in.ChannelNames) == 0 && in.DefaultTimestamp == 0 {
		return nil
	}
	if len(in.Timestamps) != len(in.ChannelNames) {
		return fmt.Errorf("invalid TimeTickMsg")
	}

	prev, ok := t.proxyTimeTick[in.Base.SourceID]
	if !ok {
		return fmt.Errorf("skip ChannelTimeTickMsg from un-recognized proxy node %d", in.Base.SourceID)
	}

	// if ddl operation not finished, skip current ts update
	ddlMinTs := t.GetDdlMinTimeTick()
	if in.DefaultTimestamp > ddlMinTs {
		log.Debug("ddl not finished", zap.Int64("source id", in.Base.SourceID),
			zap.Uint64("curr ts", in.DefaultTimestamp),
			zap.Uint64("ddlMinTs", ddlMinTs),
			zap.String("reason", reason))
		return nil
	}

	if in.Base.SourceID == t.core.session.ServerID {
		if prev != nil && in.DefaultTimestamp <= prev.in.DefaultTimestamp {
			log.Debug("timestamp go back", zap.Int64("source id", in.Base.SourceID),
				zap.Uint64("curr ts", in.DefaultTimestamp),
				zap.Uint64("prev ts", prev.in.DefaultTimestamp),
				zap.String("reason", reason))
			return nil
		}
	}
	if in.DefaultTimestamp == 0 {
		mints := minTimeTick(in.Timestamps...)
		log.Debug("default time stamp is zero, set it to the min value of inputs",
			zap.Int64("proxy id", in.Base.SourceID), zap.Uint64("min ts", mints))
		in.DefaultTimestamp = mints
	}

	t.proxyTimeTick[in.Base.SourceID] = newChannelTimeTickMsg(in)
	//log.Debug("update proxyTimeTick", zap.Int64("source id", in.Base.SourceID),
	//	zap.Any("Ts", in.Timestamps), zap.Uint64("inTs", in.DefaultTimestamp), zap.String("reason", reason))

	t.sendToChannel()
	return nil
}

func (t *timetickSync) AddProxy(sess *sessionutil.Session) {
	t.lock.Lock()
	defer t.lock.Unlock()
	t.proxyTimeTick[sess.ServerID] = nil
	log.Debug("Add proxy for timeticksync", zap.Int64("serverID", sess.ServerID))
}

func (t *timetickSync) DelProxy(sess *sessionutil.Session) {
	t.lock.Lock()
	defer t.lock.Unlock()
	if _, ok := t.proxyTimeTick[sess.ServerID]; ok {
		delete(t.proxyTimeTick, sess.ServerID)
		log.Debug("Remove proxy from timeticksync", zap.Int64("serverID", sess.ServerID))
		t.sendToChannel()
	}
}

func (t *timetickSync) GetProxy(sess []*sessionutil.Session) {
	t.lock.Lock()
	defer t.lock.Unlock()
	for _, s := range sess {
		t.proxyTimeTick[s.ServerID] = nil
	}
}

// StartWatch watch proxy node change and process all channels' timetick msg
func (t *timetickSync) StartWatch(wg *sync.WaitGroup) {
	defer wg.Done()
	for {
		select {
		case <-t.core.ctx.Done():
			log.Debug("rootcoord context done", zap.Error(t.core.ctx.Err()))
			return
		case proxyTimetick, ok := <-t.sendChan:
			if !ok {
				log.Debug("timetickSync sendChan closed")
				return
			}

			// reduce each channel to get min timestamp
			local := proxyTimetick[t.core.session.ServerID]
			if len(local.in.ChannelNames) == 0 {
				continue
			}

			hdr := fmt.Sprintf("send ts to %d channels", len(local.in.ChannelNames))
			tr := timerecord.NewTimeRecorder(hdr)
			wg := sync.WaitGroup{}
			for _, chanName := range local.in.ChannelNames {
				wg.Add(1)
				go func(chanName string) {
					mints := local.getTimetick(chanName)
					for _, tt := range proxyTimetick {
						ts := tt.getTimetick(chanName)
						if ts < mints {
							mints = ts
						}
					}
					if err := t.SendTimeTickToChannel([]string{chanName}, mints); err != nil {
						log.Debug("SendTimeTickToChannel fail", zap.Error(err), zap.String("channel", chanName))
					}
					wg.Done()
				}(chanName)
			}
			wg.Wait()
			span := tr.ElapseSpan()
			// rootcoord send tt msg to all channels every 200ms by default
			if span.Milliseconds() > 200 {
				log.Warn("rootcoord send tt to all channels too slowly",
					zap.Int("chanNum", len(local.in.ChannelNames)),
					zap.Int64("span", span.Milliseconds()))
			}
		}
	}
}

// SendTimeTickToChannel send each channel's min timetick to msg stream
func (t *timetickSync) SendTimeTickToChannel(chanNames []string, ts typeutil.Timestamp) error {
	msgPack := msgstream.MsgPack{}
	baseMsg := msgstream.BaseMsg{
		BeginTimestamp: ts,
		EndTimestamp:   ts,
		HashValues:     []uint32{0},
	}
	timeTickResult := internalpb.TimeTickMsg{
		Base: &commonpb.MsgBase{
			MsgType:   commonpb.MsgType_TimeTick,
			MsgID:     0,
			Timestamp: ts,
			SourceID:  t.core.session.ServerID,
		},
	}
	timeTickMsg := &msgstream.TimeTickMsg{
		BaseMsg:     baseMsg,
		TimeTickMsg: timeTickResult,
	}
	msgPack.Msgs = append(msgPack.Msgs, timeTickMsg)

	if err := t.core.dmlChannels.Broadcast(chanNames, &msgPack); err != nil {
		return err
	}

	for _, chanName := range chanNames {
		metrics.RootCoordInsertChannelTimeTick.WithLabelValues(chanName).Set(float64(tsoutil.Mod24H(ts)))
	}
	return nil
}

// GetProxyNum return the num of detected proxy node
func (t *timetickSync) GetProxyNum() int {
	t.lock.Lock()
	defer t.lock.Unlock()
	return len(t.proxyTimeTick)
}

// GetChanNum return the num of channel
func (t *timetickSync) GetChanNum() int {
	return t.core.dmlChannels.GetNumChannels()
}

func minTimeTick(tt ...typeutil.Timestamp) typeutil.Timestamp {
	var ret typeutil.Timestamp
	for _, t := range tt {
		if ret == 0 {
			ret = t
		} else {
			if t < ret {
				ret = t
			}
		}
	}
	return ret
}
