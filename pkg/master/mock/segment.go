package mock

import (
	"bytes"
	"encoding/gob"
	masterpb "github.com/czs007/suvlim/pkg/master/grpc/master"
	"github.com/golang/protobuf/proto"
)

type SegmentStats struct {
	SegementID uint64
	MemorySize uint64
	MemoryRate float64
	Status     masterpb.SegmentStatus
	Rows       int64
}

// map[SegmentID]SegmentCloseTime
type SegmentCloseLog map[uint64]uint64

func SegmentMarshal(s SegmentStats) ([]byte, error) {
	var nb bytes.Buffer
	enc := gob.NewEncoder(&nb)
	err := enc.Encode(s)
	if err != nil {
		return []byte{}, err
	}
	return nb.Bytes(), nil
}

func SegmentUnMarshal(data []byte) (SegmentStats, error) {
	var pbSS masterpb.SegmentStat
	err := proto.Unmarshal(data, &pbSS)
	if err != nil {
		return SegmentStats{}, err
	}
	var ss = SegmentStats{
		SegementID: pbSS.SegmentId,
		MemorySize: pbSS.MemorySize,
		MemoryRate: float64(pbSS.MemoryRate),
		Status:     pbSS.Status,
		Rows:       pbSS.Rows,
	}
	return ss, nil
}

type Segment struct {
	SegmentID      uint64                 `json:"segment_id"`
	CollectionID   uint64                 `json:"collection_id"`
	PartitionTag   string                 `json:"partition_tag"`
	ChannelStart   int                    `json:"channel_start"`
	ChannelEnd     int                    `json:"channel_end"`
	OpenTimeStamp  uint64                 `json:"open_timestamp"`
	CloseTimeStamp uint64                 `json:"close_timestamp"`
	CollectionName string                 `json:"collection_name"`
	Status         masterpb.SegmentStatus `json:"segment_status"`
	Rows           int64                  `json:"rows"`
}

func NewSegment(id uint64, collectioID uint64, cName string, ptag string, chStart int, chEnd int, openTime uint64, closeTime uint64) Segment {
	return Segment{
		SegmentID:      id,
		CollectionID:   collectioID,
		CollectionName: cName,
		PartitionTag:   ptag,
		ChannelStart:   chStart,
		ChannelEnd:     chEnd,
		OpenTimeStamp:  openTime,
		CloseTimeStamp: closeTime,
	}
}
func Segment2JSON(s Segment) (string, error) {
	b, err := json.Marshal(&s)
	if err != nil {
		return "", err
	}

	return string(b), nil
}

func JSON2Segment(s string) (*Segment, error) {
	var c Segment
	err := json.Unmarshal([]byte(s), &c)
	if err != nil {
		return &Segment{}, err
	}
	return &c, nil
}
