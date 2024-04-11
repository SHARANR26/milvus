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

package analyzecgowrapper

/*
#cgo pkg-config: milvus_clustering

#include <stdlib.h>	// free
#include "clustering/analyze_c.h"
*/
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/util/merr"
)

func GetBinarySetKeys(cBinarySet C.CBinarySet) ([]string, error) {
	size := int(C.GetBinarySetSize(cBinarySet))
	if size == 0 {
		return nil, fmt.Errorf("BinarySet size is zero")
	}
	datas := make([]unsafe.Pointer, size)

	C.GetBinarySetKeys(cBinarySet, unsafe.Pointer(&datas[0]))
	ret := make([]string, size)
	for i := 0; i < size; i++ {
		ret[i] = C.GoString((*C.char)(datas[i]))
	}

	return ret, nil
}

func GetBinarySetSize(cBinarySet C.CBinarySet, key string) (int64, error) {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))
	ret := C.GetBinarySetValueSize(cBinarySet, cKey)
	return int64(ret), nil
}

// HandleCStatus deal with the error returned from CGO
func HandleCStatus(status *C.CStatus, extraInfo string) error {
	if status.error_code == 0 {
		return nil
	}
	errorCode := int(status.error_code)
	errorMsg := C.GoString(status.error_msg)
	defer C.free(unsafe.Pointer(status.error_msg))

	logMsg := fmt.Sprintf("%s, C Runtime Exception: %s\n", extraInfo, errorMsg)
	log.Warn(logMsg)
	return merr.WrapErrSegcore(int32(errorCode), logMsg)
}
