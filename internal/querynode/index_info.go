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

package querynode

// indexInfo stores index info, such as name, id, index params and so on
type indexInfo struct {
	indexName   string
	indexID     UniqueID
	buildID     UniqueID
	fieldID     UniqueID
	indexPaths  []string
	indexParams map[string]string
	readyLoad   bool
}

// newIndexInfo returns a new indexInfo
func newIndexInfo() *indexInfo {
	return &indexInfo{
		indexPaths:  make([]string, 0),
		indexParams: make(map[string]string),
	}
}

func (info *indexInfo) setIndexName(name string) {
	info.indexName = name
}

// setIndexID sets the id of index
func (info *indexInfo) setIndexID(id UniqueID) {
	info.indexID = id
}

func (info *indexInfo) setBuildID(id UniqueID) {
	info.buildID = id
}

func (info *indexInfo) setFieldID(id UniqueID) {
	info.fieldID = id
}

func (info *indexInfo) setIndexPaths(paths []string) {
	info.indexPaths = paths
}

func (info *indexInfo) setIndexParams(params map[string]string) {
	info.indexParams = params
}

func (info *indexInfo) setReadyLoad(load bool) {
	info.readyLoad = load
}

func (info *indexInfo) getIndexName() string {
	return info.indexName
}

func (info *indexInfo) getIndexID() UniqueID {
	return info.indexID
}

func (info *indexInfo) getBuildID() UniqueID {
	return info.buildID
}

func (info *indexInfo) getFieldID() UniqueID {
	return info.fieldID
}

func (info *indexInfo) getIndexPaths() []string {
	return info.indexPaths
}

func (info *indexInfo) getIndexParams() map[string]string {
	return info.indexParams
}

func (info *indexInfo) getReadyLoad() bool {
	return info.readyLoad
}
