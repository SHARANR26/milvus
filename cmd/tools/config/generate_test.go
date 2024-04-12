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

package main

import (
	"bufio"
	"bytes"
	"fmt"
	"os"
	"testing"

	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/util/paramtable"
	"github.com/stretchr/testify/assert"
	"go.uber.org/zap"
)

// Assert the milvus.yaml file is consistent to paramtable
func TestYamlFile(t *testing.T) {
	w := bytes.Buffer{}
	WriteYaml(&w)

	base := paramtable.NewBaseTable()
	f, err := os.Open(fmt.Sprintf("%s/%s", base.GetConfigDir(), "milvus.yaml"))
	assert.NoError(t, err, "expecting configs/milvus.yaml")
	defer f.Close()
	fileScanner := bufio.NewScanner(f)
	codeScanner := bufio.NewScanner(&w)
	for fileScanner.Scan() && codeScanner.Scan() {
		if fileScanner.Text() != codeScanner.Text() {
			assert.FailNow(t, fmt.Sprintf("configs/milvus.yaml is not consistent with paramtable, file: %s, code: %s",
				fileScanner.Text(), codeScanner.Text()))
		}
		log.Error("", zap.Any("file", fileScanner.Text()), zap.Any("code", codeScanner.Text()))
	}
}
