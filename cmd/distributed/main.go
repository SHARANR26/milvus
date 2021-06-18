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
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"syscall"

	"github.com/milvus-io/milvus/cmd/distributed/roles"
)

const (
	roleRootCoord  = "rootcoord"
	roleQueryCoord = "querycoord"
	roleIndexCoord = "indexcoord"
	roleDataCoord  = "datacoord"
	roleProxyNode  = "proxynode"
	roleQueryNode  = "querynode"
	roleIndexNode  = "indexnode"
	roleDataNode   = "datanode"
	roleMixture    = "mixture"
	roleStandalone = "standalone"
)

func getPidFileName(serverType string, alias string) string {
	var filename string
	if len(alias) != 0 {
		filename = fmt.Sprintf("%s-%s.pid", serverType, alias)
	} else {
		filename = serverType + ".pid"
	}
	return filename
}

func createPidFile(filename string, runtimeDir string) (*os.File, error) {
	fileFullName := path.Join(runtimeDir, filename)

	fd, err := os.OpenFile(fileFullName, os.O_CREATE|os.O_RDWR, 0664)
	if err != nil {
		return nil, fmt.Errorf("file %s is locked, error = %w", filename, err)
	}
	fmt.Println("open pid file:", fileFullName)

	err = syscall.Flock(int(fd.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
	if err != nil {
		return nil, fmt.Errorf("file %s is locked, error = %w", filename, err)
	}
	fmt.Println("lock pid file:", fileFullName)

	fd.Truncate(0)
	_, err = fd.WriteString(fmt.Sprintf("%d", os.Getpid()))
	if err != nil {
		return nil, fmt.Errorf("file %s write fail, error = %w", filename, err)
	}

	return fd, nil
}

func closePidFile(fd *os.File) {
	fd.Close()
}

func removePidFile(fd *os.File) {
	syscall.Close(int(fd.Fd()))
	os.Remove(fd.Name())
}

func stopPid(filename string, runtimeDir string) error {
	var pid int

	fd, err := os.OpenFile(path.Join(runtimeDir, filename), os.O_RDONLY, 0664)
	if err != nil {
		return err
	}
	defer closePidFile(fd)

	_, err = fmt.Fscanf(fd, "%d", &pid)
	if err != nil {
		return err
	}
	process, err := os.FindProcess(pid)
	if err != nil {
		return err
	}
	err = process.Signal(syscall.SIGTERM)
	if err != nil {
		return err
	}
	return nil
}

func makeRuntimeDir(dir string) error {
	st, err := os.Stat(dir)
	if os.IsNotExist(err) {
		err = os.Mkdir(dir, 0755)
		if err != nil {
			return fmt.Errorf("create runtime dir %s failed", dir)
		}
		return nil
	}
	if !st.IsDir() {
		return fmt.Errorf("%s is not directory", dir)
	}
	tmpFile, err := ioutil.TempFile(dir, "tmp")
	if err != nil {
		return err
	}
	fileName := tmpFile.Name()
	tmpFile.Close()
	os.Remove(fileName)
	return nil
}

func main() {
	if len(os.Args) < 3 {
		_, _ = fmt.Fprint(os.Stderr, "usage: milvus [command] [server type] [flags]\n")
		return
	}
	command := os.Args[1]
	serverType := os.Args[2]
	flags := flag.NewFlagSet(os.Args[0], flag.ExitOnError)

	var svrAlias string
	flags.StringVar(&svrAlias, "alias", "", "set alias")

	var enableRootCoord, enableQueryCoord, enableIndexCoord, enableDataCoord bool
	flags.BoolVar(&enableRootCoord, roleRootCoord, false, "enable root coordinator")
	flags.BoolVar(&enableQueryCoord, roleQueryCoord, false, "enable query coordinator")
	flags.BoolVar(&enableIndexCoord, roleIndexCoord, false, "enable index coordinator")
	flags.BoolVar(&enableDataCoord, roleDataCoord, false, "enable data coordinator")

	if err := flags.Parse(os.Args[3:]); err != nil {
		os.Exit(-1)
	}

	var localMsg = false
	role := roles.MilvusRoles{}
	switch serverType {
	case roleRootCoord:
		role.EnableRootCoord = true
	case roleProxyNode:
		role.EnableProxyNode = true
	case roleQueryCoord:
		role.EnableQueryCoord = true
	case roleQueryNode:
		role.EnableQueryNode = true
	case roleDataCoord:
		role.EnableDataCoord = true
	case roleDataNode:
		role.EnableDataNode = true
	case roleIndexCoord:
		role.EnableIndexCoord = true
	case roleIndexNode:
		role.EnableIndexNode = true
	case roleMixture:
		role.EnableRootCoord = enableRootCoord
		role.EnableQueryCoord = enableQueryCoord
		role.EnableDataCoord = enableDataCoord
		role.EnableIndexCoord = enableIndexCoord
	case roleStandalone:
		role.EnableRootCoord = true
		role.EnableProxyNode = true
		role.EnableQueryCoord = true
		role.EnableQueryNode = true
		role.EnableDataCoord = true
		role.EnableDataNode = true
		role.EnableIndexCoord = true
		role.EnableIndexNode = true
		role.EnableMsgStreamCoord = true
		localMsg = true
	default:
		fmt.Fprintf(os.Stderr, "Unknown server type = %s\n", serverType)
		os.Exit(-1)
	}

	runtimeDir := "/run/milvus"
	if err := makeRuntimeDir(runtimeDir); err != nil {
		fmt.Fprintf(os.Stderr, "Set runtime dir at %s failed, set it to /tmp/milvus directory\n", runtimeDir)
		runtimeDir = "/tmp/milvus"
		if err = makeRuntimeDir(runtimeDir); err != nil {
			fmt.Fprintf(os.Stderr, "Create runtime directory at %s failed\n", runtimeDir)
			os.Exit(-1)
		}
	}

	filename := getPidFileName(serverType, svrAlias)
	switch command {
	case "run":
		fd, err := createPidFile(filename, runtimeDir)
		if err != nil {
			panic(err)
		}
		defer removePidFile(fd)
		role.Run(localMsg)
	case "stop":
		if err := stopPid(filename, runtimeDir); err != nil {
			panic(err)
		}
	default:
		fmt.Fprintf(os.Stderr, "unknown command : %s", command)
	}
}
