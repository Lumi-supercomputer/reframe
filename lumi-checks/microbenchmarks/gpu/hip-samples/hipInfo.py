# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

# rfmdocstart: streamtest
import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class HipInfo(rfm.RegressionTest):
    valid_systems = ['lumi:eap']
    valid_prog_environs = ['builtin']
    prebuild_cmds = [
        'curl -O https://raw.githubusercontent.com/ROCm-Developer-Tools/HIP/develop/samples/1_Utils/hipInfo/hipInfo.cpp'
    ]
    build_system = 'SingleSource'
    sourcepath = 'hipInfo.cpp'
    build_locally = False
    num_gpus_per_node = 1
    #variables = {
    #        'HIP_PATH': '/opt/rocm/hip'
    #}

    @run_before('compile')
    def set_compiler_flags(self):
        #self.build_system.lang = 'hip'
        self.build_system.cc = 'hipcc'
        self.build_system.cxx = 'hipcc'

    @sanity_function
    def validate_solution(self):
        return sn.assert_found(r'Solution Validates', self.stdout)

    @performance_function('MB/s', perf_key='Copy')
    def extract_copy_perf(self):
        return sn.extractsingle(r'Copy:\s+(\S+)\s+.*', self.stdout, 1, float)

    @performance_function('MB/s', perf_key='Scale')
    def extract_scale_perf(self):
        return sn.extractsingle(r'Scale:\s+(\S+)\s+.*', self.stdout, 1, float)

    @performance_function('MB/s', perf_key='Add')
    def extract_add_perf(self):
        return sn.extractsingle(r'Add:\s+(\S+)\s+.*', self.stdout, 1, float)

    @performance_function('MB/s', perf_key='Triad')
    def extract_triad_perf(self):
        return sn.extractsingle(r'Triad:\s+(\S+)\s+.*', self.stdout, 1, float)
# rfmdocend: streamtest
