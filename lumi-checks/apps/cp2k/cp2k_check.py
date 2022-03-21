# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


class Cp2kCheck(rfm.RunOnlyRegressionTest):
    modules = ['CP2K']
    executable = 'cp2k.psmp'
    executable_opts = ['H2O-256.inp']
    maintainers = ['LM']
    tags = {'scs'}
    strict_check = False

    @run_after('init')
    def set_prgenv(self):
        if self.current_system.name in ['eiger', 'pilatus','lumi']:
            self.valid_prog_environs = ['cpeGNU']
        else:
            self.valid_prog_environs = ['builtin']

    @sanity_function
    def assert_energy_diff(self):
        energy = sn.extractsingle(
            r'\s+ENERGY\| Total FORCE_EVAL \( QS \) '
            r'energy [\[\(]a\.u\.[\]\)]:\s+(?P<energy>\S+)',
            self.stdout, 'energy', float, item=-1
        )
        energy_reference = -4404.2323
        energy_diff = sn.abs(energy-energy_reference)
        return sn.all([
            sn.assert_found(r'PROGRAM STOPPED IN', self.stdout),
            sn.assert_eq(sn.count(sn.extractall(
                r'(?i)(?P<step_count>STEP NUMBER)',
                self.stdout, 'step_count')), 10),
            sn.assert_lt(energy_diff, 1e-4)
        ])

    @performance_function('s')
    def time(self):
        return sn.extractsingle(r'^ CP2K(\s+[\d\.]+){4}\s+(?P<perf>\S+)',
                                self.stdout, 'perf', float)


@rfm.simple_test
class Cp2kCpuCheck(Cp2kCheck):
    scale = parameter(['small', 'large'])
    valid_systems = ['daint:mc', 'eiger:mc', 'pilatus:mc']
    refs_by_scale = {
        'small': {
            'dom:mc': {'time': (164, None, 0.07, 's')},
            'daint:mc': {'time': (164, None, 0.07, 's')},
            'eiger:mc': {'time': (70.0, None, 0.08, 's')},
            'pilatus:mc': {'time': (70.0, None, 0.08, 's')},
            'lumi:small': {'time': (105.0, None, 0.08, 's')}
        },
        'large': {
            'daint:mc': {'time': (120, None, 0.15, 's')},
            'eiger:mc': {'time': (46.0, None, 0.05, 's')},
            'pilatus:mc': {'time': (46.0, None, 0.05, 's')}
            'lumi:stadard': {'time': (46.0, None, 0.05, 's')},
        }
    }

    @run_after('init')
    def setup_by__scale(self):
        self.descr = f'CP2K CPU check (version: {self.scale})'
        self.tags |= {'maintenance', 'production'}
        if self.scale == 'small':
            self.valid_systems += ['dom:mc', 'lumi:lumi-c']
            if self.current_system.name in ['daint', 'dom']:
                self.num_tasks = 216
                self.num_tasks_per_node = 36
            elif self.current_system.name in ['eiger', 'pilatus', 'lumi']:
                self.num_tasks = 64
                self.num_tasks_per_node = 16
                self.num_cpus_per_task = 16
                self.num_tasks_per_core = 1
                self.use_multithreading = False
                self.variables = {
                    'MPICH_OFI_STARTUP_CONNECT': '1',
                    'OMP_NUM_THREADS': '8',
                    'OMP_PLACES': 'cores',
                    'OMP_PROC_BIND': 'close'
                }
        else:
            if self.current_system.name in ['daint', 'dom']:
                self.num_tasks = 576
                self.num_tasks_per_node = 36
            elif self.current_system.name in ['eiger', 'pilatus']:
                self.num_tasks = 256
                self.num_tasks_per_node = 16
                self.num_cpus_per_task = 16
                self.num_tasks_per_core = 1
                self.use_multithreading = False
                self.variables = {
                    'MPICH_OFI_STARTUP_CONNECT': '1',
                    'OMP_NUM_THREADS': '8',
                    'OMP_PLACES': 'cores',
                    'OMP_PROC_BIND': 'close'
                }

        self.reference = self.refs_by_scale[self.scale]

    @run_before('run')
    def set_task_distribution(self):
        self.job.options = ['--distribution=block:block']
        self.job.options = ['--time=5']

    @run_before('run')
    def set_cpu_binding(self):
        self.job.launcher.options = ['--cpu-bind=cores']


#@rfm.simple_test
#class Cp2kGpuCheck(Cp2kCheck):
#    scale = parameter(['small', 'large'])
#    valid_systems = ['daint:gpu']
#    num_gpus_per_node = 1
#    num_tasks_per_node = 6
#    num_cpus_per_task = 2
#    variables = {
#        'CRAY_CUDA_MPS': '1',
#        'OMP_NUM_THREADS': str(num_cpus_per_task)
#    }
#    refs_by_scale = {
#        'small': {
#            'dom:gpu': {'time': (234, None, 0.05, 's')},
#            'daint:gpu': {'time': (234, None, 0.05, 's')}
#        },
#        'large': {
#            'daint:gpu': {'time': (147, None, 0.05, 's')}
#        }
#    }
#
#    @run_after('init')
#    def setup_by_scale(self):
#        self.descr = f'CP2K GPU check (version: {self.scale})'
#        if self.scale == 'small':
#            self.valid_systems += ['dom:gpu']
#            self.num_tasks = 36
#        else:
#            self.num_tasks = 96
#
#        self.reference = self.refs_by_scale[self.scale]
#        self.tags |= {'maintenance', 'production'}
