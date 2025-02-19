# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm


@rfm.simple_test
class SomeTest(rfm.RegressionTest):
    valid_systems = []
    valid_prog_environs = []


class NotATest:
    pass
