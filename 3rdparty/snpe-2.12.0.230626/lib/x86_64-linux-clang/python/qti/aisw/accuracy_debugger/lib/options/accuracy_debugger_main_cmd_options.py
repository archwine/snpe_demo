# =============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.accuracy_debugger.lib.options.cmd_options import CmdOptions

import argparse
import sys
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import ParameterError

class MainCmdOptions(CmdOptions):

    def __init__(self, args):
        super().__init__('main', args)

    def initialize(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="Script that runs Framework Diagnosis, Inference Engine or Verification.",
            add_help=False,
            allow_abbrev=False
        )

        components = self.parser.add_argument_group('Arguments to select which component of the tool to run.  '
                                                    'Arguments are mutually exclusive (at most 1 can be selected).  If none are selected, then all components are run')
        components.add_argument('--framework_diagnosis', action="store_true", default=False,
                            help="Run framework")
        components.add_argument('--inference_engine', action="store_true", default=False,
                            help="Run inference engine")
        components.add_argument('--verification', action="store_true", default=False,
                            help="Run verification")

        optional = self.parser.add_argument_group('optional arguments')
        optional.add_argument('-h', '--help', action="store_true", default=False,
                            help="Show this help message.  To show help for any of the components, run script with --help and --<component>.  "
                            "For example, to show the help for Framework Diagnosis, run script with the following: "
                            "--help --framework_diagnosis")

        self.initialized = True

    def verify_update_parsed_args(self, parsed_args):
        # print help for MainCmdOptions
        if (parsed_args.help):
            self.parser.print_help()

        # currently only support running one component at a time, or running wrapper
        component_args_list = [ parsed_args.framework_diagnosis,
                                parsed_args.inference_engine,
                                parsed_args.verification]

        # check to ensure only one or no components are selected
        if (sum(component_args_list) > 1):
            raise ParameterError("Too many components selected. Please run script with only one or no components")

        return parsed_args

    def parse(self):
        if (not self.initialized):
            self.initialize()
        opts, _ = self.parser.parse_known_args(self.args)
        return self.verify_update_parsed_args(opts)
