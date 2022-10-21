# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""VF2 Partial Layout plugin."""

from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import VF2Layout
from qiskit.transpiler.passes.layout.vf2_layout import VF2LayoutStopReason
from qiskit.transpiler.preset_passmanagers import common
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin

from .vf2_partial_layout import VF2PartialLayout


def _choose_layout_condition(property_set):
    # layout hasn't been set yet
    return not property_set["layout"]


def _vf2_match_not_found(property_set):
    # If a layout hasn't been set by the time we run vf2 layout we need to
    # run layout
    if property_set["layout"] is None:
        return True
    # if VF2 layout stopped for any reason other than solution found we need
    # to run layout since VF2 didn't converge.
    if (
        property_set["VF2Layout_stop_reason"] is not None
        and property_set["VF2Layout_stop_reason"]
        is not VF2LayoutStopReason.SOLUTION_FOUND
    ):
        return True
    return False


class VF2PartialLayoutPlugin(PassManagerStagePlugin):
    """Plugin for using vf2 partial layout."""

    def pass_manager(self, pass_manager_config, optimization_level=None):
        """build vf2 partial layout plugin stage pass manager."""
        call_limit = int(3e7)
        if optimization_level == 0:
            call_limit = int(1e4)
        elif optimization_level == 1:
            call_limit = int(1e5)
        elif optimization_level == 2:
            call_limit = int(5e6)
        _choose_layout_0 = VF2Layout(
            coupling_map=pass_manager_config.coupling_map,
            seed=pass_manager_config.seed_transpiler,
            properties=pass_manager_config.backend_properties,
            target=pass_manager_config.target,
            call_limit=call_limit,
        )
        layout_pm = PassManager()
        layout_pm.append(_choose_layout_0, condition=_choose_layout_condition)
        layout_pm.append(
            VF2PartialLayout(
                coupling_map=pass_manager_config.coupling_map,
                seed=pass_manager_config.seed_transpiler,
                target=pass_manager_config.target,
                call_limit=int(5e4),
            ),
            condition=_vf2_match_not_found,
        )
        layout_pm += common.generate_embed_passmanager(pass_manager_config.coupling_map)
        return layout_pm
