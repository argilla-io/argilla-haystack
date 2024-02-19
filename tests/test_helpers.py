# Copyright 2023-present, Argilla, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

import pytest
from argilla_haystack.helpers import (
    create_svg_with_durations,
    create_tree_with_durations,
)


@pytest.mark.parametrize(
    "agent_duration, tool_durations, expected_result",
    [
        ("10.255", {}, [(0, 0, "AGENT", "10.255")]),
        (
            "10.255",
            {"tool1": "1", "tool2": "2.56", "tool3": "2.1"},
            [
                (0, 0, "AGENT", "10.255"),
                (1, 1, "TOOL1", "1"),
                (2, 1, "TOOL2", "2.56"),
                (3, 1, "TOOL3", "2.1"),
            ],
        ),
    ],
)
def test_create_tree_with_durations(agent_duration, tool_durations, expected_result):
    assert create_tree_with_durations(agent_duration, tool_durations) == expected_result


def test_create_svg_with_dynamic_content():
    data = [
        (0, 0, "AGENT", "10.255"),
        (1, 1, "TOOL1", "1"),
        (2, 1, "TOOL2", "2.56"),
        (3, 1, "TOOL3", "2.1"),
    ]
    svg_output = create_svg_with_durations(data)

    for _, _, label, duration in data:
        label_pattern = re.compile(rf'<tspan x="0" y="0">{label}</tspan>')
        duration_pattern = re.compile(rf'<tspan x="0" y="0">{duration}</tspan>')
        assert label_pattern.search(svg_output)
        assert duration_pattern.search(svg_output)
