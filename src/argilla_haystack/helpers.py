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

from typing import Dict, List


def create_tree_with_durations(agent_duration: str, tool_durations: Dict) -> List:
    """Create the tree data to be converted to an SVG, including the agent and tools duration."""
    data = []
    data.append((0, 0, "AGENT", agent_duration))
    current_row = 1

    if tool_durations:
        for tool_name, duration in tool_durations.items():
            data.append((current_row, 1, tool_name.upper(), duration))
            current_row += 1

    return data


def create_svg_with_durations(data: List) -> str:
    """Create an SVG with the tree data."""
    box_height = 47
    box_width = box_height * 10
    row_constant = box_height + 7
    indent_constant = 40
    font_size_node_name = box_height * 0.4188
    font_size_time = font_size_node_name - 4
    text_centering = box_height * 0.6341
    node_name_indent = box_height * 0.35
    time_indent = box_height * 8.75

    body = ""
    for each in data:
        row, indent, node_name, node_time = each
        body_raw = f"""
<g transform="translate({indent*indent_constant}, {row*row_constant})">
<rect x=".5" y=".5" width="{box_width}" height="{box_height}" rx="8.49" ry="8.49" style="fill: #24272e; stroke: #afdfe5; stroke-miterlimit: 10;"/>
<text transform="translate({node_name_indent} {text_centering})" style="fill: #fff; font-size: {font_size_node_name}px;"><tspan x="0" y="0">{node_name}</tspan></text>
<text transform="translate({time_indent} {text_centering})" style="fill: #b7d989; font-size: {font_size_time}px; font-style: italic;"><tspan x="0" y="0">{node_time}</tspan></text>
</g>
        """
        body += body_raw

    base = f"""
<?xml version="1.0" encoding="UTF-8"?>
<svg id="Layer_1" data-name="Layer 1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 750 {len(data)*row_constant}">
{body}
</svg>
    """
    base = base.strip()
    return base
