"""
package: dgcv - Differential Geometry with Complex Variables

module: dgcv._aux.styles


---
Many themes in the THEME_REGISTRY, which are collections CSS properties used to style colors on html outputs, are AI-generated. And this module's tooling for emitting those themed CSS sets was heavily refactored with AI.

SPDX-License-Identifier: Apache-2.0
SPDX-AI-Disclosure: ai-generated
"""

# -----------------------------------------------------------------------------
# imports and broadcasting
# -----------------------------------------------------------------------------
from __future__ import annotations

import colorsys
import difflib
import random
import re
from dataclasses import MISSING, asdict, dataclass, field, fields, replace
from string import Template
from typing import Dict, List, Optional, Tuple, Union

from ._config import get_dgcv_settings_registry

__all__ = ["get_dgcv_themes", "get_style", "ThemeConfig"]


# -----------------------------------------------------------------------------
# body
# -----------------------------------------------------------------------------

dgcv_display_theme = "matte_slate_soft"
dgcv_custom_variables = [
    "--dgcv-border-width",
    "--dgcv-border-radius",
    "--dgcv-hover-transform",
    "--dgcv-table-shadow",
    "--dgcv-special-background",
    "--dgcv-text-shadow",
    "--dgcv-hover-transition",
    "--dgcv-border-image",
    "--dgcv-hover-font-weight",
]


@dataclass
class ThemeConfig:
    """
    Background/foreground pairs:

        ``bg_surface``      /  ``text_heading``
        ``bg_primary``      /  ``text_main``
        ``bg_alt``          /  ``text_alt``       (falls back to text_main if unset)
        ``bg_hover``        /  ``text_hover``
        ``bg_action``       /  ``text_on_action``
        ``bg_action_hover`` /  ``text_on_action``
        ``bg_error``        /  ``text_on_error``
        ``bg_success``      /  ``text_on_success``

        optional pair in custom_css_vars:
        ``--dgcv-special-background``      /  ``--dgcv-special-text``
        If ``text_heading`` doesn't contrast with ``--dgcv-special-background`` then
        ``--dgcv-special-text`` should be provided.

    Attributes:
        bg_primary: Main background color for table/panel body areas.
        text_main: Default body text color. Paired with ``bg_primary``.
        bg_surface: Header/caption background. Paired with ``text_heading``.
        text_heading: Text color for headers and captions. Paired with ``bg_surface``.
        bg_alt: Alternating row background. Paired with ``text_alt``.
        bg_hover: Background applied on hover. Paired with ``text_hover``.
        text_hover: Text color applied on hover. Paired with ``bg_hover``.
        bg_action: Background for action/button elements. Paired with ``text_on_action``.
        text_on_action: Text color for action elements. Paired with ``bg_action``.
        bg_action_hover: Hover background for action elements. Paired with ``text_on_action``.
        bg_error: Background for error states. Paired with ``text_on_error``.
        text_on_error: Text color for error states. Paired with ``bg_error``.
        bg_success: Background for success states. Paired with ``text_on_success``.
        text_on_success: Text color for success states. Paired with ``bg_success``.
        border_main: Primary border color, used for outer borders and header dividers.
        border_alt: Secondary border color, used for row dividers and subtle borders.
        font_family: CSS font-family string. Defaults to ``"inherit"``.
        text_alt: Alternating row text color. Paired with ``bg_alt``. Falls back
            to ``text_main`` if not set.
        custom_css_vars: Optional dict of additional CSS custom properties to
            emit alongside the standard palette. Keys may include or omit the
            leading ``--``; if omitted the ``dgcv-`` prefix is applied
            automatically. See the dgcv custom variables registry for supported
            values and their effects.
    """

    bg_primary: str
    text_main: str
    bg_surface: str
    text_heading: str
    bg_alt: str
    bg_hover: str
    text_hover: str
    bg_action: str
    text_on_action: str
    bg_action_hover: str
    bg_error: str
    text_on_error: str
    bg_success: str
    text_on_success: str
    border_main: str
    border_alt: str

    font_family: str = "inherit"
    text_alt: Optional[str] = None
    custom_css_vars: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.text_alt is None:
            self.text_alt = self.text_main

    def to_css_dict(self, prefix: str = "dgcv-") -> Dict[str, str]:
        """
        Returns all theme values as a dict of CSS custom property names to values.

        Standard fields are emitted as ``--{prefix}{field-name}`` with underscores
        replaced by hyphens. Entries in ``custom_css_vars`` are appended after the
        standard fields; keys that already start with ``--`` are used verbatim,
        otherwise the prefix is applied.

        Args:
            prefix: CSS variable prefix. Defaults to ``"dgcv-"``.

        Returns:
            Ordered dict of CSS variable names to their string values.
        """
        css_vars = {}

        for key, value in asdict(self).items():
            if key == "custom_css_vars" or value is None:
                continue

            css_key = f"--{prefix}{key.replace('_', '-')}"
            css_vars[css_key] = str(value)

        for key, value in self.custom_css_vars.items():
            if key.startswith("--"):
                css_key = key
            else:
                css_key = f"--{prefix}{key.replace('_', '-')}"

            css_vars[css_key] = str(value)

        return css_vars

    def to_css_string(self, prefix: str = "dgcv-") -> str:
        """
        Returns all theme values as a newline-separated string of CSS declarations,
        indented for embedding directly inside a ``{ }`` block.

        Args:
            prefix: CSS variable prefix. Defaults to ``"dgcv-"``.

        Returns:
            String of the form ``--dgcv-bg-primary: #fff;\\n    --dgcv-text-main: #000;``
            suitable for insertion into a ``:root { }`` or scoped selector block.
        """
        css_dict = self.to_css_dict(prefix)
        return "\n    ".join(f"{k}: {v};" for k, v in css_dict.items())

    def registry_format(self, name: str) -> str:
        lines = [f'"{name}": ThemeConfig(']
        fields = [
            ("bg_primary", self.bg_primary),
            ("bg_surface", self.bg_surface),
            ("bg_alt", self.bg_alt),
            ("bg_hover", self.bg_hover),
            ("text_main", self.text_main),
            ("text_heading", self.text_heading),
            ("text_hover", self.text_hover),
            ("text_alt", self.text_alt),
            ("border_main", self.border_main),
            ("border_alt", self.border_alt),
            ("bg_action", self.bg_action),
            ("text_on_action", self.text_on_action),
            ("bg_action_hover", self.bg_action_hover),
            ("bg_error", self.bg_error),
            ("text_on_error", self.text_on_error),
            ("bg_success", self.bg_success),
            ("text_on_success", self.text_on_success),
            ("font_family", self.font_family),
        ]
        for k, v in fields:
            lines.append(f'    {k}="{v}",')
        if self.custom_css_vars:
            lines.append("    custom_css_vars={")
            for k, v in self.custom_css_vars.items():
                lines.append(f'        "{k}": "{v}",')
            lines.append("    },")
        lines.append("),")
        result = "\n".join(lines)
        print(result)
        return result


THEME_REGISTRY: Dict[str, ThemeConfig] = {
    "1980s_neon": ThemeConfig(
        bg_primary="#3b3b58",
        bg_surface="#9400d3",
        bg_alt="#282a36",
        bg_hover="#00ff7f",
        text_main="#ffffff",
        text_heading="#00ff00",
        text_hover="#000000",
        border_main="#ff1493",
        bg_action="#ff1493",
        text_on_action="#ffffff",
        bg_action_hover="#ff69b4",
        bg_error="#ff0000",
        text_on_error="#ffffff",
        bg_success="#00ff00",
        text_on_success="#000000",
        border_alt="#4b0082",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-image": "linear-gradient(90deg, #ff1493, #9400d3, #00ff7f) 1",
            "--dgcv-border-radius": "0",
            "--dgcv-border-width": "2px",
            "--dgcv-text-shadow": "0px 0px 8px #00ff00, 0px 0px 2px #ffffff",
            "--dgcv-table-shadow": "0 0 20px rgba(0, 255, 127, 0.7), 0 0 60px rgba(148, 0, 211, 0.4)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.2s ease",
        },
    ),
    "appalachian": ThemeConfig(
        bg_primary="#F0F8FF",
        bg_surface="#2E8B57",
        bg_alt="#B0E0E6",
        bg_hover="#5F9EA0",
        text_main="#2F4F4F",
        text_heading="#ffffff",
        text_hover="#ffffff",
        border_main="#2E8B57",
        bg_action="#1E90FF",
        text_on_action="#ffffff",
        bg_action_hover="#00BFFF",
        bg_error="#D32F2F",
        text_on_error="#ffffff",
        bg_success="#388E3C",
        text_on_success="#ffffff",
        border_alt="#87CEFA",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-special-background": "linear-gradient(160deg, #5f9ea0 0%, #52898b 50%, #457375 100%)",
            "--dgcv-table-shadow": "0 0 12px rgba(46, 139, 87, 0.35)",
            "--dgcv-text-shadow": "0 1px 0 rgba(255,255,255,0.5)",
        },
    ),
    "aurora_borealis": ThemeConfig(
        bg_primary="#0a0e1a",
        bg_surface="#0d1f2d",
        bg_alt="#0d1a2a",
        bg_hover="#00e5cc",
        text_main="#c8f0e8",
        text_heading="#c8f0e8",
        text_hover="#0a0e1a",
        text_alt="#a8d8cc",
        border_main="#00b4a0",
        border_alt="#1a3a4a",
        bg_action="#7b2fff",
        text_on_action="#ffffff",
        bg_action_hover="#9d5fff",
        bg_error="#ff4b2b",
        text_on_error="#ffffff",
        bg_success="#00e676",
        text_on_success="#0a0e1a",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(170deg, #0a0e1a 0%, #0d3320 35%, #0a1a3a 65%, #1a0a2e 100%)",
            "--dgcv-table-shadow": "0 0 20px rgba(0, 180, 160, 0.4), 0 0 60px rgba(123, 47, 255, 0.2)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.3s ease",
            "--dgcv-text-shadow": "0 0 8px rgba(0, 229, 204, 0.6)",
        },
    ),
    "autumn": ThemeConfig(
        bg_primary="#ffe0b2",
        bg_surface="#bf360c",
        bg_alt="#ffcc80",
        bg_hover="#ffb74d",
        text_main="#5d1a06",
        text_heading="#ffffff",
        text_hover="#5d1a06",
        border_main="#bf360c",
        bg_action="#e65100",
        text_on_action="#ffffff",
        bg_action_hover="#ff9800",
        bg_error="#b71c1c",
        text_on_error="#ffffff",
        bg_success="#33691e",
        text_on_success="#ffffff",
        border_alt="#ffb74d",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-special-background": "linear-gradient(170deg, #ffe0b2 0%, #ffcc80 60%, #ffb74d 100%)",
            "--dgcv-special-text": "#5d1a06",
            "--dgcv-table-shadow": "0 0 14px rgba(191, 54, 12, 0.4)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-text-shadow": "0 1px 2px rgba(93, 26, 6, 0.2)",
        },
    ),
    "back_to_the_future": ThemeConfig(
        bg_primary="#0059b3",
        bg_surface="#003366",
        bg_alt="#004080",
        bg_hover="#ff7f50",
        text_main="#f7e014",
        text_heading="#f7e014",
        text_hover="#000000",
        border_main="#c0c0c0",
        bg_action="#ff7f50",
        text_on_action="#000000",
        bg_action_hover="#ff9966",
        bg_error="#ff3333",
        text_on_error="#ffffff",
        bg_success="#00e676",
        text_on_success="#000000",
        border_alt="#808080",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-image": "linear-gradient(45deg, #c0c0c0, #ff7f50, #00ffff) 1",
            "--dgcv-border-radius": "0",
            "--dgcv-text-shadow": "0px 0px 8px #ff7f50, 0px 0px 2px #f7e014",
            "--dgcv-table-shadow": "0 4px 16px rgba(255, 127, 80, 0.8), 0 0 40px rgba(0, 255, 255, 0.2)",
            "--dgcv-hover-transform": "scale(1.001)",
        },
    ),
    "blue": ThemeConfig(
        bg_primary="#24283b",
        bg_surface="#24283b",
        bg_alt="#2f3549",
        bg_hover="#2f3549",
        text_main="#a9b1d6",
        text_heading="#a9b1d6",
        text_hover="#7dcfff",
        border_main="#a9b1d6",
        bg_action="#3d59a1",
        text_on_action="#ffffff",
        bg_action_hover="#7dcfff",
        bg_error="#f7768e",
        text_on_error="#24283b",
        bg_success="#9ece6a",
        text_on_success="#24283b",
        border_alt="#414868",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-table-shadow": "0 0 10px rgba(125, 207, 255, 0.5)",
        },
    ),
    "blue_gray": ThemeConfig(
        bg_primary="#2e3440",
        bg_surface="#2e3440",
        bg_alt="#3b4252",
        bg_hover="#434c5e",
        text_main="#d8dee9",
        text_heading="#eceff4",
        text_hover="#8fbcbb",
        border_main="#4c566a",
        bg_action="#5e81ac",
        text_on_action="#eceff4",
        bg_action_hover="#81a1c1",
        bg_error="#bf616a",
        text_on_error="#eceff4",
        bg_success="#a3be8c",
        text_on_success="#2e3440",
        border_alt="#434c5e",
        font_family="'Inter', sans-serif",
    ),
    "blue_plain": ThemeConfig(
        bg_primary="#ffffff",
        bg_surface="#0056b3",
        bg_alt="#f7f7f7",
        bg_hover="#cce5ff",
        text_main="#000000",
        text_heading="#ffffff",
        text_hover="#000000",
        border_main="#0056b3",
        bg_action="#007bff",
        text_on_action="#ffffff",
        bg_action_hover="#0056b3",
        bg_error="#dc3545",
        text_on_error="#ffffff",
        bg_success="#28a745",
        text_on_success="#ffffff",
        border_alt="#e9ecef",
        font_family="inherit",
    ),
    "blueprint": ThemeConfig(
        bg_primary="#002b4f",
        bg_surface="#003366",
        bg_alt="#003366",
        bg_hover="#336699",
        text_main="#ffffff",
        text_heading="#ffffff",
        text_hover="#ffffff",
        border_main="#cccccc",
        bg_action="#4da6ff",
        text_on_action="#002b4f",
        bg_action_hover="#80bfff",
        bg_error="#ff4d4d",
        text_on_error="#ffffff",
        bg_success="#4dff4d",
        text_on_success="#002b4f",
        border_alt="#e6e6e6",
        font_family="Roboto Mono, monospace",
        custom_css_vars={
            "--dgcv-border-width": "2px",
        },
    ),
    "brass": ThemeConfig(
        bg_primary="#FFF8DC",
        bg_surface="#B87333",
        bg_alt="#E6D5B8",
        bg_hover="#CD7F32",
        text_main="#4B3621",
        text_heading="#ffffff",
        text_hover="#ffffff",
        border_main="#B87333",
        bg_action="#8B4513",
        text_on_action="#ffffff",
        bg_action_hover="#A0522D",
        bg_error="#B22222",
        text_on_error="#ffffff",
        bg_success="#556B2F",
        text_on_success="#ffffff",
        border_alt="#D2B48C",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-border-image": "linear-gradient(135deg, #cd7f32, #ffe08a, #b87333, #ffd700, #cd7f32) 1",
            "--dgcv-border-radius": "0",
            "--dgcv-table-shadow": "0 0 12px rgba(205, 127, 50, 0.6), inset 0 1px 0 rgba(255, 224, 138, 0.4)",
            "--dgcv-text-shadow": "0 1px 0 rgba(255,255,255,0.3)",
        },
    ),
    "brushed_metal": ThemeConfig(
        bg_primary="#b0b8c1",
        bg_surface="#78909c",
        bg_alt="#9aa4ad",
        bg_hover="#cfd8dc",
        text_main="#1c2833",
        text_heading="#1c2833",
        text_hover="#1c2833",
        text_alt="#263238",
        border_main="#546e7a",
        border_alt="#90a4ae",
        bg_action="#455a64",
        text_on_action="#eceff1",
        bg_action_hover="#607d8b",
        bg_error="#c62828",
        text_on_error="#ffffff",
        bg_success="#2e7d32",
        text_on_success="#ffffff",
        font_family="'Courier New', monospace",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": (
                'url("data:image/svg+xml;utf8,'
                "<svg xmlns='http://www.w3.org/2000/svg' width='100%' height='100%'>"
                "<filter id='brushed' x='0%' y='0%' width='100%' height='100%'>"
                "<feTurbulence type='fractalNoise' baseFrequency='0.02 0.95' numOctaves='2' result='noise'/>"
                "<feColorMatrix type='matrix' values='1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 0.12 0' result='coloredNoise'/>"
                "<feBlend mode='multiply' in='SourceGraphic' in2='coloredNoise'/>"
                "</filter>"
                "<rect width='100%' height='100%' filter='url(%23brushed)' fill='none'/>"
                '</svg>"), '
                "linear-gradient(180deg, #d4d9dd 0%, #9aa4ad 40%, #b8bfc5 60%, #cfd4d8 80%, #a8b0b8 100%)"
            ),
            "--dgcv-table-shadow": "0 2px 8px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.3)",
            "--dgcv-hover-transform": "none",
            "--dgcv-hover-transition": "background-color 0.15s ease",
            "--dgcv-text-shadow": "0 1px 0 rgba(255,255,255,0.4)",
        },
    ),
    "coffee_shop": ThemeConfig(
        bg_primary="#fffaf0",
        bg_surface="#3e2723",
        bg_alt="#f5f5dc",
        bg_hover="#deb887",
        text_main="#3e2723",
        text_heading="#ffffff",
        text_hover="#3e2723",
        border_main="#8b4513",
        bg_action="#5d4037",
        text_on_action="#ffffff",
        bg_action_hover="#8d6e63",
        bg_error="#b71c1c",
        text_on_error="#ffffff",
        bg_success="#33691e",
        text_on_success="#ffffff",
        border_alt="#eaddc5",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-border-radius": "8px",
            "--dgcv-table-shadow": "0 0 10px rgba(62, 39, 35, 0.2)",
            "--dgcv-text-shadow": "none",
        },
    ),
    "cosmos": ThemeConfig(
        bg_primary="#070812",
        bg_surface="#10122a",
        bg_alt="#0b0c1c",
        bg_hover="#7fe0e8",
        text_main="#eef0fa",
        text_heading="#b69cf0",
        text_hover="#070812",
        text_alt="#a6a2c4",
        border_main="#5a4a98",
        border_alt="#1a1838",
        bg_action="#7a3ca8",
        text_on_action="#ffffff",
        bg_action_hover="#9a4cc0",
        bg_error="#d83a5a",
        text_on_error="#ffffff",
        bg_success="#2aa882",
        text_on_success="#06120e",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-text": "#eef0fa",
            "--dgcv-special-background": "radial-gradient(1.5px 1.5px at 10% 18%, #fff, transparent), radial-gradient(1px 1px at 22% 30%, #cfe0ff, transparent), radial-gradient(2px 2px at 35% 14%, #fff, transparent), radial-gradient(1px 1px at 48% 26%, #fff, transparent), radial-gradient(1.5px 1.5px at 60% 12%, #e8d0f0, transparent), radial-gradient(1px 1px at 72% 30%, #fff, transparent), radial-gradient(2px 2px at 85% 18%, #cfe0ff, transparent), radial-gradient(1px 1px at 15% 52%, #fff, transparent), radial-gradient(1.5px 1.5px at 40% 60%, #fff, transparent), radial-gradient(1px 1px at 58% 52%, #e8d0f0, transparent), radial-gradient(2px 2px at 78% 64%, #fff, transparent), radial-gradient(1px 1px at 28% 80%, #cfe0ff, transparent), radial-gradient(1.5px 1.5px at 52% 86%, #fff, transparent), radial-gradient(1px 1px at 70% 82%, #fff, transparent), radial-gradient(1px 1px at 90% 78%, #e8d0f0, transparent), radial-gradient(ellipse 50% 45% at 75% 72%, rgba(168,76,176,0.45) 0%, transparent 60%), radial-gradient(ellipse 45% 40% at 30% 68%, rgba(58,138,160,0.40) 0%, transparent 60%), radial-gradient(ellipse 60% 50% at 55% 40%, rgba(106,76,168,0.30) 0%, transparent 65%), radial-gradient(circle at 50% 50%, #0c1024, #070812 80%)",
            "--dgcv-table-shadow": "0 0 20px rgba(122, 60, 168, 0.45), 0 0 60px rgba(58, 138, 160, 0.25), inset 0 1px 0 rgba(238, 240, 250, 0.12)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.3s ease",
            "--dgcv-text-shadow": "0 0 10px rgba(182, 156, 240, 0.5)",
        },
    ),
    "dark_blue": ThemeConfig(
        bg_primary="#002b4f",
        bg_surface="#001f3f",
        bg_alt="#004080",
        bg_hover="#0059b3",
        text_main="#7fdbff",
        text_heading="#7fdbff",
        text_hover="#7fdbff",
        border_main="#7fdbff",
        bg_action="#0074d9",
        text_on_action="#ffffff",
        bg_action_hover="#39cccc",
        bg_error="#ff4136",
        text_on_error="#ffffff",
        bg_success="#2ecc40",
        text_on_success="#001f3f",
        border_alt="#0059b3",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "1px",
        },
    ),
    "dark_high_contrast": ThemeConfig(
        bg_primary="#1e1e1e",
        bg_surface="#000000",
        bg_alt="#333333",
        bg_hover="#4d4d4d",
        text_main="#ffffff",
        text_heading="#ffffff",
        text_hover="#ffffff",
        border_main="#ffffff",
        bg_action="#ffffff",
        text_on_action="#000000",
        bg_action_hover="#cccccc",
        bg_error="#ff0000",
        text_on_error="#ffffff",
        bg_success="#00ff00",
        text_on_success="#000000",
        border_alt="#666666",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "1px",
        },
    ),
    "dark_high_contrast_bright": ThemeConfig(
        bg_primary="#1e1e1e",
        bg_surface="#000000",
        bg_alt="#4d4d4d",
        bg_hover="#9370db",
        text_main="#ff69b4",
        text_heading="#00ffff",
        text_hover="#ffffff",
        border_main="#ffff00",
        bg_action="#00ffff",
        text_on_action="#000000",
        bg_action_hover="#ff00ff",
        bg_error="#ff0000",
        text_on_error="#ffffff",
        bg_success="#00ff00",
        text_on_success="#000000",
        border_alt="#ff00ff",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-table-shadow": "0 0 15px rgba(255, 105, 180, 0.5), 0 0 30px rgba(0, 255, 255, 0.3)",
            "--dgcv-text-shadow": "0 0 6px currentColor",
        },
    ),
    "dark_modern": ThemeConfig(
        bg_primary="#1c1c1c",
        bg_surface="#2c2c2c",
        bg_alt="#3a3a3a",
        bg_hover="#484848",
        text_main="#f5f5f5",
        text_heading="#f5f5f5",
        text_hover="#7fdbff",
        border_main="#444444",
        bg_action="#0074D9",
        text_on_action="#ffffff",
        bg_action_hover="#39CCCC",
        bg_error="#FF4136",
        text_on_error="#ffffff",
        bg_success="#2ECC40",
        text_on_success="#1c1c1c",
        border_alt="#555555",
    ),
    "dark_purple": ThemeConfig(
        bg_primary="#3b1a4f",
        bg_surface="#2c003e",
        bg_alt="#503a66",
        bg_hover="#4a3a57",
        text_main="#d3d3d3",
        text_heading="#e6e6e6",
        text_hover="#d3d3d3",
        border_main="#8c0099",
        bg_action="#8a2be2",
        text_on_action="#ffffff",
        bg_action_hover="#9932cc",
        bg_error="#ff1493",
        text_on_error="#ffffff",
        bg_success="#32cd32",
        text_on_success="#000000",
        border_alt="#68427d",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "1px",
        },
    ),
    "dessert": ThemeConfig(
        bg_primary="#ffebcd",
        bg_surface="#8b4513",
        bg_alt="#fffacd",
        bg_hover="#ffd700",
        text_main="#8b4513",
        text_heading="#ffffff",
        text_hover="#ffffff",
        border_main="#d2b48c",
        bg_action="#d2691e",
        text_on_action="#ffffff",
        bg_action_hover="#a0522d",
        bg_error="#cd5c5c",
        text_on_error="#ffffff",
        bg_success="#6b8e23",
        text_on_success="#ffffff",
        border_alt="#ffe4b5",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-image": "linear-gradient(45deg, #d2b48c, #ffd700, #d2b48c) 1",
            "--dgcv-border-radius": "0",
            "--dgcv-table-shadow": "0 4px 12px rgba(139, 69, 19, 0.2)",
            "--dgcv-hover-transform": "scale(1.001)",
        },
    ),
    "dusk_gradient": ThemeConfig(
        bg_primary="#0a080f",
        bg_surface="#130f18",
        bg_alt="#0e0b14",
        bg_hover="#cc7733",
        text_main="#ecddd4",
        text_heading="#d4a088",
        text_hover="#0a080f",
        text_alt="#c8b8cc",
        border_main="#6633aa",
        border_alt="#1e1428",
        bg_action="#6633aa",
        text_on_action="#ecddd4",
        bg_action_hover="#cc7733",
        bg_error="#cc2244",
        text_on_error="#ecddd4",
        bg_success="#2a7755",
        text_on_success="#ecddd4",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #0a080f 0%, #180a20 15%, #2a1040 28%, #4a1a30 40%, #7a2a20 52%, #aa5522 62%, #cc7733 72%, #ddaa55 80%, #180a20 90%, #0a080f 100%)",
            "--dgcv-table-shadow": "0 0 20px rgba(102, 51, 170, 0.5), 0 0 60px rgba(204, 119, 51, 0.2), inset 0 1px 0 rgba(220, 170, 120, 0.12)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.2s ease",
            "--dgcv-text-shadow": "0 0 12px rgba(180, 100, 60, 0.6)",
            "--dgcv-border-image": "linear-gradient(135deg, #6633aa, #aa3366, #cc7733, #ddaa55, #cc7733, #6633aa) 1",
            "--dgcv-border-radius": "0",
        },
    ),
    "fire_gradient": ThemeConfig(
        bg_primary="#0f0700",
        bg_surface="#1a0c00",
        bg_alt="#120900",
        bg_hover="#ff6600",
        text_main="#f5ddc8",
        text_heading="#ffaa55",
        text_hover="#0f0700",
        text_alt="#e0c0a0",
        border_main="#aa3300",
        border_alt="#2a1200",
        bg_action="#aa3300",
        text_on_action="#f5ddc8",
        bg_action_hover="#ff6600",
        bg_error="#ff2200",
        text_on_error="#0f0700",
        bg_success="#336644",
        text_on_success="#f5ddc8",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #0f0700 0%, #2a0a00 12%, #4a1500 24%, #6a2200 35%, #aa3300 48%, #cc5500 58%, #ff6600 68%, #ffaa00 78%, #ffcc44 86%, #0f0700 100%)",
            "--dgcv-table-shadow": "0 0 20px rgba(170, 51, 0, 0.6), 0 0 60px rgba(255, 102, 0, 0.2), inset 0 1px 0 rgba(255, 200, 100, 0.12)",
            "--plaque-fill": "#1a0c00",
            "--plaque-border": "#ffaa55",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.2s ease",
            "--dgcv-text-shadow": "0 0 12px rgba(255, 120, 0, 0.6)",
            "--dgcv-border-image": "linear-gradient(135deg, #ff2200, #ff6600, #ffaa00, #ffcc44, #ff6600, #ff2200) 1",
            "--dgcv-border-radius": "0",
        },
    ),
    "foggy_lights": ThemeConfig(
        bg_primary="#f8f9fa",
        bg_surface="#f8f9fa",
        bg_alt="#e9ecef",
        bg_hover="#b0c4de",
        text_main="#495057",
        text_heading="#343a40",
        text_hover="#212529",
        border_main="#778899",
        bg_action="#4682B4",
        text_on_action="#ffffff",
        bg_action_hover="#5F9EA0",
        bg_error="#DC143C",
        text_on_error="#ffffff",
        bg_success="#2E8B57",
        text_on_success="#ffffff",
        border_alt="#CED4DA",
        font_family="Verdana, sans-serif",
        custom_css_vars={
            "--dgcv-hover-transform": "scale(1.001)",
        },
    ),
    "forest_floor": ThemeConfig(
        bg_primary="#1b4332",
        bg_surface="#1b4332",
        bg_alt="#2d6a4f",
        bg_hover="#b7e4c7",
        text_main="#d8f3dc",
        text_heading="#d8f3dc",
        text_hover="#081c15",
        border_main="#6b705c",
        bg_action="#40916c",
        text_on_action="#ffffff",
        bg_action_hover="#52b788",
        bg_error="#d90429",
        text_on_error="#ffffff",
        bg_success="#74c69d",
        text_on_success="#081c15",
        border_alt="#40916c",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-special-background": "linear-gradient(170deg, #1b4332 0%, #2d6a4f 50%, #1b4332 100%)",
            "--dgcv-table-shadow": "0 0 16px rgba(27, 67, 50, 0.6)",
            "--dgcv-text-shadow": "0 1px 2px rgba(0,0,0,0.4)",
        },
    ),
    "gothic": ThemeConfig(
        bg_primary="#1c1c1c",
        bg_surface="#2c0033",
        bg_alt="#330033",
        bg_hover="#660000",
        text_main="#e6e6e6",
        text_heading="#a80000",
        text_hover="#e6e6e6",
        border_main="#a80000",
        bg_action="#800000",
        text_on_action="#ffffff",
        bg_action_hover="#4d0000",
        bg_error="#ff0000",
        text_on_error="#000000",
        bg_success="#2e8b57",
        text_on_success="#ffffff",
        border_alt="#4a004a",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-border-image": "linear-gradient(180deg, #a80000, #4a004a, #a80000) 1",
            "--dgcv-border-radius": "0",
            "--dgcv-table-shadow": "0 0 20px rgba(168, 0, 0, 0.5), inset 0 0 10px rgba(74, 0, 74, 0.3)",
            "--dgcv-text-shadow": "0 0 6px rgba(168, 0, 0, 0.6)",
            "--dgcv-hover-transform": "scale(1.001)",
        },
    ),
    "graffiti": ThemeConfig(
        bg_primary="#ff4500",
        bg_surface="#1e90ff",
        bg_alt="#ffa500",
        bg_hover="#39FF14",
        text_main="#ffffff",
        text_heading="#ffffff",
        text_hover="#000000",
        border_main="#ffa500",
        bg_action="#ff00ff",
        text_on_action="#ffffff",
        bg_action_hover="#da70d6",
        bg_error="#8b0000",
        text_on_error="#ffffff",
        bg_success="#00ff00",
        text_on_success="#000000",
        border_alt="#ff8c00",
        font_family="Permanent Marker, cursive",
        custom_css_vars={
            "--dgcv-border-width": "3px",
            "--dgcv-border-image": "linear-gradient(90deg, #ff4500, #ff00ff, #39FF14, #1e90ff, #ff4500) 1",
            "--dgcv-border-radius": "0",
            "--dgcv-table-shadow": "0 0 20px rgba(30, 144, 255, 0.7), 0 0 40px rgba(255, 0, 255, 0.4)",
            "--dgcv-text-shadow": "2px 2px 0px rgba(0,0,0,0.5)",
            "--dgcv-hover-transform": "rotate(0.5deg) scale(1.002)",
        },
    ),
    "graph_paper": ThemeConfig(
        bg_primary="#f2faff",
        bg_surface="#ffffff",
        bg_alt="#ffffff",
        bg_hover="#e6f7ff",
        text_main="#000000",
        text_heading="#000000",
        text_hover="#000000",
        border_main="#cccccc",
        bg_action="#0066cc",
        text_on_action="#ffffff",
        bg_action_hover="#3385ff",
        bg_error="#cc0000",
        text_on_error="#ffffff",
        bg_success="#009900",
        text_on_success="#ffffff",
        border_alt="#e6e6e6",
        font_family="Roboto, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
        },
    ),
    "gruvbox_colorful": ThemeConfig(
        bg_primary="#fbf1c7",
        bg_surface="#689d6a",
        bg_alt="#ebdbb2",
        bg_hover="#d3869b",
        text_main="#9d0006",
        text_heading="#fbf1c7",
        text_hover="#282828",
        text_alt="#076678",
        border_main="#427b58",
        bg_action="#d65d0e",
        text_on_action="#fbf1c7",
        bg_action_hover="#fe8019",
        bg_error="#cc241d",
        text_on_error="#fbf1c7",
        bg_success="#79740e",
        text_on_success="#fbf1c7",
        border_alt="#d5c4a1",
        font_family="Comic Sans MS, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-table-shadow": "0 0 10px rgba(0, 0, 0, 0.1)",
        },
    ),
    "gruvbox_dark": ThemeConfig(
        bg_primary="#282828",
        bg_surface="#1d2021",
        bg_alt="#32302f",
        bg_hover="#3c3836",
        text_main="#ebdbb2",
        text_alt="#83a598",
        text_heading="#fabd2f",
        text_hover="#8ec07c",
        border_main="#504945",
        bg_action="#458588",
        text_on_action="#ebdbb2",
        bg_action_hover="#83a598",
        bg_error="#cc241d",
        text_on_error="#ebdbb2",
        bg_success="#98971a",
        text_on_success="#1d2021",
        border_alt="#665c54",
        font_family="monospace",
    ),
    "gruvbox_light": ThemeConfig(
        bg_primary="#fbf1c7",
        bg_surface="#ebdbb2",
        bg_alt="#f2e5bc",
        bg_hover="#bdae93",
        text_main="#3c3836",
        text_heading="#9d0006",
        text_hover="#076678",
        border_main="#d5c4a1",
        bg_action="#076678",
        text_on_action="#fbf1c7",
        bg_action_hover="#458588",
        bg_error="#9d0006",
        text_on_error="#fbf1c7",
        bg_success="#79740e",
        text_on_success="#fbf1c7",
        border_alt="#e8d8b0",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-table-shadow": "0 0 10px rgba(0, 0, 0, 0.05)",
        },
    ),
    "gruvbox_reds": ThemeConfig(
        bg_primary="#fbf1c7",
        bg_surface="#af3a03",
        bg_alt="#d5c4a1",
        bg_hover="#fabd2f",
        text_main="#076678",
        text_heading="#fbf1c7",
        text_hover="#282828",
        text_alt="#9d0006",
        border_main="#d65d0e",
        bg_action="#8f3f71",
        text_on_action="#fbf1c7",
        bg_action_hover="#b16286",
        bg_error="#cc241d",
        text_on_error="#fbf1c7",
        bg_success="#79740e",
        text_on_success="#fbf1c7",
        border_alt="#bdae93",
        font_family="Comic Sans MS, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-table-shadow": "0 0 10px rgba(215, 153, 33, 0.4)",
        },
    ),
    "lunar": ThemeConfig(
        bg_primary="#dfe7ec",
        bg_surface="#394b59",
        bg_alt="#f5f8fa",
        bg_hover="#cbd6e2",
        text_main="#333333",
        text_heading="#b0c4de",
        text_hover="#333333",
        border_main="#394b59",
        bg_action="#5c7080",
        text_on_action="#ffffff",
        bg_action_hover="#738694",
        bg_error="#db3737",
        text_on_error="#ffffff",
        bg_success="#0f9960",
        text_on_success="#ffffff",
        border_alt="#e1e8ed",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "1px",
        },
    ),
    "matte_amber": ThemeConfig(
        bg_primary="#0f0c07",
        bg_surface="#181208",
        bg_alt="#120e06",
        bg_hover="#cc8822",
        text_main="#f0e4c8",
        text_heading="#e8c070",
        text_hover="#0f0c07",
        text_alt="#d8cc99",
        border_main="#7a5510",
        border_alt="#251a08",
        bg_action="#7a5510",
        text_on_action="#f0e4c8",
        bg_action_hover="#cc8822",
        bg_error="#cc3322",
        text_on_error="#f0e4c8",
        bg_success="#336644",
        text_on_success="#f0e4c8",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #0f0c07 0%, #201508 50%, #0f0c07 100%)",
            "--dgcv-table-shadow": "inset 0 1px 0 rgba(240, 200, 100, 0.05)",
            "--dgcv-hover-transition": "all 0.2s ease",
        },
    ),
    "matte_blue": ThemeConfig(
        bg_primary="#080c14",
        bg_surface="#0c1220",
        bg_alt="#0a0f1a",
        bg_hover="#4477cc",
        text_main="#d8e4f8",
        text_heading="#99bbee",
        text_hover="#080c14",
        text_alt="#b0c8e8",
        border_main="#1e3a6e",
        border_alt="#111a2e",
        bg_action="#1e3a6e",
        text_on_action="#d8e4f8",
        bg_action_hover="#4477cc",
        bg_error="#cc3344",
        text_on_error="#ffffff",
        bg_success="#2a8855",
        text_on_success="#d8e4f8",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #080c14 0%, #0c1a2e 50%, #080c14 100%)",
            "--dgcv-table-shadow": "inset 0 1px 0 rgba(100, 160, 255, 0.05)",
            "--dgcv-hover-transition": "all 0.2s ease",
        },
    ),
    "matte_cyan": ThemeConfig(
        bg_primary="#070f0f",
        bg_surface="#0a1818",
        bg_alt="#081212",
        bg_hover="#1aaa99",
        text_main="#cceee8",
        text_heading="#55ccbb",
        text_hover="#070f0f",
        text_alt="#99ddd4",
        border_main="#136655",
        border_alt="#0f2020",
        bg_action="#136655",
        text_on_action="#cceee8",
        bg_action_hover="#1aaa99",
        bg_error="#cc3344",
        text_on_error="#cceee8",
        bg_success="#1aaa99",
        text_on_success="#070f0f",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #070f0f 0%, #0c2020 50%, #070f0f 100%)",
            "--dgcv-table-shadow": "inset 0 1px 0 rgba(85, 220, 200, 0.05)",
            "--dgcv-hover-transition": "all 0.2s ease",
        },
    ),
    "matte_green": ThemeConfig(
        bg_primary="#070f0a",
        bg_surface="#0a1810",
        bg_alt="#081209",
        bg_hover="#22aa55",
        text_main="#d0eedd",
        text_heading="#66cc88",
        text_hover="#070f0a",
        text_alt="#a8d8bb",
        border_main="#1a5530",
        border_alt="#102018",
        bg_action="#1a5530",
        text_on_action="#d0eedd",
        bg_action_hover="#22aa55",
        bg_error="#cc3322",
        text_on_error="#d0eedd",
        bg_success="#22aa55",
        text_on_success="#070f0a",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #070f0a 0%, #0c2015 50%, #070f0a 100%)",
            "--dgcv-table-shadow": "inset 0 1px 0 rgba(100, 220, 140, 0.05)",
            "--dgcv-hover-transition": "all 0.2s ease",
        },
    ),
    "matte_red": ThemeConfig(
        bg_primary="#0f0907",
        bg_surface="#180f0a",
        bg_alt="#120b08",
        bg_hover="#cc4422",
        text_main="#f0ddd8",
        text_heading="#e8a090",
        text_hover="#0f0907",
        text_alt="#d8c0b8",
        border_main="#7a2a1a",
        border_alt="#2a1510",
        bg_action="#7a2a1a",
        text_on_action="#f0ddd8",
        bg_action_hover="#cc4422",
        bg_error="#ff4422",
        text_on_error="#0f0907",
        bg_success="#336644",
        text_on_success="#f0ddd8",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #0f0907 0%, #200e08 50%, #0f0907 100%)",
            "--dgcv-table-shadow": "inset 0 1px 0 rgba(240, 160, 120, 0.05)",
            "--dgcv-hover-transition": "all 0.2s ease",
        },
    ),
    "matte_slate": ThemeConfig(
        bg_primary="#090a0c",
        bg_surface="#111318",
        bg_alt="#0c0e12",
        bg_hover="#5577aa",
        text_main="#d8dde8",
        text_heading="#99aabf",
        text_hover="#090a0c",
        text_alt="#b0bcc8",
        border_main="#2a3344",
        border_alt="#181c24",
        bg_action="#2a3344",
        text_on_action="#d8dde8",
        bg_action_hover="#5577aa",
        bg_error="#cc3344",
        text_on_error="#d8dde8",
        bg_success="#2a6644",
        text_on_success="#d8dde8",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #090a0c 0%, #111820 50%, #090a0c 100%)",
            "--dgcv-table-shadow": "inset 0 1px 0 rgba(150, 180, 220, 0.05)",
            "--dgcv-hover-transition": "all 0.2s ease",
        },
    ),
    "matte_slate_soft": ThemeConfig(
        bg_primary="#111214",
        bg_surface="#18191d",
        bg_alt="#141518",
        bg_hover="#5577aa",
        text_main="#c8cdd6",
        text_heading="#c8cdd6",
        text_hover="#111214",
        text_alt="#8892a0",
        border_main="#2a2d34",
        border_alt="#1e2026",
        bg_action="#2a2d34",
        text_on_action="#c8cdd6",
        bg_action_hover="#5577aa",
        bg_error="#cc3344",
        text_on_error="#c8cdd6",
        bg_success="#2a6644",
        text_on_success="#c8cdd6",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #18191d 0%, #1c1e24 50%, #18191d 100%)",
            "--dgcv-table-shadow": "inset 0 1px 0 rgba(150, 180, 220, 0.04)",
            "--dgcv-hover-transition": "all 0.2s ease",
        },
    ),
    "matte_violet": ThemeConfig(
        bg_primary="#0a0a0f",
        bg_surface="#100e18",
        bg_alt="#0d0b14",
        bg_hover="#9966ff",
        text_main="#e0d8f0",
        text_heading="#c8b8f0",
        text_hover="#0a0a0f",
        text_alt="#c0b0e8",
        border_main="#5533aa",
        border_alt="#1a1428",
        bg_action="#5533aa",
        text_on_action="#ffffff",
        bg_action_hover="#7744cc",
        bg_error="#cc2244",
        text_on_error="#ffffff",
        bg_success="#44aa77",
        text_on_success="#0a0a0f",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #0a0a0f 0%, #120a20 40%, #1a1030 70%, #0a0a0f 100%)",
            "--dgcv-table-shadow": "inset 0 1px 0 rgba(180, 140, 255, 0.05)",
            "--dgcv-hover-transition": "all 0.2s ease",
        },
    ),
    "matisse": ThemeConfig(
        bg_primary="#b2d5eb",
        bg_surface="#ee9e84",
        bg_alt="#41b199",
        bg_hover="#df7d8c",
        text_main="#d45659",
        text_heading="#004d00",
        text_alt="#ffd6cc",
        text_hover="#b2d5eb",
        border_main="#ee9e84",
        bg_action="#d45659",
        text_on_action="#ffffff",
        bg_action_hover="#c0392b",
        bg_error="#900C3F",
        text_on_error="#ffffff",
        bg_success="#27ae60",
        text_on_success="#ffffff",
        border_alt="#318c78",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-table-shadow": "0 4px 12px rgba(238, 158, 132, 0.4)",
            "--dgcv-hover-transform": "scale(1.001)",
        },
    ),
    "matisse_light": ThemeConfig(
        bg_primary="#e6ffe6",
        bg_surface="#ffcc00",
        bg_alt="#ffd6cc",
        bg_hover="#ffe680",
        text_main="#004d00",
        text_heading="#004d00",
        text_hover="#004d00",
        border_main="#004d00",
        bg_action="#008000",
        text_on_action="#ffffff",
        bg_action_hover="#009900",
        bg_error="#ff3333",
        text_on_error="#ffffff",
        bg_success="#00cc00",
        text_on_success="#ffffff",
        border_alt="#ffb399",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-table-shadow": "0 2px 8px rgba(255, 204, 0, 0.3)",
        },
    ),
    "mist": ThemeConfig(
        bg_primary="#e0f7fa",
        bg_surface="#00796b",
        bg_alt="#b2ebf2",
        bg_hover="#a7c4c7",
        text_main="#004d40",
        text_heading="#ffffff",
        text_hover="#004d40",
        border_main="#b2dfdb",
        bg_action="#00838f",
        text_on_action="#ffffff",
        bg_action_hover="#00bcd4",
        bg_error="#c62828",
        text_on_error="#ffffff",
        bg_success="#2e7d32",
        text_on_success="#ffffff",
        border_alt="#80deea",
        font_family="Cormorant, serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-special-background": "linear-gradient(170deg, #e0f7fa 0%, #b2ebf2 60%, #e0f7fa 100%)",
            "--dgcv-special-text": "#004d40",
            "--dgcv-table-shadow": "0 0 20px rgba(0, 121, 107, 0.4)",
            "--dgcv-hover-transform": "scale(1.001)",
        },
    ),
    "monet": ThemeConfig(
        bg_primary="#f7f9e4",
        bg_surface="#2a5d67",
        bg_alt="#c6f3d8",
        bg_hover="#d0ece7",
        text_main="#2a5d67",
        text_heading="#ffffff",
        text_hover="#2a5d67",
        border_main="#2a5d67",
        bg_action="#1a3d44",
        text_on_action="#ffffff",
        bg_action_hover="#3b7a86",
        bg_error="#d9534f",
        text_on_error="#ffffff",
        bg_success="#5cb85c",
        text_on_success="#ffffff",
        border_alt="#b2dfcc",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-table-shadow": "0 2px 8px rgba(42, 93, 103, 0.2)",
            "--dgcv-hover-transform": "scale(1.001)",
        },
    ),
    "neutral_dark": ThemeConfig(
        bg_primary="#121212",
        bg_surface="#121212",
        bg_alt="#1e1e1e",
        bg_hover="#252525",
        text_main="#e0e0e0",
        text_heading="#ffffff",
        text_hover="#ffffff",
        border_main="#333333",
        bg_action="#0288d1",
        text_on_action="#ffffff",
        bg_action_hover="#03a9f4",
        bg_error="#cf6679",
        text_on_error="#000000",
        bg_success="#81c784",
        text_on_success="#000000",
        border_alt="#424242",
        font_family="system-ui, -apple-system, sans-serif",
        custom_css_vars={
            "--dgcv-hover-transform": "scale(1.001)",
        },
    ),
    "newspaper": ThemeConfig(
        bg_primary="#f4f4f4",
        bg_surface="#fafafa",
        bg_alt="#fafafa",
        bg_hover="#e0e0e0",
        text_main="#000000",
        text_heading="#000000",
        text_hover="#000000",
        border_main="#333333",
        bg_action="#000000",
        text_on_action="#ffffff",
        bg_action_hover="#4d4d4d",
        bg_error="#8b0000",
        text_on_error="#ffffff",
        bg_success="#006400",
        text_on_success="#ffffff",
        border_alt="#808080",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-table-shadow": "0px 2px 5px rgba(0, 0, 0, 0.1)",
        },
    ),
    "ocean": ThemeConfig(
        bg_primary="#4682b4",
        bg_surface="#1e90ff",
        bg_alt="#87cefa",
        bg_hover="#1e90ff",
        text_main="#ffffff",
        text_heading="#ffffff",
        text_hover="#ffffff",
        border_main="#1e90ff",
        bg_action="#000080",
        text_on_action="#ffffff",
        bg_action_hover="#0000cd",
        bg_error="#dc143c",
        text_on_error="#ffffff",
        bg_success="#20b2aa",
        text_on_success="#ffffff",
        border_alt="#5f9ea0",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-image": "linear-gradient(90deg, #00bfff, #1e90ff, #4682b4) 1",
            "--dgcv-border-radius": "0",
            "--dgcv-text-shadow": "0px 0px 4px rgba(0, 191, 255, 0.6)",
            "--dgcv-table-shadow": "0 4px 12px rgba(30, 144, 255, 0.6)",
            "--dgcv-hover-transform": "scale(1.001)",
        },
    ),
    "ocean_gradient": ThemeConfig(
        bg_primary="#020d1a",
        bg_surface="#041424",
        bg_alt="#041020",
        bg_hover="#00e5ff",
        text_main="#b0eaf5",
        text_heading="#b0eaf5",
        text_hover="#020d1a",
        text_alt="#80d0e8",
        border_main="#0077a8",
        border_alt="#0a2a3a",
        bg_action="#006994",
        text_on_action="#ffffff",
        bg_action_hover="#0090c8",
        bg_error="#ff4b2b",
        text_on_error="#ffffff",
        bg_success="#00e676",
        text_on_success="#020d1a",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "radial-gradient(ellipse at top, #0a2a4a 0%, #041424 40%, #020810 100%)",
            "--dgcv-table-shadow": "0 0 30px rgba(0, 119, 168, 0.5), 0 0 80px rgba(0, 229, 255, 0.15)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.3s ease",
            "--dgcv-text-shadow": "0 0 10px rgba(0, 229, 255, 0.5)",
        },
    ),
    "oil": ThemeConfig(
        bg_primary="#0a0a0f",
        bg_surface="#12101a",
        bg_alt="#0d0d14",
        bg_hover="#9c84b8",
        text_main="#e8e0f0",
        text_heading="#e8e0f0",
        text_hover="#0a0a0f",
        text_alt="#d0c8e8",
        border_main="#6644aa",
        border_alt="#1a1428",
        bg_action="#2244cc",
        text_on_action="#ffffff",
        bg_action_hover="#88b0b4",
        bg_error="#ff4b2b",
        text_on_error="#ffffff",
        bg_success="#00e676",
        text_on_success="#0a0a0f",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #0a0a0f 0%, #1a0a2e 15%, #0a1a3a 30%, #0a2a1a 45%, #1a2a0a 55%, #2a1a0a 65%, #2a0a1a 75%, #1a0a2e 88%, #0a0a0f 100%)",
            "--dgcv-table-shadow": "0 0 20px rgba(102, 68, 170, 0.6), 0 0 60px rgba(34, 68, 204, 0.3), inset 0 1px 0 rgba(200, 150, 255, 0.15)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.3s ease",
            "--dgcv-text-shadow": "0 0 12px rgba(180, 120, 255, 0.7)",
            "--dgcv-border-image": "linear-gradient(135deg, #ff00ff, #0044ff, #00ffcc, #ffaa00, #ff00ff) 1",
            "--dgcv-border-radius": "0",
        },
    ),
    "oil_amber": ThemeConfig(
        bg_primary="#0f0c07",
        bg_surface="#181208",
        bg_alt="#120e06",
        bg_hover="#cc8822",
        text_main="#f0e4c8",
        text_heading="#e8c070",
        text_hover="#0f0c07",
        text_alt="#d8cc99",
        border_main="#7a5510",
        border_alt="#251a08",
        bg_action="#7a5510",
        text_on_action="#f0e4c8",
        bg_action_hover="#cc8822",
        bg_error="#cc3322",
        text_on_error="#f0e4c8",
        bg_success="#336644",
        text_on_success="#f0e4c8",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #0f0c07 0%, #201508 50%, #0f0c07 100%)",
            "--dgcv-table-shadow": "0 0 20px rgba(122, 85, 16, 0.6), inset 0 1px 0 rgba(240, 200, 100, 0.08)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.3s ease",
            "--dgcv-text-shadow": "0 0 10px rgba(204, 136, 34, 0.5)",
            "--dgcv-border-image": "linear-gradient(135deg, #cc8822, #7a5510, #cc8822) 1",
            "--dgcv-border-radius": "0",
        },
    ),
    "oil_blue": ThemeConfig(
        bg_primary="#080c14",
        bg_surface="#0c1220",
        bg_alt="#0a0f1a",
        bg_hover="#4477cc",
        text_main="#d8e4f8",
        text_heading="#99bbee",
        text_hover="#080c14",
        text_alt="#b0c8e8",
        border_main="#1e3a6e",
        border_alt="#111a2e",
        bg_action="#1e3a6e",
        text_on_action="#d8e4f8",
        bg_action_hover="#4477cc",
        bg_error="#cc3344",
        text_on_error="#ffffff",
        bg_success="#2a8855",
        text_on_success="#d8e4f8",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #080c14 0%, #0c1a2e 50%, #080c14 100%)",
            "--dgcv-table-shadow": "0 0 20px rgba(30, 58, 110, 0.6), inset 0 1px 0 rgba(100, 160, 255, 0.1)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.3s ease",
            "--dgcv-text-shadow": "0 0 10px rgba(68, 119, 204, 0.5)",
            "--dgcv-border-image": "linear-gradient(135deg, #4477cc, #1e3a6e, #4477cc) 1",
            "--dgcv-border-radius": "0",
        },
    ),
    "oil_cyan": ThemeConfig(
        bg_primary="#070f0f",
        bg_surface="#0a1818",
        bg_alt="#081212",
        bg_hover="#1aaa99",
        text_main="#cceee8",
        text_heading="#55ccbb",
        text_hover="#070f0f",
        text_alt="#99ddd4",
        border_main="#136655",
        border_alt="#0f2020",
        bg_action="#136655",
        text_on_action="#cceee8",
        bg_action_hover="#1aaa99",
        bg_error="#cc3344",
        text_on_error="#cceee8",
        bg_success="#1aaa99",
        text_on_success="#070f0f",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #070f0f 0%, #0c2020 50%, #070f0f 100%)",
            "--dgcv-table-shadow": "0 0 20px rgba(19, 102, 85, 0.6), inset 0 1px 0 rgba(85, 220, 200, 0.08)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.3s ease",
            "--dgcv-text-shadow": "0 0 10px rgba(26, 170, 153, 0.5)",
            "--dgcv-border-image": "linear-gradient(135deg, #1aaa99, #136655, #1aaa99) 1",
            "--dgcv-border-radius": "0",
        },
    ),
    "oil_green": ThemeConfig(
        bg_primary="#070f0a",
        bg_surface="#0a1810",
        bg_alt="#081209",
        bg_hover="#22aa55",
        text_main="#d0eedd",
        text_heading="#66cc88",
        text_hover="#070f0a",
        text_alt="#a8d8bb",
        border_main="#1a5530",
        border_alt="#102018",
        bg_action="#1a5530",
        text_on_action="#d0eedd",
        bg_action_hover="#22aa55",
        bg_error="#cc3322",
        text_on_error="#d0eedd",
        bg_success="#22aa55",
        text_on_success="#070f0a",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #070f0a 0%, #0c2015 50%, #070f0a 100%)",
            "--dgcv-table-shadow": "0 0 20px rgba(26, 85, 48, 0.6), inset 0 1px 0 rgba(100, 220, 140, 0.08)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.3s ease",
            "--dgcv-text-shadow": "0 0 10px rgba(34, 170, 85, 0.5)",
            "--dgcv-border-image": "linear-gradient(135deg, #22aa55, #1a5530, #22aa55) 1",
            "--dgcv-border-radius": "0",
        },
    ),
    "oil_red": ThemeConfig(
        bg_primary="#0f0907",
        bg_surface="#180f0a",
        bg_alt="#120b08",
        bg_hover="#cc4422",
        text_main="#f0ddd8",
        text_heading="#e8a090",
        text_hover="#0f0907",
        text_alt="#d8c0b8",
        border_main="#7a2a1a",
        border_alt="#2a1510",
        bg_action="#7a2a1a",
        text_on_action="#f0ddd8",
        bg_action_hover="#cc4422",
        bg_error="#ff4422",
        text_on_error="#0f0907",
        bg_success="#336644",
        text_on_success="#f0ddd8",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #0f0907 0%, #200e08 50%, #0f0907 100%)",
            "--dgcv-table-shadow": "0 0 20px rgba(122, 42, 26, 0.6), inset 0 1px 0 rgba(240, 160, 120, 0.08)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.3s ease",
            "--dgcv-text-shadow": "0 0 10px rgba(204, 68, 34, 0.5)",
            "--dgcv-border-image": "linear-gradient(135deg, #cc4422, #7a2a1a, #cc4422) 1",
            "--dgcv-border-radius": "0",
        },
    ),
    "oil_slate": ThemeConfig(
        bg_primary="#090a0c",
        bg_surface="#111318",
        bg_alt="#0c0e12",
        bg_hover="#5577aa",
        text_main="#d8dde8",
        text_heading="#99aabf",
        text_hover="#090a0c",
        text_alt="#b0bcc8",
        border_main="#2a3344",
        border_alt="#181c24",
        bg_action="#2a3344",
        text_on_action="#d8dde8",
        bg_action_hover="#5577aa",
        bg_error="#cc3344",
        text_on_error="#d8dde8",
        bg_success="#2a6644",
        text_on_success="#d8dde8",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #090a0c 0%, #111820 50%, #090a0c 100%)",
            "--dgcv-table-shadow": "0 0 20px rgba(42, 51, 68, 0.7), inset 0 1px 0 rgba(150, 180, 220, 0.08)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.3s ease",
            "--dgcv-text-shadow": "0 0 10px rgba(85, 119, 170, 0.4)",
            "--dgcv-border-image": "linear-gradient(135deg, #5577aa, #2a3344, #5577aa) 1",
            "--dgcv-border-radius": "0",
        },
    ),
    "oil_violet": ThemeConfig(
        bg_primary="#0a0a0f",
        bg_surface="#100e18",
        bg_alt="#0d0b14",
        bg_hover="#9966ff",
        text_main="#e0d8f0",
        text_heading="#c8b8f0",
        text_hover="#0a0a0f",
        text_alt="#c0b0e8",
        border_main="#5533aa",
        border_alt="#1a1428",
        bg_action="#5533aa",
        text_on_action="#ffffff",
        bg_action_hover="#7744cc",
        bg_error="#cc2244",
        text_on_error="#ffffff",
        bg_success="#44aa77",
        text_on_success="#0a0a0f",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #0a0a0f 0%, #120a20 40%, #1a1030 70%, #0a0a0f 100%)",
            "--dgcv-table-shadow": "0 0 20px rgba(85, 51, 170, 0.5), inset 0 1px 0 rgba(180, 140, 255, 0.1)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.3s ease",
            "--dgcv-text-shadow": "0 0 10px rgba(153, 102, 255, 0.6)",
            "--dgcv-border-image": "linear-gradient(135deg, #9966ff, #5533aa, #9966ff) 1",
            "--dgcv-border-radius": "0",
        },
    ),
    "outer_space": ThemeConfig(
        bg_primary="#1d1d1d",
        bg_surface="#2b2d42",
        bg_alt="#333366",
        bg_hover="#4b0082",
        text_main="#ffffff",
        text_heading="#ffffff",
        text_hover="#ffffff",
        border_main="#6c63ff",
        bg_action="#6c63ff",
        text_on_action="#ffffff",
        bg_action_hover="#8c82ff",
        bg_error="#ff4d4d",
        text_on_error="#ffffff",
        bg_success="#00e676",
        text_on_success="#1d1d1d",
        border_alt="#404080",
        font_family="Orbitron, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-special-background": "radial-gradient(ellipse at top, #2b2d42 0%, #1d1d1d 60%, #0a0a14 100%)",
            "--dgcv-table-shadow": "0 4px 20px rgba(75, 0, 130, 0.8), 0 0 60px rgba(108, 99, 255, 0.2)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-text-shadow": "0 0 8px rgba(108, 99, 255, 0.8)",
        },
    ),
    "paper_graphite": ThemeConfig(
        bg_primary="#f5f4f2",
        bg_surface="#eceae6",
        bg_alt="#f0eeeb",
        bg_hover="#6b7280",
        text_main="#1c1c1e",
        text_heading="#2c2c30",
        text_hover="#f5f4f2",
        text_alt="#4a4a50",
        border_main="#b0aaa0",
        border_alt="#d8d4ce",
        bg_action="#4a4a50",
        text_on_action="#f5f4f2",
        bg_action_hover="#6b7280",
        bg_error="#cc3344",
        text_on_error="#f5f4f2",
        bg_success="#3a6644",
        text_on_success="#f5f4f2",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #eceae6 0%, #e4e0da 50%, #eceae6 100%)",
            "--dgcv-table-shadow": "none",
            "--dgcv-hover-transition": "all 0.2s ease",
            "--dgcv-text-shadow": "none",
        },
    ),
    "paper_ivory": ThemeConfig(
        bg_primary="#faf7f0",
        bg_surface="#f0ebe0",
        bg_alt="#f5f1e8",
        bg_hover="#7a6a50",
        text_main="#1e1a14",
        text_heading="#2e2618",
        text_hover="#faf7f0",
        text_alt="#4a4030",
        border_main="#c8b89a",
        border_alt="#e0d8c8",
        bg_action="#5a4a32",
        text_on_action="#faf7f0",
        bg_action_hover="#7a6a50",
        bg_error="#cc3344",
        text_on_error="#faf7f0",
        bg_success="#3a6644",
        text_on_success="#faf7f0",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #f0ebe0 0%, #e8e0d0 50%, #f0ebe0 100%)",
            "--dgcv-table-shadow": "none",
            "--dgcv-hover-transition": "all 0.2s ease",
            "--dgcv-text-shadow": "none",
        },
    ),
    "paper_sage": ThemeConfig(
        bg_primary="#f2f5f0",
        bg_surface="#e4ece0",
        bg_alt="#eaf0e6",
        bg_hover="#4a6a50",
        text_main="#141c12",
        text_heading="#1e2e1a",
        text_hover="#f2f5f0",
        text_alt="#384a34",
        border_main="#90aa88",
        border_alt="#c8d8c0",
        bg_action="#304a30",
        text_on_action="#f2f5f0",
        bg_action_hover="#4a6a50",
        bg_error="#cc3344",
        text_on_error="#f2f5f0",
        bg_success="#3a6644",
        text_on_success="#f2f5f0",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #e4ece0 0%, #d8e8d0 50%, #e4ece0 100%)",
            "--dgcv-table-shadow": "none",
            "--dgcv-hover-transition": "all 0.2s ease",
            "--dgcv-text-shadow": "none",
        },
    ),
    "paper_slate": ThemeConfig(
        bg_primary="#f2f4f7",
        bg_surface="#e4e8f0",
        bg_alt="#eaecf2",
        bg_hover="#4a6080",
        text_main="#141820",
        text_heading="#202838",
        text_hover="#f2f4f7",
        text_alt="#384458",
        border_main="#9aaac0",
        border_alt="#ccd4e0",
        bg_action="#304460",
        text_on_action="#f2f4f7",
        bg_action_hover="#4a6080",
        bg_error="#cc3344",
        text_on_error="#f2f4f7",
        bg_success="#3a6644",
        text_on_success="#f2f4f7",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #e4e8f0 0%, #d8dfec 50%, #e4e8f0 100%)",
            "--dgcv-table-shadow": "none",
            "--dgcv-hover-transition": "all 0.2s ease",
            "--dgcv-text-shadow": "none",
        },
    ),
    "paper_white": ThemeConfig(
        bg_primary="#fafafa",
        bg_surface="#f0f0f0",
        bg_alt="#f5f5f5",
        bg_hover="#444444",
        text_main="#111111",
        text_heading="#222222",
        text_hover="#fafafa",
        text_alt="#555555",
        border_main="#cccccc",
        border_alt="#e4e4e4",
        bg_action="#333333",
        text_on_action="#fafafa",
        bg_action_hover="#444444",
        bg_error="#cc3344",
        text_on_error="#fafafa",
        bg_success="#3a6644",
        text_on_success="#fafafa",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(135deg, #f0f0f0 0%, #e8e8e8 50%, #f0f0f0 100%)",
            "--dgcv-table-shadow": "none",
            "--dgcv-hover-transition": "all 0.2s ease",
            "--dgcv-text-shadow": "none",
        },
    ),
    "parchment": ThemeConfig(
        bg_primary="#f1e5c9",
        bg_surface="#e7d9b8",
        bg_alt="#f5ecd6",
        bg_hover="#e0d0a8",
        text_main="#4a3320",
        text_heading="#3a2716",
        text_hover="#2e1f12",
        text_alt="#6e5638",
        border_main="#6b4f2e",
        border_alt="#cdb98c",
        bg_action="#8b5a2b",
        text_on_action="#ffffff",
        bg_action_hover="#a0522d",
        bg_error="#8b0000",
        text_on_error="#ffffff",
        bg_success="#556b2f",
        text_on_success="#ffffff",
        font_family="Dancing Script, cursive",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-special-text": "#4a3320",
            "--dgcv-special-background": 'url("data:image/png;base64,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"), #ece0c0',
            "--dgcv-table-shadow": "0 0 14px rgba(92, 67, 39, 0.35), inset 0 0 30px rgba(92, 67, 39, 0.06)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-text-shadow": "0 1px 0 rgba(255, 255, 255, 0.4)",
        },
    ),
    "picasso_blue": ThemeConfig(
        bg_primary="#eaf2f8",
        bg_surface="#154360",
        bg_alt="#d6eaf8",
        bg_hover="#aed6f1",
        text_main="#154360",
        text_heading="#ffffff",
        text_hover="#154360",
        border_main="#154360",
        bg_action="#1f618d",
        text_on_action="#ffffff",
        bg_action_hover="#2980b9",
        bg_error="#c0392b",
        text_on_error="#ffffff",
        bg_success="#27ae60",
        text_on_success="#ffffff",
        border_alt="#85c1e9",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-table-shadow": "0 2px 10px rgba(21, 67, 96, 0.2)",
            "--dgcv-hover-transform": "scale(1.001)",
        },
    ),
    "presentation": ThemeConfig(
        bg_primary="#ffffff",
        bg_surface="#007acc",
        bg_alt="#e0f4ff",
        bg_hover="#cce5ff",
        text_main="#000000",
        text_heading="#ffffff",
        text_hover="#000000",
        border_main="#007acc",
        bg_action="#005c99",
        text_on_action="#ffffff",
        bg_action_hover="#004080",
        bg_error="#cc0000",
        text_on_error="#ffffff",
        bg_success="#009933",
        text_on_success="#ffffff",
        border_alt="#99ccff",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "1px",
        },
    ),
    "purple_black_grad": ThemeConfig(
        bg_primary="#161312",
        bg_surface="#272020",
        bg_alt="#322a29",
        bg_hover="#5632cc",
        text_main="#f3f1f0",
        text_heading="#a1ffce",
        text_hover="#ffffff",
        text_alt="#f4f0ef",
        border_main="#895f5b",
        border_alt="#978280",
        bg_action="#5632cc",
        text_on_action="#ffffff",
        bg_action_hover="#4428a3",
        bg_error="#ff4b2b",
        text_on_error="#ffffff",
        bg_success="#00ff87",
        text_on_success="#000000",
        font_family="Impact, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-hover-transform": "rotate(1deg) scale(1.01)",
            "--dgcv-hover-transition": "all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275)",
            "--dgcv-special-background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            "--dgcv-table-shadow": "0 10px 30px rgba(0,0,0,0.5)",
        },
    ),
    "purple_green_grad": ThemeConfig(
        bg_primary="#141216",
        bg_surface="#242027",
        bg_alt="#2f2932",
        bg_hover="#b5cc32",
        text_main="#f2f0f3",
        text_heading="#a1ffce",
        text_hover="#ffffff",
        text_alt="#f3eff4",
        border_main="#7a5b89",
        border_alt="#908097",
        bg_action="#b5cc32",
        text_on_action="#141216",
        bg_action_hover="#91a328",
        bg_error="#ff4b2b",
        text_on_error="#ffffff",
        bg_success="#00ff87",
        text_on_success="#000000",
        font_family="Impact, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-hover-transform": "rotate(1deg) scale(1.01)",
            "--dgcv-hover-transition": "all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275)",
            "--dgcv-special-background": "linear-gradient(135deg, #2d4a1e 0%, #4a3d6b 50%, #1e2d4a 100%)",
            "--dgcv-table-shadow": "0 10px 30px rgba(0,0,0,0.5)",
        },
    ),
    "purples": ThemeConfig(
        bg_primary="#2D2B55",
        bg_surface="#3F3F7A",
        bg_alt="#4A4A8A",
        bg_hover="#6A5ACD",
        text_main="#F5F5F5",
        text_heading="#E6E6FA",
        text_hover="#F5F5F5",
        border_main="#6A5ACD",
        bg_action="#8A2BE2",
        text_on_action="#ffffff",
        bg_action_hover="#9370DB",
        bg_error="#FF1493",
        text_on_error="#ffffff",
        bg_success="#00FA9A",
        text_on_success="#2D2B55",
        border_alt="#7B68EE",
        font_family="Impact, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-special-background": "linear-gradient(160deg, #2D2B55 0%, #3F3F7A 50%, #2D2B55 100%)",
            "--dgcv-table-shadow": "0 0 20px rgba(106, 90, 205, 0.7), 0 0 60px rgba(138, 43, 226, 0.3)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-text-shadow": "0 0 8px rgba(106, 90, 205, 0.5)",
        },
    ),
    "radial_blues": ThemeConfig(
        bg_primary="#1a1a2e",
        bg_surface="#2c2c54",
        bg_alt="#0f3460",
        bg_hover="#22a6b3",
        text_main="#e0e0e0",
        text_heading="#ffffff",
        text_hover="#ffffff",
        border_main="#ffffff",
        bg_action="#e94560",
        text_on_action="#ffffff",
        bg_action_hover="#ff6b81",
        bg_error="#ff4757",
        text_on_error="#ffffff",
        bg_success="#2ed573",
        text_on_success="#1a1a2e",
        border_alt="#16213e",
        font_family="Trebuchet MS, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "radial-gradient(circle, #1a1a2e, #0f3460)",
            "--dgcv-table-shadow": "0 0 10px rgba(34, 166, 179, 0.8)",
        },
    ),
    "rain_forest": ThemeConfig(
        bg_primary="#90ee90",
        bg_surface="#228b22",
        bg_alt="#98fb98",
        bg_hover="#006400",
        text_main="#2f4f4f",
        text_heading="#ffffff",
        text_hover="#ffffff",
        border_main="#228b22",
        bg_action="#008000",
        text_on_action="#ffffff",
        bg_action_hover="#006400",
        bg_error="#b22222",
        text_on_error="#ffffff",
        bg_success="#32cd32",
        text_on_success="#000000",
        border_alt="#3cb371",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-image": "linear-gradient(45deg, #006400, #32cd32, #228b22) 1",
            "--dgcv-border-radius": "0",
            "--dgcv-special-background": "linear-gradient(170deg, #90ee90 0%, #98fb98 50%, #90ee90 100%)",
            "--dgcv-special-text": "#2f4f4f",
            "--dgcv-text-shadow": "0px 0px 4px rgba(0, 100, 0, 0.4)",
            "--dgcv-table-shadow": "0 4px 12px rgba(0, 100, 0, 0.5)",
            "--dgcv-hover-transform": "scale(1.001)",
        },
    ),
    "rembrandt": ThemeConfig(
        bg_primary="#fffaf0",
        bg_surface="#523d2e",
        bg_alt="#f7e6d5",
        bg_hover="#d4b996",
        text_main="#523d2e",
        text_heading="#ffffff",
        text_hover="#523d2e",
        border_main="#6c4f3d",
        bg_action="#38291f",
        text_on_action="#ffffff",
        bg_action_hover="#6c4f3d",
        bg_error="#a52a2a",
        text_on_error="#ffffff",
        bg_success="#556b2f",
        text_on_success="#ffffff",
        border_alt="#ebd4b3",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-special-background": "linear-gradient(170deg, #fffaf0 0%, #f7e6d5 60%, #fffaf0 100%)",
            "--dgcv-special-text": "#523d2e",
            "--dgcv-table-shadow": "0 4px 16px rgba(82, 61, 46, 0.3), inset 0 0 40px rgba(82, 61, 46, 0.05)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-text-shadow": "0 1px 0 rgba(255,255,255,0.5)",
        },
    ),
    "sakura": ThemeConfig(
        bg_primary="#f8f1ea",
        bg_surface="#b0476e",
        bg_alt="#f2e7db",
        bg_hover="#f3c4ce",
        text_main="#4e3228",
        text_heading="#ffffff",
        text_hover="#4e3228",
        text_alt="#6e4a3a",
        border_main="#c0986e",
        border_alt="#e0c8ac",
        bg_action="#b0506c",
        text_on_action="#ffffff",
        bg_action_hover="#b85470",
        bg_error="#be3a2c",
        text_on_error="#ffffff",
        bg_success="#4a7c3e",
        text_on_success="#ffffff",
        font_family="Palatino, serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-special-text": "#ffffff",
            "--dgcv-special-background": "linear-gradient(135deg, #46291f 0%, #5a382a 28%, #6e4636 50%, #7a4a42 72%, #8a5848 100%)",
            "--dgcv-table-shadow": "0 0 18px rgba(110, 70, 52, 0.22), inset 0 1px 0 rgba(255, 255, 255, 0.5)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.3s ease",
            "--dgcv-text-shadow": "0 1px 0 rgba(255, 255, 255, 0.5)",
        },
    ),
    "slate_and_copper": ThemeConfig(
        bg_primary="#2f3e46",
        bg_surface="#2f3e46",
        bg_alt="#354f52",
        bg_hover="#b87333",
        text_main="#cad2c5",
        text_heading="#cad2c5",
        text_hover="#2f3e46",
        border_main="#b87333",
        bg_action="#b87333",
        text_on_action="#ffffff",
        bg_action_hover="#cd7f32",
        bg_error="#e07a5f",
        text_on_error="#ffffff",
        bg_success="#81b29a",
        text_on_success="#2f3e46",
        border_alt="#52796f",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-table-shadow": "0 2px 8px rgba(184, 115, 51, 0.3)",
        },
    ),
    "sourdough": ThemeConfig(
        bg_primary="#f4e1d2",
        bg_surface="#8b5a2b",
        bg_alt="#f9e8dc",
        bg_hover="#d2a679",
        text_main="#5c4033",
        text_heading="#ffffff",
        text_hover="#5c4033",
        border_main="#d2a679",
        bg_action="#a0522d",
        text_on_action="#ffffff",
        bg_action_hover="#cd853f",
        bg_error="#8b0000",
        text_on_error="#ffffff",
        bg_success="#556b2f",
        text_on_success="#ffffff",
        border_alt="#deb887",
        font_family="Verdana, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-special-background": "linear-gradient(170deg, #f4e1d2 0%, #f9e8dc 60%, #f4e1d2 100%)",
            "--dgcv-special-text": "#5c4033",
            "--dgcv-table-shadow": "0 0 10px rgba(139, 90, 43, 0.2)",
            "--dgcv-text-shadow": "0 1px 0 rgba(255,255,255,0.3)",
        },
    ),
    "starry_night": ThemeConfig(
        bg_primary="#262a6e",
        bg_surface="#34407e",
        bg_alt="#303680",
        bg_hover="#f7d84b",
        text_main="#f4f2ff",
        text_heading="#f7d84b",
        text_hover="#1a1f50",
        text_alt="#cdd6f0",
        border_main="#f7d84b",
        border_alt="#3a4a88",
        bg_action="#1670cc",
        text_on_action="#ffffff",
        bg_action_hover="#2e8ae0",
        bg_error="#e23b3b",
        text_on_error="#ffffff",
        bg_success="#2bc070",
        text_on_success="#16204e",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-special-text": "#f7d84b",
            "--dgcv-special-background": "radial-gradient(1.5px 1.5px at 12% 20%, #f7d84b, transparent), radial-gradient(2px 2px at 28% 12%, #f7d84b, transparent), radial-gradient(1px 1px at 45% 28%, #fff, transparent), radial-gradient(2px 2px at 62% 16%, #f7d84b, transparent), radial-gradient(1.5px 1.5px at 78% 24%, #fff, transparent), radial-gradient(1px 1px at 88% 12%, #f7d84b, transparent), radial-gradient(2px 2px at 18% 46%, #f7d84b, transparent), radial-gradient(1px 1px at 38% 58%, #fff, transparent), radial-gradient(1.5px 1.5px at 55% 50%, #f7d84b, transparent), radial-gradient(1px 1px at 72% 62%, #fff, transparent), radial-gradient(2px 2px at 30% 78%, #f7d84b, transparent), radial-gradient(1.5px 1.5px at 50% 86%, #fff, transparent), radial-gradient(1px 1px at 84% 80%, #f7d84b, transparent), radial-gradient(circle at 82% 78%, rgba(247,216,75,0.55) 0%, rgba(247,216,75,0.12) 14%, transparent 32%), linear-gradient(to bottom, #12123a 0%, #1a1c4e 30%, #262a6e 55%, #2e4a96 100%)",
            "--dgcv-table-shadow": "0 0 20px rgba(247, 216, 75, 0.55), 0 0 60px rgba(46, 90, 168, 0.35), inset 0 1px 0 rgba(247, 216, 75, 0.18)",
            "--dgcv-text-shadow": "0 0 10px rgba(247, 216, 75, 0.5)",
            "--dgcv-hover-transform": "scale(1.001)",
        },
    ),
    "sunset_gradient": ThemeConfig(
        bg_primary="#1a0a0a",
        bg_surface="#2d1a0d",
        bg_alt="#221010",
        bg_hover="#ff9a3c",
        text_main="#fde8d8",
        text_heading="#fde8d8",
        text_hover="#1a0a0a",
        text_alt="#f5d0b8",
        border_main="#c45e2a",
        border_alt="#3a1a10",
        bg_action="#c2185b",
        text_on_action="#ffffff",
        bg_action_hover="#e91e63",
        bg_error="#b71c1c",
        text_on_error="#ffffff",
        bg_success="#00e676",
        text_on_success="#000000",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": "linear-gradient(160deg, #ff6b35 0%, #f7431c 20%, #c2185b 50%, #7b1fa2 80%, #4a148c 100%)",
            "--dgcv-table-shadow": "0 4px 24px rgba(194, 24, 91, 0.5), 0 0 60px rgba(255, 107, 53, 0.2)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.25s ease",
            "--dgcv-text-shadow": "0 1px 4px rgba(0,0,0,0.5)",
        },
    ),
    "teals": ThemeConfig(
        bg_primary="#EDF6F9",
        bg_surface="#005F73",
        bg_alt="#CAF0F8",
        bg_hover="#90E0EF",
        text_main="#023047",
        text_heading="#ffffff",
        text_hover="#023047",
        border_main="#00A8E8",
        bg_action="#0A9396",
        text_on_action="#ffffff",
        bg_action_hover="#94D2BD",
        bg_error="#E63946",
        text_on_error="#ffffff",
        bg_success="#52B788",
        text_on_success="#ffffff",
        border_alt="#ADE8F4",
        font_family="Helvetica, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "3px",
            "--dgcv-table-shadow": "0 0 10px rgba(0, 168, 232, 0.5)",
            "--dgcv-hover-transform": "scale(1.001)",
        },
    ),
    "turtle_shell": ThemeConfig(
        bg_primary="#f5f5f5",
        bg_surface="#556b2f",
        bg_alt="#e0e0e0",
        bg_hover="#6b8e23",
        text_main="#2f4f4f",
        text_heading="#ffffff",
        text_hover="#ffffff",
        border_main="#556b2f",
        bg_action="#8fbc8f",
        text_on_action="#000000",
        bg_action_hover="#9acd32",
        bg_error="#cd5c5c",
        text_on_error="#ffffff",
        bg_success="#2e8b57",
        text_on_success="#ffffff",
        border_alt="#d3d3d3",
        font_family="Tahoma, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-table-shadow": "0 0 8px rgba(85, 107, 47, 0.3)",
            "--dgcv-border-image": "linear-gradient(135deg, #556b2f, #9acd32, #556b2f) 1",
            "--dgcv-border-radius": "0",
        },
    ),
    "underwater": ThemeConfig(
        bg_primary="#87ceeb",
        bg_surface="#00ced1",
        bg_alt="#afeeee",
        bg_hover="#1e90ff",
        text_main="#006994",
        text_heading="#ffffff",
        text_hover="#ffffff",
        border_main="#4682b4",
        bg_action="#0000cd",
        text_on_action="#ffffff",
        bg_action_hover="#4169e1",
        bg_error="#cd5c5c",
        text_on_error="#ffffff",
        bg_success="#3cb371",
        text_on_success="#ffffff",
        border_alt="#add8e6",
        font_family="Trebuchet MS, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-special-background": "linear-gradient(170deg, #87ceeb 0%, #afeeee 50%, #87ceeb 100%)",
            "--dgcv-special-text": "#006994",
            "--dgcv-table-shadow": "0 4px 16px rgba(30, 144, 255, 0.5), 0 0 40px rgba(0, 206, 209, 0.2)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-text-shadow": "0 1px 2px rgba(0, 105, 148, 0.3)",
        },
    ),
    "van_gogh": ThemeConfig(
        bg_primary="#f7f2e7",
        bg_surface="#1c6ea4",
        bg_alt="#fff7d1",
        bg_hover="#ffd700",
        text_main="#1c6ea4",
        text_heading="#ffffff",
        text_hover="#1c6ea4",
        border_main="#1c6ea4",
        bg_action="#e6b800",
        text_on_action="#1c6ea4",
        bg_action_hover="#cca300",
        bg_error="#cc3300",
        text_on_error="#ffffff",
        bg_success="#339966",
        text_on_success="#ffffff",
        border_alt="#104e76",
        font_family="inherit",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-special-background": "linear-gradient(160deg, #f7f2e7 0%, #fff7d1 50%, #f7f2e7 100%)",
            "--dgcv-special-text": "#1c6ea4",
            "--dgcv-table-shadow": "0 4px 12px rgba(28, 110, 164, 0.25)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-text-shadow": "0 1px 0 rgba(255,255,255,0.5)",
        },
    ),
    "warm_orange_purple": ThemeConfig(
        bg_primary="#FFDAB9",
        bg_surface="#4B0082",
        bg_alt="#FFE4B5",
        bg_hover="#FF7F50",
        text_main="#4B0082",
        text_heading="#ffffff",
        text_hover="#4B0082",
        border_main="#800080",
        bg_action="#FF8C00",
        text_on_action="#ffffff",
        bg_action_hover="#FFA500",
        bg_error="#B22222",
        text_on_error="#ffffff",
        bg_success="#228B22",
        text_on_success="#ffffff",
        border_alt="#FFDEAD",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-hover-transition": "background-color 0.5s ease",
            "--dgcv-table-shadow": "0 2px 8px rgba(75, 0, 130, 0.2)",
            "--dgcv-hover-transform": "scale(1.001)",
        },
    ),
    "mercury": ThemeConfig(
        bg_primary="#0d0d0f",
        bg_surface="#1a1a1d",
        bg_alt="#131315",
        bg_hover="#d49a62",
        text_main="#e4e4e7",
        text_heading="#bcbcc2",
        text_hover="#0d0d0f",
        text_alt="#8e8e93",
        border_main="#5a5a60",
        border_alt="#26262a",
        bg_action="#54545a",
        text_on_action="#e4e4e7",
        bg_action_hover="#d49a62",
        bg_error="#b83a48",
        text_on_error="#e4e4e7",
        bg_success="#3a8a64",
        text_on_success="#e4e4e7",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-text": "#e4e4e7",
            "--dgcv-special-background": "linear-gradient(135deg, #0d0d0f 0%, #1e1e22 18%, #3a3a40 36%, #57575e 52%, #756f68 66%, #9a7c5c 78%, #e0e0e4 90%, #0d0d0f 100%)",
            "--dgcv-table-shadow": "0 0 20px rgba(90, 90, 96, 0.5), 0 0 60px rgba(212, 154, 98, 0.18), inset 0 1px 0 rgba(228, 228, 231, 0.12)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.2s ease",
            "--dgcv-text-shadow": "0 0 12px rgba(188, 188, 194, 0.55)",
            "--dgcv-border-image": "linear-gradient(135deg, #5a5a60, #8e8e93, #b89a72, #d49a62, #b89a72, #8e8e93, #5a5a60) 1",
            "--dgcv-border-radius": "0",
        },
    ),
    "venus": ThemeConfig(
        bg_primary="#130e06",
        bg_surface="#241a0c",
        bg_alt="#1a1209",
        bg_hover="#f4c032",
        text_main="#f8ecce",
        text_heading="#e8b84e",
        text_hover="#130e06",
        text_alt="#c0ad88",
        border_main="#b8862a",
        border_alt="#33240f",
        bg_action="#a87420",
        text_on_action="#f8ecce",
        bg_action_hover="#f4c032",
        bg_error="#c23028",
        text_on_error="#f8ecce",
        bg_success="#5a7a28",
        text_on_success="#f8ecce",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-text": "#f8ecce",
            "--dgcv-special-background": "linear-gradient(135deg, #130e06 0%, #2a1c08 16%, #4a3210 32%, #6e4c16 48%, #8e6620 64%, #c89630 78%, #f6e4b8 90%, #130e06 100%)",
            "--dgcv-table-shadow": "0 0 20px rgba(184, 134, 42, 0.5), 0 0 60px rgba(244, 192, 50, 0.22), inset 0 1px 0 rgba(248, 236, 206, 0.12)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.2s ease",
            "--dgcv-text-shadow": "0 0 12px rgba(232, 184, 78, 0.6)",
            "--dgcv-border-image": "linear-gradient(135deg, #b8862a, #e8b84e, #f4c032, #f8d870, #f4c032, #e8b84e, #b8862a) 1",
            "--dgcv-border-radius": "0",
        },
    ),
    "earth": ThemeConfig(
        bg_primary="#03101c",
        bg_surface="#082238",
        bg_alt="#051726",
        bg_hover="#2ea86e",
        text_main="#e4f2f8",
        text_heading="#5cb0e8",
        text_hover="#03101c",
        text_alt="#9ab0aa",
        border_main="#1c6aa0",
        border_alt="#0a2c44",
        bg_action="#15705a",
        text_on_action="#e4f2f8",
        bg_action_hover="#2ea86e",
        bg_error="#cc3646",
        text_on_error="#e4f2f8",
        bg_success="#1f8a4c",
        text_on_success="#e4f2f8",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-text": "#e4f2f8",
            "--dgcv-special-background": "linear-gradient(135deg, #03101c 0%, #06243f 18%, #0c4a78 36%, #137060 52%, #1f9460 66%, #2eb0a0 78%, #cdeef0 90%, #03101c 100%)",
            "--dgcv-table-shadow": "0 0 20px rgba(28, 106, 160, 0.45), 0 0 60px rgba(46, 168, 110, 0.22), inset 0 1px 0 rgba(228, 242, 248, 0.12)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.2s ease",
            "--dgcv-text-shadow": "0 0 12px rgba(92, 176, 232, 0.55)",
            "--dgcv-border-image": "linear-gradient(135deg, #1c6aa0, #1f9460, #2eb0a0, #4cc8b4, #2eb0a0, #1f9460, #1c6aa0) 1",
            "--dgcv-border-radius": "0",
        },
    ),
    "mars": ThemeConfig(
        bg_primary="#160605",
        bg_surface="#2a0e0b",
        bg_alt="#1d0908",
        bg_hover="#f0663c",
        text_main="#f6e3da",
        text_heading="#e89a82",
        text_hover="#160605",
        text_alt="#c2a098",
        border_main="#a8341c",
        border_alt="#42150f",
        bg_action="#a8341c",
        text_on_action="#f6e3da",
        bg_action_hover="#f0663c",
        bg_error="#d8202a",
        text_on_error="#f6e3da",
        bg_success="#2e6b42",
        text_on_success="#f6e3da",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-text": "#f6e3da",
            "--dgcv-special-background": "linear-gradient(135deg, #160605 0%, #330f0a 16%, #5e1a0f 32%, #8a2816 48%, #b0381c 64%, #d9542a 78%, #f3cdbe 90%, #160605 100%)",
            "--dgcv-table-shadow": "0 0 20px rgba(168, 52, 28, 0.5), 0 0 60px rgba(217, 84, 42, 0.22), inset 0 1px 0 rgba(246, 227, 218, 0.12)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.2s ease",
            "--dgcv-text-shadow": "0 0 12px rgba(232, 154, 130, 0.55)",
            "--dgcv-border-image": "linear-gradient(135deg, #a8341c, #cf5028, #e8825a, #f0a878, #e8825a, #cf5028, #a8341c) 1",
            "--dgcv-border-radius": "0",
        },
    ),
    "jupiter": ThemeConfig(
        bg_primary="#120b06",
        bg_surface="#221308",
        bg_alt="#180d07",
        bg_hover="#ec8a32",
        text_main="#f4e6d2",
        text_heading="#e0a464",
        text_hover="#120b06",
        text_alt="#bca890",
        border_main="#a85420",
        border_alt="#2e1a0e",
        bg_action="#9a4a1e",
        text_on_action="#f4e6d2",
        bg_action_hover="#ec8a32",
        bg_error="#c8302a",
        text_on_error="#f4e6d2",
        bg_success="#2e7044",
        text_on_success="#f4e6d2",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-text": "#f4e6d2",
            "--dgcv-special-background": "linear-gradient(135deg, #120b06 0%, #2e1810 16%, #5e3318 32%, #8a4a1e 48%, #b85e24 64%, #d8782e 78%, #f0d6ae 90%, #120b06 100%)",
            "--dgcv-table-shadow": "0 0 20px rgba(168, 84, 32, 0.5), 0 0 60px rgba(236, 138, 50, 0.22), inset 0 1px 0 rgba(244, 230, 210, 0.12)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.2s ease",
            "--dgcv-text-shadow": "0 0 12px rgba(224, 164, 100, 0.6)",
            "--dgcv-border-image": "linear-gradient(135deg, #a85420, #d8782e, #ec8a32, #f2b06a, #ec8a32, #d8782e, #a85420) 1",
            "--dgcv-border-radius": "0",
        },
    ),
    "saturn": ThemeConfig(
        bg_primary="#161009",
        bg_surface="#251a10",
        bg_alt="#1c130b",
        bg_hover="#f0c468",
        text_main="#fdf3da",
        text_heading="#e8b860",
        text_hover="#161009",
        text_alt="#c4cdd0",
        border_main="#cc9a38",
        border_alt="#352414",
        bg_action="#c87a2a",
        text_on_action="#fdf3da",
        bg_action_hover="#a85e20",
        bg_error="#bc2438",
        text_on_error="#fdf3da",
        bg_success="#2f8478",
        text_on_success="#fdf3da",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-text": "#e8b860",
            "--dgcv-special-background": "linear-gradient(135deg, #161009 0%, #4a3216 14%, #9a6a26 28%, #e8b860 42%, #fcd98a 56%, #e8e0d0 68%, #b8d0cc 80%, #6aa8a4 90%, #161009 100%)",
            "--dgcv-table-shadow": "0 0 20px rgba(204, 154, 56, 0.45), 0 0 60px rgba(106, 168, 164, 0.25), inset 0 1px 0 rgba(253, 243, 218, 0.15)",
            "--plaque-fill": "#251a10",
            "--plaque-border": "#6aa8a4",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.2s ease",
            "--dgcv-text-shadow": "0 0 12px rgba(232, 184, 96, 0.6)",
            "--dgcv-border-image": "linear-gradient(135deg, #cc9a38, #e8b860, #e8e0d0, #6aa8a4, #cc9a38) 1",
            "--dgcv-border-radius": "0",
        },
    ),
    "uranus": ThemeConfig(
        bg_primary="#04100f",
        bg_surface="#0a201f",
        bg_alt="#061817",
        bg_hover="#48c4be",
        text_main="#def4f2",
        text_heading="#6ec8c4",
        text_hover="#04100f",
        text_alt="#9ab8b6",
        border_main="#2a7a76",
        border_alt="#0e2a28",
        bg_action="#1f706c",
        text_on_action="#def4f2",
        bg_action_hover="#48c4be",
        bg_error="#bf3a52",
        text_on_error="#def4f2",
        bg_success="#268a64",
        text_on_success="#def4f2",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-text": "#def4f2",
            "--dgcv-special-background": "linear-gradient(135deg, #04100f 0%, #082423 18%, #0e4240 36%, #166260 52%, #1f8480 66%, #34a8a2 78%, #d4f2f0 90%, #04100f 100%)",
            "--dgcv-table-shadow": "0 0 20px rgba(42, 122, 118, 0.5), 0 0 60px rgba(72, 196, 190, 0.2), inset 0 1px 0 rgba(222, 244, 242, 0.12)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.2s ease",
            "--dgcv-text-shadow": "0 0 12px rgba(110, 200, 196, 0.6)",
            "--dgcv-border-image": "linear-gradient(135deg, #2a7a76, #34a8a2, #48c4be, #7adcd6, #48c4be, #34a8a2, #2a7a76) 1",
            "--dgcv-border-radius": "0",
        },
    ),
    "neptune": ThemeConfig(
        bg_primary="#04061c",
        bg_surface="#0a1038",
        bg_alt="#070a26",
        bg_hover="#3a64e8",
        text_main="#e2eafa",
        text_heading="#7d9cf0",
        text_hover="#04061c",
        text_alt="#94a0c4",
        border_main="#2840a8",
        border_alt="#0d1440",
        bg_action="#2840a8",
        text_on_action="#e2eafa",
        bg_action_hover="#3a64e8",
        bg_error="#c01e44",
        text_on_error="#e2eafa",
        bg_success="#1a7050",
        text_on_success="#e2eafa",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-text": "#e2eafa",
            "--dgcv-special-background": "linear-gradient(135deg, #04061c 0%, #0a1242 18%, #122363 36%, #1c378f 52%, #2848c0 66%, #3a5fd4 78%, #7d9ae8 84%, #cdd6f5 91%, #04061c 100%)",
            "--dgcv-table-shadow": "0 0 20px rgba(40, 64, 168, 0.55), 0 0 60px rgba(58, 100, 232, 0.25), inset 0 1px 0 rgba(226, 234, 250, 0.12)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.2s ease",
            "--dgcv-text-shadow": "0 0 12px rgba(125, 156, 240, 0.6)",
            "--dgcv-border-image": "linear-gradient(135deg, #2840a8, #3a5fd4, #5a86ec, #88aaf4, #5a86ec, #3a5fd4, #2840a8) 1",
            "--dgcv-border-radius": "0",
        },
    ),
    "pluto": ThemeConfig(
        bg_primary="#0b0a0f",
        bg_surface="#171420",
        bg_alt="#100e16",
        bg_hover="#bd9580",
        text_main="#eae6f0",
        text_heading="#b8a6c8",
        text_hover="#0b0a0f",
        text_alt="#9890a0",
        border_main="#5a5060",
        border_alt="#221e28",
        bg_action="#5a4a60",
        text_on_action="#eae6f0",
        bg_action_hover="#bd9580",
        bg_error="#a83048",
        text_on_error="#eae6f0",
        bg_success="#3a6e54",
        text_on_success="#eae6f0",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-text": "#eae6f0",
            "--dgcv-special-background": "linear-gradient(135deg, #0b0a0f 0%, #1a1620 18%, #2e2636 36%, #4a3c50 52%, #6e5868 66%, #a8847a 78%, #ece4f0 90%, #0b0a0f 100%)",
            "--dgcv-table-shadow": "0 0 20px rgba(90, 80, 96, 0.5), 0 0 60px rgba(189, 149, 128, 0.2), inset 0 1px 0 rgba(234, 230, 240, 0.12)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.2s ease",
            "--dgcv-text-shadow": "0 0 12px rgba(184, 166, 200, 0.55)",
            "--dgcv-border-image": "linear-gradient(135deg, #5a5060, #8a7494, #b8a6c8, #c4937a, #b8a6c8, #8a7494, #5a5060) 1",
            "--dgcv-border-radius": "0",
        },
    ),
    "quilt_bargello": ThemeConfig(
        bg_primary="#13182e",
        bg_surface="#1e2742",
        bg_alt="#171c34",
        bg_hover="#e07ab0",
        text_main="#eef2f8",
        text_heading="#54cfc4",
        text_hover="#13182e",
        text_alt="#aab8c8",
        border_main="#4a6ab0",
        border_alt="#1e2742",
        bg_action="#2e5aa8",
        text_on_action="#ffffff",
        bg_action_hover="#3e6ec0",
        bg_error="#c83a5a",
        text_on_error="#ffffff",
        bg_success="#2a9e7a",
        text_on_success="#08201a",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-text": "#eef2f8",
            "--dgcv-special-background": "repeating-linear-gradient(60deg, #155e5e 0 14px, #1e4a82 14px 28px, #36347e 28px 42px, #4e2e72 42px 56px, #6e2a64 56px 70px, #4e2e72 70px 84px, #36347e 84px 98px, #1e4a82 98px 112px)",
            "--plaque-fill": "#1e2742",
            "--plaque-border": "#54cfc4",
            "--dgcv-table-shadow": "0 0 20px rgba(110, 42, 100, 0.4), 0 0 60px rgba(46, 90, 168, 0.25), inset 0 1px 0 rgba(238, 242, 248, 0.12)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.2s ease",
        },
    ),
    "quilt_patch": ThemeConfig(
        bg_primary="#1e1a16",
        bg_surface="#2c241c",
        bg_alt="#241f19",
        bg_hover="#d8a84e",
        text_main="#f2e8d8",
        text_heading="#e0a83e",
        text_hover="#1e1a16",
        text_alt="#c8b89e",
        border_main="#8a6a3e",
        border_alt="#2c241c",
        bg_action="#7a2a2a",
        text_on_action="#f2e8d8",
        bg_action_hover="#9a3636",
        bg_error="#b83030",
        text_on_error="#f2e8d8",
        bg_success="#3a7a4a",
        text_on_success="#f2e8d8",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-text": "#f2e8d8",
            "--dgcv-special-background": "repeating-linear-gradient(45deg, rgba(255,255,255,0.05) 0 1px, transparent 1px 4px) 0 0 / 4px 4px, repeating-linear-gradient(-45deg, rgba(0,0,0,0.06) 0 1px, transparent 1px 5px) 0 0 / 5px 5px, repeating-linear-gradient(0deg, #8a7048 0 1px, transparent 1px 22px) 0 0 / 22px 22px, repeating-linear-gradient(90deg, #8a7048 0 1px, transparent 1px 22px) 0 0 / 22px 22px, conic-gradient(#7a2a2a 0 25%, #1f5a52 25% 50%, #243a6a 50% 75%, #6a4a1e 75% 100%) 0 0 / 22px 22px, #7a2a2a",
            "--plaque-fill": "#2c241c",
            "--plaque-border": "#e0a83e",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.2s ease",
        },
    ),
    "quilt_applique": ThemeConfig(
        bg_primary="#0F2420",
        bg_surface="#163530",
        bg_alt="#122A26",
        bg_hover="#F0A878",
        text_main="#F0ECE0",
        text_heading="#F0A878",
        text_hover="#0F2420",
        text_alt="#A8C0B8",
        border_main="#C87A5A",
        border_alt="#2A4C44",
        bg_action="#3E9B7A",
        text_on_action="#062018",
        bg_action_hover="#54B292",
        bg_error="#A8382C",
        text_on_error="#FBE6DE",
        bg_success="#2F6B4A",
        text_on_success="#E2F0E2",
        font_family="'Bookman Old Style', Bookman, 'Palatino Linotype', Palatino, Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "$randomized": (
                "wx=0:139 wy=0:139 ws=110:170 mx=18:82 my=20:80 ms=150:230 sx=10:88 sy=14:86 ss=130:200 sd=0:1 px"
                "=8:90 py=10:88 ps=64:104 pp=0:3 gx=0:159 gy=0:159 gs=150:200"
            ),
            "$assemble": "--dgcv-special-background",
            "$part_1": (
                "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140' view"
                "Box='0 0 140 140'%3E%3Cfilter id='wa' x='0' y='0' width='100%' height='100%' color-interpolation"
                "-filters='sRGB'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.06 0.85' numOctaves='2' s"
                "eed='7' stitchTiles='stitch' result='n'/%3E%3CfeColorMatrix in='n' type='matrix' values='0 0 0 0"
                " 0.93 0 0 0 0 0.89 0 0 0 0 0.79 0.30 0 0 0 -0.15'/%3E%3C/filter%3E%3Cfilter id='wb' x='0' y='0' "
                "width='100%' height='100%' color-interpolation-filters='sRGB'%3E%3CfeTurbulence type='fractalNoi"
                "se' baseFrequency='0.85 0.06' numOctaves='2' seed='11' stitchTiles='stitch' result='n'/%3E%3CfeC"
                "olorMatrix in='n' type='matrix' values='0 0 0 0 0.05 0 0 0 0 0.11 0 0 0 0 0.09 0.28 0 0 0 -0.14'"
                "/%3E%3C/filter%3E%3Crect width='140' height='140' filter='url%28%23wa%29'/%3E%3Crect width='140'"
                " height='140' filter='url%28%23wb%29'/%3E%3C/svg%3E\") ${wx}px ${wy}px / ${ws}px ${ws}px repeat,"
                " url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='-60 -60 120 120'%3E"
                "%3Cdefs%3E%3ClinearGradient id='pf' gradientUnits='userSpaceOnUse' x1='0' y1='-47' x2='0' y2='-1"
                "0'%3E%3Cstop offset='0' stop-color='%23F09274'/%3E%3Cstop offset='1' stop-color='%23DC6A50'/%3E%"
                "3C/linearGradient%3E%3ClinearGradient id='uf' gradientUnits='userSpaceOnUse' x1='0' y1='-47' x2="
                "'0' y2='-10'%3E%3Cstop offset='0' stop-color='%235A2844'/%3E%3Cstop offset='1' stop-color='%233A"
                "162C'/%3E%3C/linearGradient%3E%3ClinearGradient id='cf' gradientUnits='userSpaceOnUse' x1='0' y1"
                "='-14' x2='0' y2='14'%3E%3Cstop offset='0' stop-color='%23EFC46C'/%3E%3Cstop offset='1' stop-col"
                "or='%23C89532'/%3E%3C/linearGradient%3E%3C/defs%3E%3Ccircle cx='0' cy='0' r='56' fill='none' str"
                "oke='%23F2E6CE' stroke-width='0.9' stroke-dasharray='3 3.6' opacity='0.20'/%3E%3Ccircle cx='0' c"
                "y='0' r='51' fill='none' stroke='%23F2E6CE' stroke-width='0.9' stroke-dasharray='3 3.6' opacity="
                "'0.26'/%3E%3Cpath d='M 0,-10 C 9,-20 12,-32 0,-47 C -12,-32 -9,-20 0,-10 Z' transform='rotate%28"
                "22.5%29 scale%280.78%29' fill='url%28%23uf%29' stroke='%23F2E6CE' stroke-width='1.7' stroke-dash"
                "array='3.4 3' stroke-opacity='0.55' stroke-linejoin='round'/%3E%3Cpath d='M 0,-10 C 9,-20 12,-32"
                " 0,-47 C -12,-32 -9,-20 0,-10 Z' transform='rotate%2867.5%29 scale%280.78%29' fill='url%28%23uf%"
                "29' stroke='%23F2E6CE' stroke-width='1.7' stroke-dasharray='3.4 3' stroke-opacity='0.55' stroke-"
                "linejoin='round'/%3E%3Cpath d='M 0,-10 C 9,-20 12,-32 0,-47 C -12,-32 -9,-20 0,-10 Z' transform="
                "'rotate%28112.5%29 scale%280.78%29' fill='url%28%23uf%29' stroke='%23F2E6CE' stroke-width='1.7' "
                "stroke-dasharray='3.4 3' stroke-opacity='0.55' stroke-linejoin='round'/%3E%3Cpath d='M 0,-10 C 9"
                ",-20 12,-32 0,-47 C -12,-32 -9,-20 0,-10 Z' transform='rotate%28157.5%29 scale%280.78%29' fill='"
                "url%28%23uf%29' stroke='%23F2E6CE' stroke-width='1.7' stroke-dasharray='3.4 3' stroke-opacity='0"
                ".55' stroke-linejoin='round'/%3E%3Cpath d='M 0,-10 C 9,-20 12,-32 0,-47 C -12,-32 -9,-20 0,-10 Z"
                "' transform='rotate%28202.5%29 scale%280.78%29' fill='url%28%23uf%29' stroke='%23F2E6CE' stroke-"
                "width='1.7' stroke-dasharray='3.4 3' stroke-opacity='0.55' stroke-linejoin='round'/%3E%3Cpath d="
                "'M 0,-10 C 9,-20 12,-32 0,-47 C -12,-32 -9,-20 0,-10 Z' transform='rotate%28247.5%29 scale%280.7"
                "8%29' fill='url%28%23uf%29' stroke='%23F2E6CE' stroke-width='1.7' stroke-dasharray='3.4 3' strok"
                "e-opacity='0.55' stroke-linejoin='round'/%3E%3Cpath d='M 0,-10 C 9,-20 12,-32 0,-47 C -12,-32 -9"
                ",-20 0,-10 Z' transform='rotate%28292.5%29 scale%280.78%29' fill='url%28%23uf%29' stroke='%23F2E"
                "6CE' stroke-width='1.7' stroke-dasharray='3.4 3' stroke-opacity='0.55' stroke-linejoin='round'/%"
                "3E%3Cpath d='M 0,-10 C 9,-20 12,-32 0,-47 C -12,-32 -9,-20 0,-10 Z' transform='rotate%28337.5%29"
                " scale%280.78%29' fill='url%28%23uf%29' stroke='%23F2E6CE' stroke-width='1.7' stroke-dasharray='"
                "3.4 3' stroke-opacity='0.55' stroke-linejoin='round'/%3E%3Cpath d='M 0,-10 C 9,-20 12,-32 0,-47 "
                "C -12,-32 -9,-20 0,-10 Z' transform='rotate%280%29' fill='url%28%23pf%29' stroke='%23F2E6CE' str"
                "oke-width='1.5' stroke-dasharray='3.2 2.8' stroke-opacity='0.72' stroke-linejoin='round'/%3E%3Cp"
                "ath d='M 0,-10 C 9,-20 12,-32 0,-47 C -12,-32 -9,-20 0,-10 Z' transform='rotate%2845%29' fill='u"
                "rl%28%23pf%29' stroke='%23F2E6CE' stroke-width='1.5' stroke-dasharray='3.2 2.8' stroke-opacity='"
                "0.72' stroke-linejoin='round'/%3E%3Cpath d='M 0,-10 C 9,-20 12,-32 0,-47 C -12,-32 -9,-20 0,-10 "
                "Z' transform='rotate%2890%29' fill='url%28%23pf%29' stroke='%23F2E6CE' stroke-width='1.5' stroke"
                "-dasharray='3.2 2.8' stroke-opacity='0.72' stroke-linejoin='round'/%3E%3Cpath d='M 0,-10 C 9,-20"
                " 12,-32 0,-47 C -12,-32 -9,-20 0,-10 Z' transform='rotate%28135%29' fill='url%28%23pf%29' stroke"
                "='%23F2E6CE' stroke-width='1.5' stroke-dasharray='3.2 2.8' stroke-opacity='0.72' stroke-linejoin"
                "='round'/%3E%3Cpath d='M 0,-10 C 9,-20 12,-32 0,-47 C -12,-32 -9,-20 0,-10 Z' transform='rotate%"
                "28180%29' fill='url%28%23pf%29' stroke='%23F2E6CE' stroke-width='1.5' stroke-dasharray='3.2 2.8'"
                " stroke-opacity='0.72' stroke-linejoin='round'/%3E%3Cpath d='M 0,-10 C 9,-20 12,-32 0,-47 C -12,"
                "-32 -9,-20 0,-10 Z' transform='rotate%28225%29' fill='url%28%23pf%29' stroke='%23F2E6CE' stroke-"
                "width='1.5' stroke-dasharray='3.2 2.8' stroke-opacity='0.72' stroke-linejoin='round'/%3E%3Cpath "
                "d='M 0,-10 C 9,-20 12,-32 0,-47 C -12,-32 -9,-20 0,-10 Z' transform='rotate%28270%29' fill='url%"
                "28%23pf%29' stroke='%23F2E6CE' stroke-width='1.5' stroke-dasharray='3.2 2.8' stroke-opacity='0.7"
                "2' stroke-linejoin='round'/%3E%3Cpath d='M 0,-10 C 9,-20 12,-32 0,-47 C -12,-32 -9,-20 0,-10 Z' "
                "transform='rotate%28315%29' fill='url%28%23pf%29' stroke='%23F2E6CE' stroke-width='1.5' stroke-d"
                "asharray='3.2 2.8' stroke-opacity='0.72' stroke-linejoin='round'/%3E%3Ccircle cx='0' cy='0' r='1"
                "3.5' fill='url%28%23cf%29' stroke='%23F2E6CE' stroke-width='1.6' stroke-dasharray='3.2 2.8' stro"
                "ke-opacity='0.75'/%3E%3Ccircle cx='0' cy='0' r='5.5' fill='%233A162C' stroke='%23F2E6CE' stroke-"
                "width='1.1' stroke-dasharray='2.4 2.2' stroke-opacity='0.6'/%3E%3C/svg%3E\") ${mx}% ${my}% / min"
                "(${ms}px, 46%) auto no-repeat"
            ),
            "$part_2@detail": (
                "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='-60 -40 120 80'%3E%3"
                "Cdefs%3E%3ClinearGradient id='lf' gradientUnits='userSpaceOnUse' x1='0' y1='-11' x2='0' y2='7'%3"
                "E%3Cstop offset='0' stop-color='%237FC79A'/%3E%3Cstop offset='1' stop-color='%2357A374'/%3E%3C/l"
                "inearGradient%3E%3ClinearGradient id='bd' gradientUnits='userSpaceOnUse' x1='0' y1='-26' x2='0' "
                "y2='0'%3E%3Cstop offset='0' stop-color='%23EFC46C'/%3E%3Cstop offset='1' stop-color='%23DC6A50'/"
                "%3E%3C/linearGradient%3E%3C/defs%3E%3Cg%3E%3Cpath d='M -52,26 C -32,20 -14,6 0,-12 C 11,-25 24,-"
                "28 38,-27' fill='none' stroke='%234E9268' stroke-width='5' stroke-linecap='round'/%3E%3Cpath d='"
                "M -52,26 C -32,20 -14,6 0,-12 C 11,-25 24,-28 38,-27' fill='none' stroke='%23F2E6CE' stroke-widt"
                "h='1' stroke-dasharray='2.6 3.2' opacity='0.45'/%3E%3Cpath d='M 0,0 C 8,-9 22,-11 31,-3 C 22,7 8"
                ",7 0,0 Z' transform='translate%28-38 22%29 rotate%2838%29 scale%280.85%29' fill='url%28%23lf%29'"
                " stroke='%23F2E6CE' stroke-width='1.4' stroke-dasharray='2.8 2.6' stroke-opacity='0.6' stroke-li"
                "nejoin='round'/%3E%3Cpath d='M 0,0 C 8,-9 22,-11 31,-3 C 22,7 8,7 0,0 Z' transform='translate%28"
                "-24 15%29 rotate%28215%29 scale%280.75%29' fill='url%28%23lf%29' stroke='%23F2E6CE' stroke-width"
                "='1.4' stroke-dasharray='2.8 2.6' stroke-opacity='0.6' stroke-linejoin='round'/%3E%3Cpath d='M 0"
                ",0 C 8,-9 22,-11 31,-3 C 22,7 8,7 0,0 Z' transform='translate%28-8 0%29 rotate%2852%29 scale%280"
                ".9%29' fill='url%28%23lf%29' stroke='%23F2E6CE' stroke-width='1.4' stroke-dasharray='2.8 2.6' st"
                "roke-opacity='0.6' stroke-linejoin='round'/%3E%3Cpath d='M 0,0 C 8,-9 22,-11 31,-3 C 22,7 8,7 0,"
                "0 Z' transform='translate%286 -16%29 rotate%28228%29 scale%280.8%29' fill='url%28%23lf%29' strok"
                "e='%23F2E6CE' stroke-width='1.4' stroke-dasharray='2.8 2.6' stroke-opacity='0.6' stroke-linejoin"
                "='round'/%3E%3Cpath d='M 0,0 C 8,-9 22,-11 31,-3 C 22,7 8,7 0,0 Z' transform='translate%2822 -25"
                "%29 rotate%2828%29 scale%280.68%29' fill='url%28%23lf%29' stroke='%23F2E6CE' stroke-width='1.4' "
                "stroke-dasharray='2.8 2.6' stroke-opacity='0.6' stroke-linejoin='round'/%3E%3Cpath d='M 0,-26 C "
                "11,-14 11,-4 0,0 C -11,-4 -11,-14 0,-26 Z' transform='translate%2838 -27%29 rotate%2872%29 scale"
                "%280.72%29' fill='url%28%23bd%29' stroke='%23F2E6CE' stroke-width='1.4' stroke-dasharray='2.8 2."
                "6' stroke-opacity='0.65' stroke-linejoin='round'/%3E%3Cpath d='M -54,33 C -34,27 -16,13 -2,-5 C "
                "9,-18 24,-21 40,-20' fill='none' stroke='%23F2E6CE' stroke-width='0.9' stroke-dasharray='3 3.6' "
                "opacity='0.22'/%3E%3C/g%3E%3C/svg%3E\") calc(${sx}% + ${sd} * 9999px) ${sy}% / min(${ss}px, 42%)"
                " auto no-repeat, url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='-60"
                " -40 120 80'%3E%3Cdefs%3E%3ClinearGradient id='lf' gradientUnits='userSpaceOnUse' x1='0' y1='-11"
                "' x2='0' y2='7'%3E%3Cstop offset='0' stop-color='%237FC79A'/%3E%3Cstop offset='1' stop-color='%2"
                "357A374'/%3E%3C/linearGradient%3E%3ClinearGradient id='bd' gradientUnits='userSpaceOnUse' x1='0'"
                " y1='-26' x2='0' y2='0'%3E%3Cstop offset='0' stop-color='%23EFC46C'/%3E%3Cstop offset='1' stop-c"
                "olor='%23DC6A50'/%3E%3C/linearGradient%3E%3C/defs%3E%3Cg transform='scale%28-1 1%29'%3E%3Cpath d"
                "='M -52,26 C -32,20 -14,6 0,-12 C 11,-25 24,-28 38,-27' fill='none' stroke='%234E9268' stroke-wi"
                "dth='5' stroke-linecap='round'/%3E%3Cpath d='M -52,26 C -32,20 -14,6 0,-12 C 11,-25 24,-28 38,-2"
                "7' fill='none' stroke='%23F2E6CE' stroke-width='1' stroke-dasharray='2.6 3.2' opacity='0.45'/%3E"
                "%3Cpath d='M 0,0 C 8,-9 22,-11 31,-3 C 22,7 8,7 0,0 Z' transform='translate%28-38 22%29 rotate%2"
                "838%29 scale%280.85%29' fill='url%28%23lf%29' stroke='%23F2E6CE' stroke-width='1.4' stroke-dasha"
                "rray='2.8 2.6' stroke-opacity='0.6' stroke-linejoin='round'/%3E%3Cpath d='M 0,0 C 8,-9 22,-11 31"
                ",-3 C 22,7 8,7 0,0 Z' transform='translate%28-24 15%29 rotate%28215%29 scale%280.75%29' fill='ur"
                "l%28%23lf%29' stroke='%23F2E6CE' stroke-width='1.4' stroke-dasharray='2.8 2.6' stroke-opacity='0"
                ".6' stroke-linejoin='round'/%3E%3Cpath d='M 0,0 C 8,-9 22,-11 31,-3 C 22,7 8,7 0,0 Z' transform="
                "'translate%28-8 0%29 rotate%2852%29 scale%280.9%29' fill='url%28%23lf%29' stroke='%23F2E6CE' str"
                "oke-width='1.4' stroke-dasharray='2.8 2.6' stroke-opacity='0.6' stroke-linejoin='round'/%3E%3Cpa"
                "th d='M 0,0 C 8,-9 22,-11 31,-3 C 22,7 8,7 0,0 Z' transform='translate%286 -16%29 rotate%28228%2"
                "9 scale%280.8%29' fill='url%28%23lf%29' stroke='%23F2E6CE' stroke-width='1.4' stroke-dasharray='"
                "2.8 2.6' stroke-opacity='0.6' stroke-linejoin='round'/%3E%3Cpath d='M 0,0 C 8,-9 22,-11 31,-3 C "
                "22,7 8,7 0,0 Z' transform='translate%2822 -25%29 rotate%2828%29 scale%280.68%29' fill='url%28%23"
                "lf%29' stroke='%23F2E6CE' stroke-width='1.4' stroke-dasharray='2.8 2.6' stroke-opacity='0.6' str"
                "oke-linejoin='round'/%3E%3Cpath d='M 0,-26 C 11,-14 11,-4 0,0 C -11,-4 -11,-14 0,-26 Z' transfor"
                "m='translate%2838 -27%29 rotate%2872%29 scale%280.72%29' fill='url%28%23bd%29' stroke='%23F2E6CE"
                "' stroke-width='1.4' stroke-dasharray='2.8 2.6' stroke-opacity='0.65' stroke-linejoin='round'/%3"
                "E%3Cpath d='M -54,33 C -34,27 -16,13 -2,-5 C 9,-18 24,-21 40,-20' fill='none' stroke='%23F2E6CE'"
                " stroke-width='0.9' stroke-dasharray='3 3.6' opacity='0.22'/%3E%3C/g%3E%3C/svg%3E\") calc(${sx}%"
                ' + (1 - ${sd}) * 9999px) ${sy}% / min(${ss}px, 42%) auto no-repeat, url("data:image/svg+xml,%3C'
                "svg xmlns='http://www.w3.org/2000/svg' viewBox='-40 -40 80 80'%3E%3Cdefs%3E%3ClinearGradient id="
                "'pk' gradientUnits='userSpaceOnUse' x1='0' y1='-26' x2='0' y2='0'%3E%3Cstop offset='0' stop-colo"
                "r='%23EFC46C'/%3E%3Cstop offset='1' stop-color='%23C89532'/%3E%3C/linearGradient%3E%3C/defs%3E%3"
                "Ccircle cx='0' cy='0' r='34' fill='none' stroke='%23F2E6CE' stroke-width='0.9' stroke-dasharray="
                "'3 3.6' opacity='0.22'/%3E%3Cpath d='M 0,-26 C 11,-14 11,-4 0,0 C -11,-4 -11,-14 0,-26 Z' transf"
                "orm='rotate%280%29' fill='url%28%23pk%29' stroke='%23F2E6CE' stroke-width='1.5' stroke-dasharray"
                "='3 2.6' stroke-opacity='0.68' stroke-linejoin='round'/%3E%3Cpath d='M 0,-26 C 11,-14 11,-4 0,0 "
                "C -11,-4 -11,-14 0,-26 Z' transform='rotate%2890%29' fill='url%28%23pk%29' stroke='%23F2E6CE' st"
                "roke-width='1.5' stroke-dasharray='3 2.6' stroke-opacity='0.68' stroke-linejoin='round'/%3E%3Cpa"
                "th d='M 0,-26 C 11,-14 11,-4 0,0 C -11,-4 -11,-14 0,-26 Z' transform='rotate%28180%29' fill='url"
                "%28%23pk%29' stroke='%23F2E6CE' stroke-width='1.5' stroke-dasharray='3 2.6' stroke-opacity='0.68"
                "' stroke-linejoin='round'/%3E%3Cpath d='M 0,-26 C 11,-14 11,-4 0,0 C -11,-4 -11,-14 0,-26 Z' tra"
                "nsform='rotate%28270%29' fill='url%28%23pk%29' stroke='%23F2E6CE' stroke-width='1.5' stroke-dash"
                "array='3 2.6' stroke-opacity='0.68' stroke-linejoin='round'/%3E%3Ccircle cx='0' cy='0' r='5' fil"
                "l='%235A2844' stroke='%23F2E6CE' stroke-width='1.1' stroke-dasharray='2.4 2.2' stroke-opacity='0"
                ".6'/%3E%3C/svg%3E\") calc(${px}% + max(0, ${pp} - 2) * 9999px) ${py}% / min(${ps}px, 22%) auto n"
                "o-repeat"
            ),
            "$part_3": (
                "radial-gradient(ellipse 122% 98% at 50% 38%, rgba(255, 225, 180, 0.07) 0%, rgba(255, 225, 180, 0"
                '.02) 34%, rgba(6, 18, 15, 0.12) 66%, rgba(6, 18, 15, 0.42) 100%) center / cover no-repeat, url("'
                "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160' viewBox='0"
                " 0 160 160'%3E%3Cdefs%3E%3ClinearGradient id='sg' gradientUnits='userSpaceOnUse' x1='0' y1='0' x"
                "2='0' y2='160'%3E%3Cstop offset='0' stop-color='%230D211D'/%3E%3Cstop offset='1' stop-color='%23"
                "0A1B18'/%3E%3C/linearGradient%3E%3C/defs%3E%3Crect width='160' height='160' fill='url%28%23sg%29"
                "'/%3E%3Crect x='12' y='12' width='68' height='68' fill='%2314322C'/%3E%3Crect x='17' y='17' widt"
                "h='58' height='58' fill='none' stroke='%23F2E6CE' stroke-width='0.9' stroke-dasharray='3.4 3.6' "
                "opacity='0.16'/%3E%3Cpath d='M 12,12 L 80,80 M 80,12 L 12,80' fill='none' stroke='%23F2E6CE' str"
                "oke-width='0.7' stroke-dasharray='3 4' opacity='0.09'/%3E%3Crect x='92' y='12' width='68' height"
                "='68' fill='%23123029'/%3E%3Crect x='97' y='17' width='58' height='58' fill='none' stroke='%23F2"
                "E6CE' stroke-width='0.9' stroke-dasharray='3.4 3.6' opacity='0.16'/%3E%3Cpath d='M 92,12 L 160,8"
                "0 M 160,12 L 92,80' fill='none' stroke='%23F2E6CE' stroke-width='0.7' stroke-dasharray='3 4' opa"
                "city='0.09'/%3E%3Crect x='12' y='92' width='68' height='68' fill='%2316352E'/%3E%3Crect x='17' y"
                "='97' width='58' height='58' fill='none' stroke='%23F2E6CE' stroke-width='0.9' stroke-dasharray="
                "'3.4 3.6' opacity='0.16'/%3E%3Cpath d='M 12,92 L 80,160 M 80,92 L 12,160' fill='none' stroke='%2"
                "3F2E6CE' stroke-width='0.7' stroke-dasharray='3 4' opacity='0.09'/%3E%3Crect x='92' y='92' width"
                "='68' height='68' fill='%23112823'/%3E%3Crect x='97' y='97' width='58' height='58' fill='none' s"
                "troke='%23F2E6CE' stroke-width='0.9' stroke-dasharray='3.4 3.6' opacity='0.16'/%3E%3Cpath d='M 9"
                "2,92 L 160,160 M 160,92 L 92,160' fill='none' stroke='%23F2E6CE' stroke-width='0.7' stroke-dasha"
                "rray='3 4' opacity='0.09'/%3E%3Cpath d='M 6,-4 L 6,164' fill='none' stroke='%23F2E6CE' stroke-wi"
                "dth='1' stroke-dasharray='4.5 4' opacity='0.22'/%3E%3Cpath d='M -4,6 L 164,6' fill='none' stroke"
                "='%23F2E6CE' stroke-width='1' stroke-dasharray='4.5 4' opacity='0.22'/%3E%3Cpath d='M 86,-4 L 86"
                ",164' fill='none' stroke='%23F2E6CE' stroke-width='1' stroke-dasharray='4.5 4' opacity='0.22'/%3"
                "E%3Cpath d='M -4,86 L 164,86' fill='none' stroke='%23F2E6CE' stroke-width='1' stroke-dasharray='"
                "4.5 4' opacity='0.22'/%3E%3C/svg%3E\") ${gx}px ${gy}px / ${gs}px ${gs}px repeat, linear-gradient"
                "(to bottom, #17352E 0%, #143029 55%, #0D211D 100%) center / cover no-repeat, #14302A"
            ),
            "--dgcv-special-text": "#F4E9D2",
            "--plaque-fill": "#122E28",
            "--plaque-border": "#F0A878",
            "--dgcv-table-shadow": "0 2px 12px rgba(4, 14, 12, 0.55)",
            "--dgcv-text-shadow": "none",
            "--dgcv-hover-transform": "translateY(-1px)",
            "--dgcv-hover-transition": "background-color 160ms ease, color 160ms ease, transform 160ms ease",
        },
    ),
    "quilt_denim": ThemeConfig(
        bg_primary="#1c3552",
        bg_surface="#27486e",
        bg_alt="#203c5e",
        bg_hover="#e8b860",
        text_main="#eaf0f6",
        text_heading="#e8b860",
        text_hover="#1c3552",
        text_alt="#aac0d4",
        border_main="#cf9a3e",
        border_alt="#27486e",
        bg_action="#2a5a8e",
        text_on_action="#ffffff",
        bg_action_hover="#3a6ea8",
        bg_error="#c0392b",
        text_on_error="#ffffff",
        bg_success="#3a8a5a",
        text_on_success="#ffffff",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-text": "#e8b860",
            "--dgcv-special-background": "repeating-linear-gradient(45deg, rgba(255,255,255,0.05) 0 1px, transparent 1px 4px) 0 0 / 4px 4px, repeating-linear-gradient(-45deg, rgba(0,0,0,0.06) 0 1px, transparent 1px 5px) 0 0 / 5px 5px, repeating-linear-gradient(0deg, #4e6e98 0 1px, transparent 1px 18px) 0 0 / 36px 36px, repeating-linear-gradient(90deg, #4e6e98 0 1px, transparent 1px 18px) 0 0 / 36px 36px, conic-gradient(from 0deg, #20405f 0 25%, #2a4e78 25% 50%, #20405f 50% 75%, #2a4e78 75% 100%) 0 0 / 36px 36px, #22425f",
            "--plaque-fill": "#27486e",
            "--plaque-border": "#e8b860",
            "--dgcv-table-shadow": "0 0 20px rgba(207, 154, 62, 0.3), 0 0 60px rgba(42, 90, 142, 0.3), inset 0 1px 0 rgba(234, 240, 246, 0.12)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.2s ease",
        },
    ),
    "quilt_kagome_light": ThemeConfig(
        bg_primary="#f8f3ec",
        bg_surface="#ece8f2",
        bg_alt="#f0ece4",
        bg_hover="#f4b89a",
        text_main="#3a3142",
        text_heading="#6a4a8a",
        text_hover="#3a3142",
        text_alt="#6a6274",
        border_main="#c8bcd4",
        border_alt="#ddd4e2",
        bg_action="#7a5aa0",
        text_on_action="#ffffff",
        bg_action_hover="#8c6ab2",
        bg_error="#c84a5a",
        text_on_error="#ffffff",
        bg_success="#2a8055",
        text_on_success="#ffffff",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-text": "#3a2e44",
            "--dgcv-special-background": "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16.17' height='28.00' viewBox='0 0 16.17 28.00'%3E%3Cdefs%3E%3CclipPath id='a'%3E%3Cpolygon points='40.00,-32.50 -40.00,-32.50 -40.00,-37.50 40.00,-37.50'/%3E%3Cpolygon points='40.00,-18.50 -40.00,-18.50 -40.00,-23.50 40.00,-23.50'/%3E%3Cpolygon points='40.00,-4.50 -40.00,-4.50 -40.00,-9.50 40.00,-9.50'/%3E%3Cpolygon points='40.00,9.50 -40.00,9.50 -40.00,4.50 40.00,4.50'/%3E%3Cpolygon points='40.00,23.50 -40.00,23.50 -40.00,18.50 40.00,18.50'/%3E%3Cpolygon points='40.00,37.50 -40.00,37.50 -40.00,32.50 40.00,32.50'/%3E%3Cpolygon points='40.00,51.50 -40.00,51.50 -40.00,46.50 40.00,46.50'/%3E%3Cpolygon points='40.00,65.50 -40.00,65.50 -40.00,60.50 40.00,60.50'/%3E%3C/clipPath%3E%3C/defs%3E%3Crect width='16.17' height='28.00' fill='%23f6f1ea'/%3E%3Cg%3E%3Cpolygon points='-20.27,-57.89 -60.27,11.39 -64.60,8.89 -24.60,-60.39' fill='%23b0bce8'/%3E%3Cpolygon points='-8.15,-50.89 -48.15,18.39 -52.48,15.89 -12.48,-53.39' fill='%23b0bce8'/%3E%3Cpolygon points='3.98,-43.89 -36.02,25.39 -40.35,22.89 -0.35,-46.39' fill='%23b0bce8'/%3E%3Cpolygon points='16.10,-36.89 -23.90,32.39 -28.23,29.89 11.77,-39.39' fill='%23b0bce8'/%3E%3Cpolygon points='28.23,-29.89 -11.77,39.39 -16.10,36.89 23.90,-32.39' fill='%23b0bce8'/%3E%3Cpolygon points='40.35,-22.89 0.35,46.39 -3.98,43.89 36.02,-25.39' fill='%23b0bce8'/%3E%3Cpolygon points='52.48,-15.89 12.48,53.39 8.15,50.89 48.15,-18.39' fill='%23b0bce8'/%3E%3Cpolygon points='64.60,-8.89 24.60,60.39 20.27,57.89 60.27,-11.39' fill='%23b0bce8'/%3E%3Cpolygon points='76.72,-1.89 36.72,67.39 32.39,64.89 72.39,-4.39' fill='%23b0bce8'/%3E%3Cpolygon points='88.85,5.11 48.85,74.39 44.52,71.89 84.52,2.61' fill='%23b0bce8'/%3E%3C/g%3E%3Cg%3E%3Cpolygon points='-72.39,-4.39 -32.39,64.89 -36.72,67.39 -76.72,-1.89' fill='%23a8d8b8'/%3E%3Cpolygon points='-60.27,-11.39 -20.27,57.89 -24.60,60.39 -64.60,-8.89' fill='%23a8d8b8'/%3E%3Cpolygon points='-48.15,-18.39 -8.15,50.89 -12.48,53.39 -52.48,-15.89' fill='%23a8d8b8'/%3E%3Cpolygon points='-36.02,-25.39 3.98,43.89 -0.35,46.39 -40.35,-22.89' fill='%23a8d8b8'/%3E%3Cpolygon points='-23.90,-32.39 16.10,36.89 11.77,39.39 -28.23,-29.89' fill='%23a8d8b8'/%3E%3Cpolygon points='-11.77,-39.39 28.23,29.89 23.90,32.39 -16.10,-36.89' fill='%23a8d8b8'/%3E%3Cpolygon points='0.35,-46.39 40.35,22.89 36.02,25.39 -3.98,-43.89' fill='%23a8d8b8'/%3E%3Cpolygon points='12.48,-53.39 52.48,15.89 48.15,18.39 8.15,-50.89' fill='%23a8d8b8'/%3E%3Cpolygon points='24.60,-60.39 64.60,8.89 60.27,11.39 20.27,-57.89' fill='%23a8d8b8'/%3E%3Cpolygon points='36.72,-67.39 76.72,1.89 72.39,4.39 32.39,-64.89' fill='%23a8d8b8'/%3E%3C/g%3E%3Cg%3E%3Cpolygon points='40.00,-32.50 -40.00,-32.50 -40.00,-37.50 40.00,-37.50' fill='%23f4b89a'/%3E%3Cpolygon points='40.00,-18.50 -40.00,-18.50 -40.00,-23.50 40.00,-23.50' fill='%23f4b89a'/%3E%3Cpolygon points='40.00,-4.50 -40.00,-4.50 -40.00,-9.50 40.00,-9.50' fill='%23f4b89a'/%3E%3Cpolygon points='40.00,9.50 -40.00,9.50 -40.00,4.50 40.00,4.50' fill='%23f4b89a'/%3E%3Cpolygon points='40.00,23.50 -40.00,23.50 -40.00,18.50 40.00,18.50' fill='%23f4b89a'/%3E%3Cpolygon points='40.00,37.50 -40.00,37.50 -40.00,32.50 40.00,32.50' fill='%23f4b89a'/%3E%3Cpolygon points='40.00,51.50 -40.00,51.50 -40.00,46.50 40.00,46.50' fill='%23f4b89a'/%3E%3Cpolygon points='40.00,65.50 -40.00,65.50 -40.00,60.50 40.00,60.50' fill='%23f4b89a'/%3E%3C/g%3E%3Cg clip-path='url(%23a)'%3E%3Cpolygon points='-20.27,-57.89 -60.27,11.39 -64.60,8.89 -24.60,-60.39' fill='%23b0bce8'/%3E%3Cpolygon points='-8.15,-50.89 -48.15,18.39 -52.48,15.89 -12.48,-53.39' fill='%23b0bce8'/%3E%3Cpolygon points='3.98,-43.89 -36.02,25.39 -40.35,22.89 -0.35,-46.39' fill='%23b0bce8'/%3E%3Cpolygon points='16.10,-36.89 -23.90,32.39 -28.23,29.89 11.77,-39.39' fill='%23b0bce8'/%3E%3Cpolygon points='28.23,-29.89 -11.77,39.39 -16.10,36.89 23.90,-32.39' fill='%23b0bce8'/%3E%3Cpolygon points='40.35,-22.89 0.35,46.39 -3.98,43.89 36.02,-25.39' fill='%23b0bce8'/%3E%3Cpolygon points='52.48,-15.89 12.48,53.39 8.15,50.89 48.15,-18.39' fill='%23b0bce8'/%3E%3Cpolygon points='64.60,-8.89 24.60,60.39 20.27,57.89 60.27,-11.39' fill='%23b0bce8'/%3E%3Cpolygon points='76.72,-1.89 36.72,67.39 32.39,64.89 72.39,-4.39' fill='%23b0bce8'/%3E%3Cpolygon points='88.85,5.11 48.85,74.39 44.52,71.89 84.52,2.61' fill='%23b0bce8'/%3E%3C/g%3E%3C/svg%3E\") , #f6f1ea",
            "--plaque-fill": "#ece8f2",
            "--plaque-border": "#f4b89a",
            "--dgcv-table-shadow": "0 1px 4px rgba(106, 74, 138, 0.14), inset 0 1px 0 rgba(255, 255, 255, 0.5)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.3s ease",
        },
    ),
    "quilt_kagome_green": ThemeConfig(
        bg_primary="#f3f9f8",
        bg_surface="#e2efed",
        bg_alt="#ecf5f4",
        bg_hover="#c9e5e1",
        text_main="#143638",
        text_heading="#1f7a72",
        text_hover="#143638",
        text_alt="#46625f",
        border_main="#bcdbd6",
        border_alt="#d6eae6",
        bg_action="#236f69",
        text_on_action="#ffffff",
        bg_action_hover="#287d76",
        bg_error="#c8463f",
        text_on_error="#ffffff",
        bg_success="#237a4c",
        text_on_success="#ffffff",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-text": "#0f2d2f",
            "--dgcv-special-background": "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16.17' height='28.00' viewBox='0 0 16.17 28.00'%3E%3Cdefs%3E%3CclipPath id='a'%3E%3Cpolygon points='40.00,-32.50 -40.00,-32.50 -40.00,-37.50 40.00,-37.50'/%3E%3Cpolygon points='40.00,-18.50 -40.00,-18.50 -40.00,-23.50 40.00,-23.50'/%3E%3Cpolygon points='40.00,-4.50 -40.00,-4.50 -40.00,-9.50 40.00,-9.50'/%3E%3Cpolygon points='40.00,9.50 -40.00,9.50 -40.00,4.50 40.00,4.50'/%3E%3Cpolygon points='40.00,23.50 -40.00,23.50 -40.00,18.50 40.00,18.50'/%3E%3Cpolygon points='40.00,37.50 -40.00,37.50 -40.00,32.50 40.00,32.50'/%3E%3Cpolygon points='40.00,51.50 -40.00,51.50 -40.00,46.50 40.00,46.50'/%3E%3Cpolygon points='40.00,65.50 -40.00,65.50 -40.00,60.50 40.00,60.50'/%3E%3C/clipPath%3E%3C/defs%3E%3Crect width='16.17' height='28.00' fill='%23eef5f4'/%3E%3Cg%3E%3Cpolygon points='-20.27,-57.89 -60.27,11.39 -64.60,8.89 -24.60,-60.39' fill='%239aa6e4'/%3E%3Cpolygon points='-8.15,-50.89 -48.15,18.39 -52.48,15.89 -12.48,-53.39' fill='%239aa6e4'/%3E%3Cpolygon points='3.98,-43.89 -36.02,25.39 -40.35,22.89 -0.35,-46.39' fill='%239aa6e4'/%3E%3Cpolygon points='16.10,-36.89 -23.90,32.39 -28.23,29.89 11.77,-39.39' fill='%239aa6e4'/%3E%3Cpolygon points='28.23,-29.89 -11.77,39.39 -16.10,36.89 23.90,-32.39' fill='%239aa6e4'/%3E%3Cpolygon points='40.35,-22.89 0.35,46.39 -3.98,43.89 36.02,-25.39' fill='%239aa6e4'/%3E%3Cpolygon points='52.48,-15.89 12.48,53.39 8.15,50.89 48.15,-18.39' fill='%239aa6e4'/%3E%3Cpolygon points='64.60,-8.89 24.60,60.39 20.27,57.89 60.27,-11.39' fill='%239aa6e4'/%3E%3Cpolygon points='76.72,-1.89 36.72,67.39 32.39,64.89 72.39,-4.39' fill='%239aa6e4'/%3E%3Cpolygon points='88.85,5.11 48.85,74.39 44.52,71.89 84.52,2.61' fill='%239aa6e4'/%3E%3C/g%3E%3Cg%3E%3Cpolygon points='-72.39,-4.39 -32.39,64.89 -36.72,67.39 -76.72,-1.89' fill='%23f6977a'/%3E%3Cpolygon points='-60.27,-11.39 -20.27,57.89 -24.60,60.39 -64.60,-8.89' fill='%23f6977a'/%3E%3Cpolygon points='-48.15,-18.39 -8.15,50.89 -12.48,53.39 -52.48,-15.89' fill='%23f6977a'/%3E%3Cpolygon points='-36.02,-25.39 3.98,43.89 -0.35,46.39 -40.35,-22.89' fill='%23f6977a'/%3E%3Cpolygon points='-23.90,-32.39 16.10,36.89 11.77,39.39 -28.23,-29.89' fill='%23f6977a'/%3E%3Cpolygon points='-11.77,-39.39 28.23,29.89 23.90,32.39 -16.10,-36.89' fill='%23f6977a'/%3E%3Cpolygon points='0.35,-46.39 40.35,22.89 36.02,25.39 -3.98,-43.89' fill='%23f6977a'/%3E%3Cpolygon points='12.48,-53.39 52.48,15.89 48.15,18.39 8.15,-50.89' fill='%23f6977a'/%3E%3Cpolygon points='24.60,-60.39 64.60,8.89 60.27,11.39 20.27,-57.89' fill='%23f6977a'/%3E%3Cpolygon points='36.72,-67.39 76.72,1.89 72.39,4.39 32.39,-64.89' fill='%23f6977a'/%3E%3C/g%3E%3Cg%3E%3Cpolygon points='40.00,-32.50 -40.00,-32.50 -40.00,-37.50 40.00,-37.50' fill='%234fb1a8'/%3E%3Cpolygon points='40.00,-18.50 -40.00,-18.50 -40.00,-23.50 40.00,-23.50' fill='%234fb1a8'/%3E%3Cpolygon points='40.00,-4.50 -40.00,-4.50 -40.00,-9.50 40.00,-9.50' fill='%234fb1a8'/%3E%3Cpolygon points='40.00,9.50 -40.00,9.50 -40.00,4.50 40.00,4.50' fill='%234fb1a8'/%3E%3Cpolygon points='40.00,23.50 -40.00,23.50 -40.00,18.50 40.00,18.50' fill='%234fb1a8'/%3E%3Cpolygon points='40.00,37.50 -40.00,37.50 -40.00,32.50 40.00,32.50' fill='%234fb1a8'/%3E%3Cpolygon points='40.00,51.50 -40.00,51.50 -40.00,46.50 40.00,46.50' fill='%234fb1a8'/%3E%3Cpolygon points='40.00,65.50 -40.00,65.50 -40.00,60.50 40.00,60.50' fill='%234fb1a8'/%3E%3C/g%3E%3Cg clip-path='url(%23a)'%3E%3Cpolygon points='-20.27,-57.89 -60.27,11.39 -64.60,8.89 -24.60,-60.39' fill='%239aa6e4'/%3E%3Cpolygon points='-8.15,-50.89 -48.15,18.39 -52.48,15.89 -12.48,-53.39' fill='%239aa6e4'/%3E%3Cpolygon points='3.98,-43.89 -36.02,25.39 -40.35,22.89 -0.35,-46.39' fill='%239aa6e4'/%3E%3Cpolygon points='16.10,-36.89 -23.90,32.39 -28.23,29.89 11.77,-39.39' fill='%239aa6e4'/%3E%3Cpolygon points='28.23,-29.89 -11.77,39.39 -16.10,36.89 23.90,-32.39' fill='%239aa6e4'/%3E%3Cpolygon points='40.35,-22.89 0.35,46.39 -3.98,43.89 36.02,-25.39' fill='%239aa6e4'/%3E%3Cpolygon points='52.48,-15.89 12.48,53.39 8.15,50.89 48.15,-18.39' fill='%239aa6e4'/%3E%3Cpolygon points='64.60,-8.89 24.60,60.39 20.27,57.89 60.27,-11.39' fill='%239aa6e4'/%3E%3Cpolygon points='76.72,-1.89 36.72,67.39 32.39,64.89 72.39,-4.39' fill='%239aa6e4'/%3E%3Cpolygon points='88.85,5.11 48.85,74.39 44.52,71.89 84.52,2.61' fill='%239aa6e4'/%3E%3C/g%3E%3C/svg%3E\") , #eef5f4",
            "--plaque-fill": "#e2efed",
            "--plaque-border": "#4fb1a8",
            "--dgcv-table-shadow": "0 1px 4px rgba(20, 54, 56, 0.16), inset 0 1px 0 rgba(255, 255, 255, 0.5)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.3s ease",
        },
    ),
    "quilt_kagome": ThemeConfig(
        bg_primary="#f8f1e0",
        bg_surface="#efe6cf",
        bg_alt="#f3ecd7",
        bg_hover="#e9d9af",
        text_main="#36280f",
        text_heading="#93571f",
        text_hover="#36280f",
        text_alt="#6f6142",
        border_main="#d9c89e",
        border_alt="#e7dbbb",
        bg_action="#8f5a1e",
        text_on_action="#ffffff",
        bg_action_hover="#9c6422",
        bg_error="#b23a2a",
        text_on_error="#ffffff",
        bg_success="#5f7a2e",
        text_on_success="#ffffff",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-text": "#36280f",
            "--dgcv-special-background": "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16.17' height='28.00' viewBox='0 0 16.17 28.00'%3E%3Cdefs%3E%3CclipPath id='a'%3E%3Cpolygon points='40.00,-32.50 -40.00,-32.50 -40.00,-37.50 40.00,-37.50'/%3E%3Cpolygon points='40.00,-18.50 -40.00,-18.50 -40.00,-23.50 40.00,-23.50'/%3E%3Cpolygon points='40.00,-4.50 -40.00,-4.50 -40.00,-9.50 40.00,-9.50'/%3E%3Cpolygon points='40.00,9.50 -40.00,9.50 -40.00,4.50 40.00,4.50'/%3E%3Cpolygon points='40.00,23.50 -40.00,23.50 -40.00,18.50 40.00,18.50'/%3E%3Cpolygon points='40.00,37.50 -40.00,37.50 -40.00,32.50 40.00,32.50'/%3E%3Cpolygon points='40.00,51.50 -40.00,51.50 -40.00,46.50 40.00,46.50'/%3E%3Cpolygon points='40.00,65.50 -40.00,65.50 -40.00,60.50 40.00,60.50'/%3E%3C/clipPath%3E%3C/defs%3E%3Crect width='16.17' height='28.00' fill='%23f5edda'/%3E%3Cg%3E%3Cpolygon points='-20.27,-57.89 -60.27,11.39 -64.60,8.89 -24.60,-60.39' fill='%23d97c4a'/%3E%3Cpolygon points='-8.15,-50.89 -48.15,18.39 -52.48,15.89 -12.48,-53.39' fill='%23d97c4a'/%3E%3Cpolygon points='3.98,-43.89 -36.02,25.39 -40.35,22.89 -0.35,-46.39' fill='%23d97c4a'/%3E%3Cpolygon points='16.10,-36.89 -23.90,32.39 -28.23,29.89 11.77,-39.39' fill='%23d97c4a'/%3E%3Cpolygon points='28.23,-29.89 -11.77,39.39 -16.10,36.89 23.90,-32.39' fill='%23d97c4a'/%3E%3Cpolygon points='40.35,-22.89 0.35,46.39 -3.98,43.89 36.02,-25.39' fill='%23d97c4a'/%3E%3Cpolygon points='52.48,-15.89 12.48,53.39 8.15,50.89 48.15,-18.39' fill='%23d97c4a'/%3E%3Cpolygon points='64.60,-8.89 24.60,60.39 20.27,57.89 60.27,-11.39' fill='%23d97c4a'/%3E%3Cpolygon points='76.72,-1.89 36.72,67.39 32.39,64.89 72.39,-4.39' fill='%23d97c4a'/%3E%3Cpolygon points='88.85,5.11 48.85,74.39 44.52,71.89 84.52,2.61' fill='%23d97c4a'/%3E%3C/g%3E%3Cg%3E%3Cpolygon points='-72.39,-4.39 -32.39,64.89 -36.72,67.39 -76.72,-1.89' fill='%239ba858'/%3E%3Cpolygon points='-60.27,-11.39 -20.27,57.89 -24.60,60.39 -64.60,-8.89' fill='%239ba858'/%3E%3Cpolygon points='-48.15,-18.39 -8.15,50.89 -12.48,53.39 -52.48,-15.89' fill='%239ba858'/%3E%3Cpolygon points='-36.02,-25.39 3.98,43.89 -0.35,46.39 -40.35,-22.89' fill='%239ba858'/%3E%3Cpolygon points='-23.90,-32.39 16.10,36.89 11.77,39.39 -28.23,-29.89' fill='%239ba858'/%3E%3Cpolygon points='-11.77,-39.39 28.23,29.89 23.90,32.39 -16.10,-36.89' fill='%239ba858'/%3E%3Cpolygon points='0.35,-46.39 40.35,22.89 36.02,25.39 -3.98,-43.89' fill='%239ba858'/%3E%3Cpolygon points='12.48,-53.39 52.48,15.89 48.15,18.39 8.15,-50.89' fill='%239ba858'/%3E%3Cpolygon points='24.60,-60.39 64.60,8.89 60.27,11.39 20.27,-57.89' fill='%239ba858'/%3E%3Cpolygon points='36.72,-67.39 76.72,1.89 72.39,4.39 32.39,-64.89' fill='%239ba858'/%3E%3C/g%3E%3Cg%3E%3Cpolygon points='40.00,-32.50 -40.00,-32.50 -40.00,-37.50 40.00,-37.50' fill='%23ddb13f'/%3E%3Cpolygon points='40.00,-18.50 -40.00,-18.50 -40.00,-23.50 40.00,-23.50' fill='%23ddb13f'/%3E%3Cpolygon points='40.00,-4.50 -40.00,-4.50 -40.00,-9.50 40.00,-9.50' fill='%23ddb13f'/%3E%3Cpolygon points='40.00,9.50 -40.00,9.50 -40.00,4.50 40.00,4.50' fill='%23ddb13f'/%3E%3Cpolygon points='40.00,23.50 -40.00,23.50 -40.00,18.50 40.00,18.50' fill='%23ddb13f'/%3E%3Cpolygon points='40.00,37.50 -40.00,37.50 -40.00,32.50 40.00,32.50' fill='%23ddb13f'/%3E%3Cpolygon points='40.00,51.50 -40.00,51.50 -40.00,46.50 40.00,46.50' fill='%23ddb13f'/%3E%3Cpolygon points='40.00,65.50 -40.00,65.50 -40.00,60.50 40.00,60.50' fill='%23ddb13f'/%3E%3C/g%3E%3Cg clip-path='url(%23a)'%3E%3Cpolygon points='-20.27,-57.89 -60.27,11.39 -64.60,8.89 -24.60,-60.39' fill='%23d97c4a'/%3E%3Cpolygon points='-8.15,-50.89 -48.15,18.39 -52.48,15.89 -12.48,-53.39' fill='%23d97c4a'/%3E%3Cpolygon points='3.98,-43.89 -36.02,25.39 -40.35,22.89 -0.35,-46.39' fill='%23d97c4a'/%3E%3Cpolygon points='16.10,-36.89 -23.90,32.39 -28.23,29.89 11.77,-39.39' fill='%23d97c4a'/%3E%3Cpolygon points='28.23,-29.89 -11.77,39.39 -16.10,36.89 23.90,-32.39' fill='%23d97c4a'/%3E%3Cpolygon points='40.35,-22.89 0.35,46.39 -3.98,43.89 36.02,-25.39' fill='%23d97c4a'/%3E%3Cpolygon points='52.48,-15.89 12.48,53.39 8.15,50.89 48.15,-18.39' fill='%23d97c4a'/%3E%3Cpolygon points='64.60,-8.89 24.60,60.39 20.27,57.89 60.27,-11.39' fill='%23d97c4a'/%3E%3Cpolygon points='76.72,-1.89 36.72,67.39 32.39,64.89 72.39,-4.39' fill='%23d97c4a'/%3E%3Cpolygon points='88.85,5.11 48.85,74.39 44.52,71.89 84.52,2.61' fill='%23d97c4a'/%3E%3C/g%3E%3C/svg%3E\") , #f5edda",
            "--plaque-fill": "#efe6cf",
            "--plaque-border": "#ddb13f",
            "--dgcv-table-shadow": "0 1px 4px rgba(54, 40, 15, 0.16), inset 0 1px 0 rgba(255, 255, 255, 0.5)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.3s ease",
        },
    ),
    "quilt_kagome_dark": ThemeConfig(
        bg_primary="#181320",
        bg_surface="#2a2336",
        bg_alt="#221c2e",
        bg_hover="#3a3050",
        text_main="#ece3d6",
        text_heading="#d8b25a",
        text_hover="#f4ead9",
        text_alt="#a89db0",
        border_main="#463a5c",
        border_alt="#342b46",
        bg_action="#6a4a90",
        text_on_action="#ffffff",
        bg_action_hover="#7e5aa6",
        bg_error="#b53a48",
        text_on_error="#ffffff",
        bg_success="#2c8059",
        text_on_success="#ffffff",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-text": "#f4ead9",
            "--dgcv-special-background": "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16.17' height='28.00' viewBox='0 0 16.17 28.00'%3E%3Cdefs%3E%3CclipPath id='a'%3E%3Cpolygon points='40.00,-32.50 -40.00,-32.50 -40.00,-37.50 40.00,-37.50'/%3E%3Cpolygon points='40.00,-18.50 -40.00,-18.50 -40.00,-23.50 40.00,-23.50'/%3E%3Cpolygon points='40.00,-4.50 -40.00,-4.50 -40.00,-9.50 40.00,-9.50'/%3E%3Cpolygon points='40.00,9.50 -40.00,9.50 -40.00,4.50 40.00,4.50'/%3E%3Cpolygon points='40.00,23.50 -40.00,23.50 -40.00,18.50 40.00,18.50'/%3E%3Cpolygon points='40.00,37.50 -40.00,37.50 -40.00,32.50 40.00,32.50'/%3E%3Cpolygon points='40.00,51.50 -40.00,51.50 -40.00,46.50 40.00,46.50'/%3E%3Cpolygon points='40.00,65.50 -40.00,65.50 -40.00,60.50 40.00,60.50'/%3E%3C/clipPath%3E%3C/defs%3E%3Crect width='16.17' height='28.00' fill='%23201a28'/%3E%3Cg%3E%3Cpolygon points='-20.27,-57.89 -60.27,11.39 -64.60,8.89 -24.60,-60.39' fill='%238f6a29'/%3E%3Cpolygon points='-8.15,-50.89 -48.15,18.39 -52.48,15.89 -12.48,-53.39' fill='%238f6a29'/%3E%3Cpolygon points='3.98,-43.89 -36.02,25.39 -40.35,22.89 -0.35,-46.39' fill='%238f6a29'/%3E%3Cpolygon points='16.10,-36.89 -23.90,32.39 -28.23,29.89 11.77,-39.39' fill='%238f6a29'/%3E%3Cpolygon points='28.23,-29.89 -11.77,39.39 -16.10,36.89 23.90,-32.39' fill='%238f6a29'/%3E%3Cpolygon points='40.35,-22.89 0.35,46.39 -3.98,43.89 36.02,-25.39' fill='%238f6a29'/%3E%3Cpolygon points='52.48,-15.89 12.48,53.39 8.15,50.89 48.15,-18.39' fill='%238f6a29'/%3E%3Cpolygon points='64.60,-8.89 24.60,60.39 20.27,57.89 60.27,-11.39' fill='%238f6a29'/%3E%3Cpolygon points='76.72,-1.89 36.72,67.39 32.39,64.89 72.39,-4.39' fill='%238f6a29'/%3E%3Cpolygon points='88.85,5.11 48.85,74.39 44.52,71.89 84.52,2.61' fill='%238f6a29'/%3E%3C/g%3E%3Cg%3E%3Cpolygon points='-72.39,-4.39 -32.39,64.89 -36.72,67.39 -76.72,-1.89' fill='%231c7068'/%3E%3Cpolygon points='-60.27,-11.39 -20.27,57.89 -24.60,60.39 -64.60,-8.89' fill='%231c7068'/%3E%3Cpolygon points='-48.15,-18.39 -8.15,50.89 -12.48,53.39 -52.48,-15.89' fill='%231c7068'/%3E%3Cpolygon points='-36.02,-25.39 3.98,43.89 -0.35,46.39 -40.35,-22.89' fill='%231c7068'/%3E%3Cpolygon points='-23.90,-32.39 16.10,36.89 11.77,39.39 -28.23,-29.89' fill='%231c7068'/%3E%3Cpolygon points='-11.77,-39.39 28.23,29.89 23.90,32.39 -16.10,-36.89' fill='%231c7068'/%3E%3Cpolygon points='0.35,-46.39 40.35,22.89 36.02,25.39 -3.98,-43.89' fill='%231c7068'/%3E%3Cpolygon points='12.48,-53.39 52.48,15.89 48.15,18.39 8.15,-50.89' fill='%231c7068'/%3E%3Cpolygon points='24.60,-60.39 64.60,8.89 60.27,11.39 20.27,-57.89' fill='%231c7068'/%3E%3Cpolygon points='36.72,-67.39 76.72,1.89 72.39,4.39 32.39,-64.89' fill='%231c7068'/%3E%3C/g%3E%3Cg%3E%3Cpolygon points='40.00,-32.50 -40.00,-32.50 -40.00,-37.50 40.00,-37.50' fill='%23a23d54'/%3E%3Cpolygon points='40.00,-18.50 -40.00,-18.50 -40.00,-23.50 40.00,-23.50' fill='%23a23d54'/%3E%3Cpolygon points='40.00,-4.50 -40.00,-4.50 -40.00,-9.50 40.00,-9.50' fill='%23a23d54'/%3E%3Cpolygon points='40.00,9.50 -40.00,9.50 -40.00,4.50 40.00,4.50' fill='%23a23d54'/%3E%3Cpolygon points='40.00,23.50 -40.00,23.50 -40.00,18.50 40.00,18.50' fill='%23a23d54'/%3E%3Cpolygon points='40.00,37.50 -40.00,37.50 -40.00,32.50 40.00,32.50' fill='%23a23d54'/%3E%3Cpolygon points='40.00,51.50 -40.00,51.50 -40.00,46.50 40.00,46.50' fill='%23a23d54'/%3E%3Cpolygon points='40.00,65.50 -40.00,65.50 -40.00,60.50 40.00,60.50' fill='%23a23d54'/%3E%3C/g%3E%3Cg clip-path='url(%23a)'%3E%3Cpolygon points='-20.27,-57.89 -60.27,11.39 -64.60,8.89 -24.60,-60.39' fill='%238f6a29'/%3E%3Cpolygon points='-8.15,-50.89 -48.15,18.39 -52.48,15.89 -12.48,-53.39' fill='%238f6a29'/%3E%3Cpolygon points='3.98,-43.89 -36.02,25.39 -40.35,22.89 -0.35,-46.39' fill='%238f6a29'/%3E%3Cpolygon points='16.10,-36.89 -23.90,32.39 -28.23,29.89 11.77,-39.39' fill='%238f6a29'/%3E%3Cpolygon points='28.23,-29.89 -11.77,39.39 -16.10,36.89 23.90,-32.39' fill='%238f6a29'/%3E%3Cpolygon points='40.35,-22.89 0.35,46.39 -3.98,43.89 36.02,-25.39' fill='%238f6a29'/%3E%3Cpolygon points='52.48,-15.89 12.48,53.39 8.15,50.89 48.15,-18.39' fill='%238f6a29'/%3E%3Cpolygon points='64.60,-8.89 24.60,60.39 20.27,57.89 60.27,-11.39' fill='%238f6a29'/%3E%3Cpolygon points='76.72,-1.89 36.72,67.39 32.39,64.89 72.39,-4.39' fill='%238f6a29'/%3E%3Cpolygon points='88.85,5.11 48.85,74.39 44.52,71.89 84.52,2.61' fill='%238f6a29'/%3E%3C/g%3E%3C/svg%3E\") , #201a28",
            "--plaque-fill": "#2a2336",
            "--plaque-border": "#a23d54",
            "--dgcv-table-shadow": "0 1px 6px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.06)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.3s ease",
        },
    ),
    "quilt_gingham_red": ThemeConfig(
        bg_primary="#fdfbf6",
        bg_surface="#f9efe9",
        bg_alt="#fdf5f1",
        bg_hover="#f3cfca",
        text_main="#3a1414",
        text_heading="#a3242a",
        text_hover="#3a1414",
        text_alt="#6e4642",
        border_main="#e3c2bd",
        border_alt="#efd9d5",
        bg_action="#a81f22",
        text_on_action="#ffffff",
        bg_action_hover="#bd3033",
        bg_error="#8f1a20",
        text_on_error="#ffffff",
        bg_success="#2f7d4f",
        text_on_success="#ffffff",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-text": "#2a0d0d",
            "--dgcv-special-background": "repeating-linear-gradient(90deg, rgba(198, 40, 40, 0.62) 0 9px, transparent 9px 18px), repeating-linear-gradient(0deg, rgba(198, 40, 40, 0.62) 0 9px, transparent 9px 18px), #fcfaf5",
            "--plaque-fill": "#f9efe9",
            "--plaque-border": "#3a1414",
            "--dgcv-table-shadow": "0 1px 4px rgba(105, 25, 25, 0.16), inset 0 1px 0 rgba(255, 255, 255, 0.6)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.3s ease",
        },
    ),
    "quilt_gingham_blue": ThemeConfig(
        bg_primary="#fbfcfe",
        bg_surface="#eaf1f9",
        bg_alt="#f3f7fc",
        bg_hover="#cbdcf0",
        text_main="#122035",
        text_heading="#244c86",
        text_hover="#122035",
        text_alt="#4a596f",
        border_main="#c2d2e8",
        border_alt="#d9e4f1",
        bg_action="#295a9c",
        text_on_action="#ffffff",
        bg_action_hover="#356cb0",
        bg_error="#b5302a",
        text_on_error="#ffffff",
        bg_success="#2f7d52",
        text_on_success="#ffffff",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-text": "#0e2138",
            "--dgcv-special-background": "repeating-linear-gradient(90deg, rgba(58, 110, 165, 0.62) 0 9px, transparent 9px 18px), repeating-linear-gradient(0deg, rgba(58, 110, 165, 0.62) 0 9px, transparent 9px 18px), #fcfaf5",
            "--plaque-fill": "#eaf1f9",
            "--plaque-border": "#244c86",
            "--dgcv-table-shadow": "0 1px 4px rgba(18, 32, 53, 0.16), inset 0 1px 0 rgba(255, 255, 255, 0.6)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.3s ease",
        },
    ),
    "plaid_meadow": ThemeConfig(
        bg_primary="#f5f1e6",
        bg_surface="#e8eee0",
        bg_alt="#f0f3e8",
        bg_hover="#cfe0cc",
        text_main="#23311f",
        text_heading="#356048",
        text_hover="#23311f",
        text_alt="#4c5c46",
        border_main="#c2d4bc",
        border_alt="#dbe6d4",
        bg_action="#2f5f9c",
        text_on_action="#ffffff",
        bg_action_hover="#3a70b0",
        bg_error="#b23a2a",
        text_on_error="#ffffff",
        bg_success="#2f7d4f",
        text_on_success="#ffffff",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-text": "#23311f",
            "--dgcv-special-background": "repeating-linear-gradient(45deg, rgba(255,255,255,0.06) 0 1px, rgba(0,0,0,0.06) 1px 2px, transparent 2px 4px), repeating-linear-gradient(90deg, rgba(159,184,151,0.5) 0px 16px, rgba(241,236,224,0.5) 16px 24px, rgba(251,251,246,0.5) 24px 26px, rgba(241,236,224,0.5) 26px 34px, rgba(157,178,214,0.5) 34px 50px, rgba(241,236,224,0.5) 50px 58px), repeating-linear-gradient(0deg, #9fb897 0px 16px, #f1ece0 16px 24px, #fbfbf6 24px 26px, #f1ece0 26px 34px, #9db2d6 34px 50px, #f1ece0 50px 58px), #f1ece0",
            "--plaque-fill": "#e8eee0",
            "--plaque-border": "#356048",
            "--dgcv-table-shadow": "0 1px 4px rgba(35, 49, 31, 0.14), inset 0 1px 0 rgba(255, 255, 255, 0.55)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.3s ease",
        },
    ),
    "plaid_camel": ThemeConfig(
        bg_primary="#f3ead7",
        bg_surface="#ece0c6",
        bg_alt="#f1e8d4",
        bg_hover="#dfca9f",
        text_main="#2e2414",
        text_heading="#8a4a20",
        text_hover="#2e2414",
        text_alt="#6b5836",
        border_main="#d6c39c",
        border_alt="#e6d8b8",
        bg_action="#8a5226",
        text_on_action="#ffffff",
        bg_action_hover="#9c6230",
        bg_error="#a83321",
        text_on_error="#ffffff",
        bg_success="#5a7029",
        text_on_success="#ffffff",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-text": "#2e2414",
            "--dgcv-special-background": "repeating-linear-gradient(45deg, rgba(255,255,255,0.06) 0 1px, rgba(0,0,0,0.06) 1px 2px, transparent 2px 4px), repeating-linear-gradient(90deg, rgba(231,217,191,0.5) 0px 14px, rgba(154,145,82,0.5) 14px 26px, rgba(243,235,218,0.5) 26px 34px, rgba(200,138,99,0.5) 34px 46px, rgba(243,235,218,0.5) 46px 52px, rgba(184,122,74,0.5) 52px 54px, rgba(243,235,218,0.5) 54px 60px), repeating-linear-gradient(0deg, #e7d9bf 0px 14px, #9a9152 14px 26px, #f3ebda 26px 34px, #c88a63 34px 46px, #f3ebda 46px 52px, #b87a4a 52px 54px, #f3ebda 54px 60px), #e7d9bf",
            "--plaque-fill": "#ece0c6",
            "--plaque-border": "#8a4a20",
            "--dgcv-table-shadow": "0 1px 4px rgba(46, 36, 20, 0.14), inset 0 1px 0 rgba(255, 255, 255, 0.55)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.3s ease",
        },
    ),
    "plaid_blackwatch": ThemeConfig(
        bg_primary="#12151c",
        bg_surface="#1e2530",
        bg_alt="#181d26",
        bg_hover="#2c384a",
        text_main="#e6e8ea",
        text_heading="#86abca",
        text_hover="#f0f2f4",
        text_alt="#9aa6b2",
        border_main="#344253",
        border_alt="#283341",
        bg_action="#2f5f9c",
        text_on_action="#ffffff",
        bg_action_hover="#3a70b0",
        bg_error="#c0453c",
        text_on_error="#ffffff",
        bg_success="#2f7d4f",
        text_on_success="#ffffff",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-text": "#e6e8ea",
            "--dgcv-special-background": "repeating-linear-gradient(45deg, rgba(255,255,255,0.06) 0 1px, rgba(0,0,0,0.06) 1px 2px, transparent 2px 4px), repeating-linear-gradient(90deg, rgba(39,64,106,0.5) 0px 18px, rgba(22,26,34,0.5) 18px 26px, rgba(35,74,48,0.5) 26px 44px, rgba(22,26,34,0.5) 44px 52px), repeating-linear-gradient(0deg, #27406a 0px 18px, #161a22 18px 26px, #234a30 26px 44px, #161a22 44px 52px), #161a22",
            "--plaque-fill": "#1e2530",
            "--plaque-border": "#86abca",
            "--dgcv-table-shadow": "0 1px 6px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.06)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.3s ease",
        },
    ),
    "plaid_claret": ThemeConfig(
        bg_primary="#160f14",
        bg_surface="#241820",
        bg_alt="#1c131a",
        bg_hover="#3a2630",
        text_main="#f0e6d8",
        text_heading="#cf9442",
        text_hover="#f6eee0",
        text_alt="#b3a08f",
        border_main="#45323c",
        border_alt="#2e2028",
        bg_action="#933243",
        text_on_action="#ffffff",
        bg_action_hover="#a93e52",
        bg_error="#c0453c",
        text_on_error="#ffffff",
        bg_success="#2f7d4f",
        text_on_success="#ffffff",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-text": "#f0e6d8",
            "--dgcv-special-background": "repeating-linear-gradient(45deg, rgba(255,255,255,0.06) 0 1px, rgba(0,0,0,0.06) 1px 2px, transparent 2px 4px), repeating-linear-gradient(90deg, rgba(94,32,48,0.5) 0px 18px, rgba(24,16,22,0.5) 18px 24px, rgba(31,58,40,0.5) 24px 42px, rgba(24,16,22,0.5) 42px 48px, rgba(138,101,38,0.5) 48px 50px, rgba(24,16,22,0.5) 50px 54px), repeating-linear-gradient(0deg, #5e2030 0px 18px, #181016 18px 24px, #1f3a28 24px 42px, #181016 42px 48px, #8a6526 48px 50px, #181016 50px 54px), #181016",
            "--plaque-fill": "#241820",
            "--plaque-border": "#cf9442",
            "--dgcv-table-shadow": "0 1px 6px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.05)",
            "--dgcv-hover-transform": "scale(1.001)",
            "--dgcv-hover-transition": "all 0.3s ease",
        },
    ),
    "meadow_day": ThemeConfig(
        bg_primary="#FFFFFF",
        bg_surface="#ECE9D8",
        bg_alt="#F2F0E4",
        bg_hover="#316AC5",
        text_main="#000000",
        text_heading="#0A246A",
        text_hover="#FFFFFF",
        text_alt="#171713",
        border_main="#63819F",
        border_alt="#C8C4B4",
        bg_action="#245EDC",
        text_on_action="#FFFFFF",
        bg_action_hover="#0B3FA8",
        bg_error="#B0362B",
        text_on_error="#FFFFFF",
        bg_success="#3B7A28",
        text_on_success="#FFFFFF",
        font_family="Tahoma, 'Trebuchet MS', Verdana, Geneva, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": (
                'url("data:image/svg+xml,'
                "%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160' viewBox='0 0 160 160'%3E %3Cf"
                "ilter id='g' x='0' y='0' width='100%' height='100%' color-interpolation-filters='sRGB'%3E %3CfeT"
                "urbulence type='fractalNoise' baseFrequency='0.8' numOctaves='2' seed='5' stitchTiles='stitch' r"
                "esult='n'/%3E %3CfeColorMatrix in='n' type='matrix' values='0 0 0 0 0.5 0 0 0 0 0.5 0 0 0 0 0.5 "
                "0.30 0 0 0 -0.14'/%3E %3C/filter%3E %3Crect width='160' height='160' filter='url%28%23g%29'/%3E "
                "%3C/svg%3E"
                '") 0 0 / 160px 160px repeat, '
                'url("data:image/svg+xml,'
                "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 800 500' preserveAspectRatio='xMidYMax sl"
                "ice'%3E %3Cdefs%3E %3ClinearGradient id='sky' x1='0' y1='0' x2='0' y2='1'%3E %3Cstop offset='0' "
                "stop-color='%231E62AC'/%3E %3Cstop offset='0.30' stop-color='%232E76C0'/%3E %3Cstop offset='0.58"
                "' stop-color='%235F9FD6'/%3E %3Cstop offset='0.80' stop-color='%23A8CCE8'/%3E %3Cstop offset='1'"
                " stop-color='%23D8E7F2'/%3E %3C/linearGradient%3E %3ClinearGradient id='haze' x1='0' y1='0' x2='"
                "0' y2='1'%3E %3Cstop offset='0' stop-color='%23DCEAF5' stop-opacity='0'/%3E %3Cstop offset='1' s"
                "top-color='%23DCEAF5' stop-opacity='0.62'/%3E %3C/linearGradient%3E %3ClinearGradient id='turf' "
                "gradientUnits='userSpaceOnUse' x1='0' y1='272' x2='0' y2='470'%3E %3Cstop offset='0' stop-color="
                "'%23B3CE73'/%3E %3Cstop offset='0.16' stop-color='%238CB656'/%3E %3Cstop offset='0.52' stop-colo"
                "r='%236A9842'/%3E %3Cstop offset='1' stop-color='%23496E30'/%3E %3C/linearGradient%3E %3ClinearG"
                "radient id='far' gradientUnits='userSpaceOnUse' x1='0' y1='326' x2='0' y2='430'%3E %3Cstop offse"
                "t='0' stop-color='%23A3BA90'/%3E %3Cstop offset='1' stop-color='%238AA06D'/%3E %3C/linearGradien"
                "t%3E %3ClinearGradient id='cfade' x1='0' y1='0' x2='0' y2='1'%3E %3Cstop offset='0' stop-color='"
                "%23FFFFFF' stop-opacity='0.55'/%3E %3Cstop offset='0.16' stop-color='%23FFFFFF' stop-opacity='1'"
                "/%3E %3Cstop offset='0.68' stop-color='%23FFFFFF' stop-opacity='1'/%3E %3Cstop offset='0.98' sto"
                "p-color='%23000000' stop-opacity='0'/%3E %3C/linearGradient%3E %3Cmask id='cmask'%3E%3Crect x='0"
                "' y='0' width='800' height='340' fill='url%28%23cfade%29'/%3E%3C/mask%3E %3Cfilter id='cloud' x="
                "'0' y='0' width='100%' height='100%' color-interpolation-filters='sRGB'%3E %3CfeTurbulence type="
                "'fractalNoise' baseFrequency='0.0055 0.019' numOctaves='5' seed='3' stitchTiles='stitch' result="
                "'n'/%3E %3CfeColorMatrix in='n' type='matrix' values='0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 -2.9 0 0 0 1"
                ".62' result='m'/%3E %3CfeComposite in='m' in2='SourceGraphic' operator='in'/%3E %3C/filter%3E %3"
                "Cfilter id='cshade' x='0' y='0' width='100%' height='100%' color-interpolation-filters='sRGB'%3E"
                " %3CfeTurbulence type='fractalNoise' baseFrequency='0.0055 0.019' numOctaves='5' seed='3' stitch"
                "Tiles='stitch' result='n'/%3E %3CfeColorMatrix in='n' type='matrix' values='0 0 0 0 0.72 0 0 0 0"
                " 0.76 0 0 0 0 0.82 -2.2 0 0 0 1.05' result='m'/%3E %3CfeOffset in='m' dx='0' dy='9' result='o'/%"
                "3E %3CfeComposite in='o' in2='SourceGraphic' operator='in'/%3E %3C/filter%3E %3Cfilter id='mottl"
                "e' x='0' y='0' width='100%' height='100%' color-interpolation-filters='sRGB'%3E %3CfeTurbulence "
                "type='fractalNoise' baseFrequency='0.014 0.045' numOctaves='4' seed='11' stitchTiles='stitch' re"
                "sult='n'/%3E %3CfeColorMatrix in='n' type='matrix' values='0 0 0 0 0.20 0 0 0 0 0.29 0 0 0 0 0.1"
                "1 0.62 0 0 0 -0.20' result='m'/%3E %3CfeComposite in='m' in2='SourceGraphic' operator='in'/%3E %"
                "3C/filter%3E %3Cfilter id='blades' x='0' y='0' width='100%' height='100%' color-interpolation-fi"
                "lters='sRGB'%3E %3CfeTurbulence type='fractalNoise' baseFrequency='0.09 0.55' numOctaves='3' see"
                "d='19' stitchTiles='stitch' result='n'/%3E %3CfeColorMatrix in='n' type='matrix' values='0 0 0 0"
                " 0.83 0 0 0 0 0.90 0 0 0 0 0.55 0.50 0 0 0 -0.22' result='m'/%3E %3CfeComposite in='m' in2='Sour"
                "ceGraphic' operator='in'/%3E %3C/filter%3E %3ClinearGradient id='near' gradientUnits='userSpaceO"
                "nUse' x1='0' y1='318' x2='0' y2='504'%3E %3Cstop offset='0' stop-color='%2360843A'/%3E %3Cstop o"
                "ffset='0.22' stop-color='%23547630'/%3E %3Cstop offset='0.60' stop-color='%2342602A'/%3E %3Cstop"
                " offset='1' stop-color='%23344C22'/%3E %3C/linearGradient%3E %3Cfilter id='nmottle' x='0' y='0' "
                "width='100%' height='100%' color-interpolation-filters='sRGB'%3E %3CfeTurbulence type='fractalNo"
                "ise' baseFrequency='0.010 0.032' numOctaves='4' seed='23' stitchTiles='stitch' result='n'/%3E %3"
                "CfeColorMatrix in='n' type='matrix' values='0 0 0 0 0.15 0 0 0 0 0.23 0 0 0 0 0.08 0.58 0 0 0 -0"
                ".19' result='m'/%3E %3CfeComposite in='m' in2='SourceGraphic' operator='in'/%3E %3C/filter%3E %3"
                "Cfilter id='nblades' x='0' y='0' width='100%' height='100%' color-interpolation-filters='sRGB'%3"
                "E %3CfeTurbulence type='fractalNoise' baseFrequency='0.055 0.32' numOctaves='3' seed='29' stitch"
                "Tiles='stitch' result='n'/%3E %3CfeColorMatrix in='n' type='matrix' values='0 0 0 0 0.74 0 0 0 0"
                " 0.84 0 0 0 0 0.48 0.46 0 0 0 -0.20' result='m'/%3E %3CfeComposite in='m' in2='SourceGraphic' op"
                "erator='in'/%3E %3C/filter%3E %3CclipPath id='nearclip'%3E %3Cpath d='M -4,463 C 130,443 280,417"
                " 400,395 C 530,371 690,344 804,320 L 804,504 L -4,504 Z'/%3E %3C/clipPath%3E %3CclipPath id='hil"
                "lclip'%3E %3Cpath d='M -4,318 C 52,296 122,278 212,277 C 332,276 420,300 520,320 C 630,342 720,3"
                "60 804,378 L 804,504 L -4,504 Z'/%3E %3C/clipPath%3E %3C/defs%3E %3Crect x='0' y='0' width='800'"
                " height='500' fill='url%28%23sky%29'/%3E %3Crect x='0' y='0' width='800' height='340' fill='%23F"
                "FFFFF' filter='url%28%23cshade%29' mask='url%28%23cmask%29' opacity='0.85'/%3E %3Crect x='0' y='"
                "0' width='800' height='340' fill='%23FFFFFF' filter='url%28%23cloud%29' mask='url%28%23cmask%29'"
                "/%3E %3Crect x='0' y='240' width='800' height='160' fill='url%28%23haze%29'/%3E %3Cpath d='M -4,"
                "352 C 88,338 200,330 300,330 C 432,330 560,336 804,349 L 804,504 L -4,504 Z' fill='url%28%23far%"
                "29'/%3E %3Cpath d='M -4,352 C 88,338 200,330 300,330 C 432,330 560,336 804,349' fill='none' stro"
                "ke='%23C2D4A8' stroke-width='2' opacity='0.5'/%3E %3Cpath d='M -4,318 C 52,296 122,278 212,277 C"
                " 332,276 420,300 520,320 C 630,342 720,360 804,378 L 804,504 L -4,504 Z' fill='url%28%23turf%29'"
                "/%3E %3Cg clip-path='url%28%23hillclip%29'%3E %3Crect x='0' y='260' width='800' height='244' fil"
                "l='%23FFFFFF' filter='url%28%23mottle%29'/%3E %3Crect x='0' y='260' width='800' height='244' fil"
                "l='%23FFFFFF' filter='url%28%23blades%29' opacity='0.55'/%3E %3Cpath d='M -4,318 C 52,296 122,27"
                "8 212,277 C 332,276 420,300 520,320 C 630,342 720,360 804,378' fill='none' stroke='%23D2E39B' st"
                "roke-width='7' opacity='0.35'/%3E %3Cpath d='M -4,318 C 52,296 122,278 212,277 C 332,276 420,300"
                " 520,320 C 630,342 720,360 804,378' fill='none' stroke='%23E2EEB4' stroke-width='2.5' opacity='0"
                ".55'/%3E %3Crect x='0' y='430' width='800' height='74' fill='%232F4A20' opacity='0.16'/%3E %3C/g"
                "%3E %3Cpath d='M -4,463 C 130,443 280,417 400,395 C 530,371 690,344 804,320 L 804,504 L -4,504 Z"
                "' fill='url%28%23near%29'/%3E %3Cg clip-path='url%28%23nearclip%29'%3E %3Crect x='0' y='300' wid"
                "th='800' height='204' fill='%23FFFFFF' filter='url%28%23nmottle%29'/%3E %3Crect x='0' y='300' wi"
                "dth='800' height='204' fill='%23FFFFFF' filter='url%28%23nblades%29' opacity='0.5'/%3E %3Cpath d"
                "='M -4,463 C 130,443 280,417 400,395 C 530,371 690,344 804,320' fill='none' stroke='%23A9C95C' s"
                "troke-width='8' opacity='0.22'/%3E %3Cpath d='M -4,463 C 130,443 280,417 400,395 C 530,371 690,3"
                "44 804,320' fill='none' stroke='%23C3DC84' stroke-width='2.5' opacity='0.5'/%3E %3C/g%3E %3C/svg"
                "%3E"
                '") center 58% / cover no-repeat, '
                "linear-gradient(to bottom, #1E62AC 0%, #2E76C0 30%, #5F9FD6 58%, "
                "#A8CCE8 80%, #D8E7F2 100%) center / cover no-repeat, "
                "#5F9FD6"
            ),
            "--dgcv-special-text": "#0A246A",
            "--plaque-fill": "#ECE9D8",
            "--plaque-border": "#0A246A",
            "--dgcv-table-shadow": "0 2px 6px rgba(10, 36, 106, 0.28)",
            "--dgcv-text-shadow": "none",
            "--dgcv-hover-transform": "none",
            "--dgcv-hover-transition": "background-color 90ms linear, color 90ms linear",
        },
    ),
    "meadow_dusk": ThemeConfig(
        bg_primary="#131A28",
        bg_surface="#1D2739",
        bg_alt="#17202F",
        bg_hover="#7FB0EA",
        text_main="#DEE5F0",
        text_heading="#F2C078",
        text_hover="#0E1522",
        text_alt="#D6DDE9",
        border_main="#6B7C97",
        border_alt="#333E55",
        bg_action="#6FA3E4",
        text_on_action="#0E1522",
        bg_action_hover="#8FBBF0",
        bg_error="#A6453C",
        text_on_error="#FFEDE8",
        bg_success="#3E6B34",
        text_on_success="#E6F2DE",
        font_family="Tahoma, 'Trebuchet MS', Verdana, Geneva, sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": (
                'url("data:image/svg+xml,'
                "%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160' viewBox='0 0 160 160'%3E %3"
                "Cfilter id='g' x='0' y='0' width='100%' height='100%' color-interpolation-filters='sRGB'%3E %3"
                "CfeTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='2' seed='5' stitchTiles='sti"
                "tch' result='n'/%3E %3CfeColorMatrix in='n' type='matrix' values='0 0 0 0 0.5 0 0 0 0 0.5 0 0 "
                "0 0 0.5 0.34 0 0 0 -0.16'/%3E %3C/filter%3E %3Crect width='160' height='160' filter='url%28%23"
                "g%29'/%3E %3C/svg%3E"
                '") 0 0 / 160px 160px repeat, '
                'url("data:image/svg+xml,'
                "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 800 500' preserveAspectRatio='xMidYMax "
                "slice'%3E %3Cdefs%3E %3ClinearGradient id='sky' x1='0' y1='0' x2='0' y2='1'%3E %3Cstop offset="
                "'0' stop-color='%23101833'/%3E %3Cstop offset='0.28' stop-color='%231E2748'/%3E %3Cstop offset"
                "='0.52' stop-color='%233A3560'/%3E %3Cstop offset='0.70' stop-color='%236E4462'/%3E %3Cstop of"
                "fset='0.86' stop-color='%23B45F53'/%3E %3Cstop offset='1' stop-color='%23E89A5C'/%3E %3C/linea"
                "rGradient%3E %3CradialGradient id='glow' cx='0.5' cy='0.5' r='0.5'%3E %3Cstop offset='0' stop-"
                "color='%23FFC078' stop-opacity='0.70'/%3E %3Cstop offset='0.45' stop-color='%23F09A5E' stop-op"
                "acity='0.34'/%3E %3Cstop offset='1' stop-color='%23E8834E' stop-opacity='0'/%3E %3C/radialGrad"
                "ient%3E %3ClinearGradient id='cloudcol' gradientUnits='userSpaceOnUse' x1='0' y1='24' x2='0' y"
                "2='352'%3E %3Cstop offset='0' stop-color='%232E3352'/%3E %3Cstop offset='0.42' stop-color='%23"
                "5B4666'/%3E %3Cstop offset='0.74' stop-color='%23B06A5C'/%3E %3Cstop offset='1' stop-color='%2"
                "3F0A96E'/%3E %3C/linearGradient%3E %3ClinearGradient id='haze' x1='0' y1='0' x2='0' y2='1'%3E "
                "%3Cstop offset='0' stop-color='%23E8945C' stop-opacity='0'/%3E %3Cstop offset='1' stop-color='"
                "%23E8945C' stop-opacity='0.34'/%3E %3C/linearGradient%3E %3ClinearGradient id='turf' gradientU"
                "nits='userSpaceOnUse' x1='0' y1='272' x2='0' y2='470'%3E %3Cstop offset='0' stop-color='%23415"
                "22F'/%3E %3Cstop offset='0.16' stop-color='%23334325'/%3E %3Cstop offset='0.52' stop-color='%2"
                "325341C'/%3E %3Cstop offset='1' stop-color='%231A2614'/%3E %3C/linearGradient%3E %3ClinearGrad"
                "ient id='far' gradientUnits='userSpaceOnUse' x1='0' y1='326' x2='0' y2='430'%3E %3Cstop offset"
                "='0' stop-color='%236B6274'/%3E %3Cstop offset='1' stop-color='%23544E63'/%3E %3C/linearGradie"
                "nt%3E %3ClinearGradient id='near' gradientUnits='userSpaceOnUse' x1='0' y1='318' x2='0' y2='50"
                "4'%3E %3Cstop offset='0' stop-color='%231B2614'/%3E %3Cstop offset='0.22' stop-color='%23151E1"
                "0'/%3E %3Cstop offset='0.60' stop-color='%2310170C'/%3E %3Cstop offset='1' stop-color='%230B11"
                "08'/%3E %3C/linearGradient%3E %3ClinearGradient id='mglow' gradientUnits='userSpaceOnUse' x1='"
                "0' y1='240' x2='0' y2='470'%3E%3Cstop offset='0' stop-color='%23FFB86E' stop-opacity='0'/%3E%3"
                "Cstop offset='0.17' stop-color='%23FFB86E' stop-opacity='0.30'/%3E%3Cstop offset='0.45' stop-c"
                "olor='%23F09A5E' stop-opacity='0.12'/%3E%3Cstop offset='1' stop-color='%23E8834E' stop-opacity"
                "='0'/%3E%3C/linearGradient%3E %3ClinearGradient id='cfade' x1='0' y1='0' x2='0' y2='1'%3E %3Cs"
                "top offset='0' stop-color='%23FFFFFF' stop-opacity='0.6'/%3E %3Cstop offset='0.18' stop-color="
                "'%23FFFFFF' stop-opacity='1'/%3E %3Cstop offset='0.80' stop-color='%23FFFFFF' stop-opacity='1'"
                "/%3E %3Cstop offset='1' stop-color='%23000000' stop-opacity='0'/%3E %3C/linearGradient%3E %3Cm"
                "ask id='cmask'%3E%3Crect x='0' y='0' width='800' height='352' fill='url%28%23cfade%29'/%3E%3C/"
                "mask%3E %3Cfilter id='cloud' x='0' y='0' width='100%' height='100%' color-interpolation-filter"
                "s='sRGB'%3E %3CfeTurbulence type='fractalNoise' baseFrequency='0.0055 0.019' numOctaves='5' se"
                "ed='3' stitchTiles='stitch' result='n'/%3E %3CfeColorMatrix in='n' type='matrix' values='0 0 0"
                " 0 1 0 0 0 0 1 0 0 0 0 1 -2.9 0 0 0 1.62' result='m'/%3E %3CfeComposite in='SourceGraphic' in2"
                "='m' operator='in'/%3E %3C/filter%3E %3Cfilter id='cshade' x='0' y='0' width='100%' height='10"
                "0%' color-interpolation-filters='sRGB'%3E %3CfeTurbulence type='fractalNoise' baseFrequency='0"
                ".0055 0.019' numOctaves='5' seed='3' stitchTiles='stitch' result='n'/%3E %3CfeColorMatrix in='"
                "n' type='matrix' values='0 0 0 0 0.10 0 0 0 0 0.10 0 0 0 0 0.18 -2.2 0 0 0 1.05' result='m'/%3"
                "E %3CfeOffset in='m' dx='0' dy='-8' result='o'/%3E %3CfeComposite in='o' in2='SourceGraphic' o"
                "perator='in'/%3E %3C/filter%3E %3Cfilter id='mottle' x='0' y='0' width='100%' height='100%' co"
                "lor-interpolation-filters='sRGB'%3E %3CfeTurbulence type='fractalNoise' baseFrequency='0.014 0"
                ".045' numOctaves='4' seed='11' stitchTiles='stitch' result='n'/%3E %3CfeColorMatrix in='n' typ"
                "e='matrix' values='0 0 0 0 0.05 0 0 0 0 0.08 0 0 0 0 0.04 0.60 0 0 0 -0.20' result='m'/%3E %3C"
                "feComposite in='m' in2='SourceGraphic' operator='in'/%3E %3C/filter%3E %3Cfilter id='blades' x"
                "='0' y='0' width='100%' height='100%' color-interpolation-filters='sRGB'%3E %3CfeTurbulence ty"
                "pe='fractalNoise' baseFrequency='0.09 0.55' numOctaves='3' seed='19' stitchTiles='stitch' resu"
                "lt='n'/%3E %3CfeColorMatrix in='n' type='matrix' values='0 0 0 0 0.92 0 0 0 0 0.68 0 0 0 0 0.3"
                "8 0.42 0 0 0 -0.22' result='m'/%3E %3CfeComposite in='m' in2='SourceGraphic' operator='in'/%3E"
                " %3C/filter%3E %3Cfilter id='nmottle' x='0' y='0' width='100%' height='100%' color-interpolati"
                "on-filters='sRGB'%3E %3CfeTurbulence type='fractalNoise' baseFrequency='0.010 0.032' numOctave"
                "s='4' seed='23' stitchTiles='stitch' result='n'/%3E %3CfeColorMatrix in='n' type='matrix' valu"
                "es='0 0 0 0 0.03 0 0 0 0 0.05 0 0 0 0 0.02 0.58 0 0 0 -0.19' result='m'/%3E %3CfeComposite in="
                "'m' in2='SourceGraphic' operator='in'/%3E %3C/filter%3E %3Cfilter id='nblades' x='0' y='0' wid"
                "th='100%' height='100%' color-interpolation-filters='sRGB'%3E %3CfeTurbulence type='fractalNoi"
                "se' baseFrequency='0.055 0.32' numOctaves='3' seed='29' stitchTiles='stitch' result='n'/%3E %3"
                "CfeColorMatrix in='n' type='matrix' values='0 0 0 0 0.62 0 0 0 0 0.45 0 0 0 0 0.24 0.40 0 0 0 "
                "-0.22' result='m'/%3E %3CfeComposite in='m' in2='SourceGraphic' operator='in'/%3E %3C/filter%3"
                "E %3CclipPath id='nearclip'%3E %3Cpath d='M -4,463 C 130,443 280,417 400,395 C 530,371 690,344"
                " 804,320 L 804,504 L -4,504 Z'/%3E %3C/clipPath%3E %3CclipPath id='hillclip'%3E %3Cpath d='M -"
                "4,318 C 52,296 122,278 212,277 C 332,276 420,300 520,320 C 630,342 720,360 804,378 L 804,504 L"
                " -4,504 Z'/%3E %3C/clipPath%3E %3C/defs%3E %3Crect x='0' y='0' width='800' height='500' fill='"
                "url%28%23sky%29'/%3E %3Cellipse cx='300' cy='330' rx='430' ry='150' fill='url%28%23glow%29'/%3"
                "E %3Crect x='0' y='0' width='800' height='352' fill='%230C1026' filter='url%28%23cshade%29' ma"
                "sk='url%28%23cmask%29' opacity='0.8'/%3E %3Crect x='0' y='0' width='800' height='352' fill='ur"
                "l%28%23cloudcol%29' filter='url%28%23cloud%29' mask='url%28%23cmask%29'/%3E %3Crect x='0' y='2"
                "50' width='800' height='160' fill='url%28%23haze%29'/%3E %3Cpath d='M -4,352 C 88,338 200,330 "
                "300,330 C 432,330 560,336 804,349 L 804,504 L -4,504 Z' fill='url%28%23far%29'/%3E %3Cpath d='"
                "M -4,352 C 88,338 200,330 300,330 C 432,330 560,336 804,349' fill='none' stroke='%23E0A882' st"
                "roke-width='2' opacity='0.45'/%3E %3Cpath d='M -4,318 C 52,296 122,278 212,277 C 332,276 420,3"
                "00 520,320 C 630,342 720,360 804,378 L 804,504 L -4,504 Z' fill='url%28%23turf%29'/%3E %3Cg cl"
                "ip-path='url%28%23hillclip%29'%3E %3Crect x='0' y='260' width='800' height='244' fill='%23FFFF"
                "FF' filter='url%28%23mottle%29'/%3E %3Crect x='0' y='260' width='800' height='244' fill='%23FF"
                "FFFF' filter='url%28%23blades%29' opacity='0.30'/%3E %3Cpath d='M -4,318 C 52,296 122,278 212,"
                "277 C 332,276 420,300 520,320 C 630,342 720,360 804,378' fill='none' stroke='%23F0A45E' stroke"
                "-width='6' opacity='0.28'/%3E %3Cpath d='M -4,318 C 52,296 122,278 212,277 C 332,276 420,300 5"
                "20,320 C 630,342 720,360 804,378' fill='none' stroke='%23FFCE92' stroke-width='2' opacity='0.5"
                "5'/%3E %3C/g%3E %3Cpath d='M -4,463 C 130,443 280,417 400,395 C 530,371 690,344 804,320 L 804,"
                "504 L -4,504 Z' fill='url%28%23near%29'/%3E %3Cg clip-path='url%28%23nearclip%29'%3E %3Crect x"
                "='0' y='300' width='800' height='204' fill='%23FFFFFF' filter='url%28%23nmottle%29'/%3E %3Crec"
                "t x='0' y='300' width='800' height='204' fill='%23FFFFFF' filter='url%28%23nblades%29' opacity"
                "='0.28'/%3E %3Cpath d='M -4,463 C 130,443 280,417 400,395 C 530,371 690,344 804,320' fill='non"
                "e' stroke='%23E09455' stroke-width='7' opacity='0.20'/%3E %3Cpath d='M -4,463 C 130,443 280,41"
                "7 400,395 C 530,371 690,344 804,320' fill='none' stroke='%23F7BC7C' stroke-width='2' opacity='"
                "0.45'/%3E %3C/g%3E %3Crect x='0' y='240' width='800' height='264' fill='url%28%23mglow%29'/%3E"
                " %3C/svg%3E"
                '") center 58% / cover no-repeat, '
                "linear-gradient(to bottom, #101833 0%, #1E2748 28%, #3A3560 52%, "
                "#6E4462 70%, #B45F53 86%, #E89A5C 100%) center / cover no-repeat, "
                "#3A3560"
            ),
            "--dgcv-special-text": "#F2C078",
            "--plaque-fill": "#1D2739",
            "--plaque-border": "#F2C078",
            "--dgcv-table-shadow": "0 2px 10px rgba(0, 0, 0, 0.55)",
            "--dgcv-text-shadow": "none",
            "--dgcv-hover-transform": "none",
            "--dgcv-hover-transition": "background-color 90ms linear, color 90ms linear",
        },
    ),
    "clownfish_reef": ThemeConfig(
        bg_primary="#0A1E2B",
        bg_surface="#123044",
        bg_alt="#0D2433",
        bg_hover="#F5714C",
        text_main="#DCE9F0",
        text_heading="#FFA47F",
        text_hover="#0A1E2B",
        text_alt="#A8C4D2",
        border_main="#4380A0",
        border_alt="#1F4860",
        bg_action="#EC5F34",
        text_on_action="#160800",
        bg_action_hover="#FF7E52",
        bg_error="#E0596E",
        text_on_error="#1A0409",
        bg_success="#2C9C7A",
        text_on_success="#04180F",
        font_family="'Avenir Next', Avenir, 'Century Gothic', 'Trebuchet MS', sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "$randomized": (
                "nx=0:169 ny=0:169 ns=150:200 "
                "hx=10:72 hy=44:80 hs=150:215 hd=0:1 "
                "mx=28:90 my=8:36 ms=92:132 md=0:1 mp=0:3 "
                "fx=4:56 fy=20:52 fs=52:78 fp=0:3 "
                "bx=56:94 by=0:99 bw=46:74 "
                "sx=36:64 sy=52:74"
            ),
            "$randomized@minimal": "hd=0:0",
            "$assemble": "--dgcv-special-background",
            "$part_1": (
                'url("data:image/svg+xml,'
                "%3Csvg xmlns='http://www.w3.org/2000/svg' width='170' height='170' viewBox='0 0 170 170'"
                "%3E %3Cfilter id='s' x='0' y='0' width='100%' height='100%' color-interpolation-filters="
                "'sRGB'%3E %3CfeTurbulence type='fractalNoise' baseFrequency='0.62' numOctaves='2' seed='"
                "4' stitchTiles='stitch' result='n'/%3E %3CfeColorMatrix in='n' type='matrix' values='0 0"
                " 0 0 0.80 0 0 0 0 0.94 0 0 0 0 1 1.15 0 0 0 -0.80'/%3E %3C/filter%3E %3Crect width='170'"
                " height='170' filter='url%28%23s%29'/%3E %3C/svg%3E"
                '") ${nx}px ${ny}px / ${ns}px ${ns}px repeat, '
                'url("data:image/svg+xml,'
                "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='-66 -48 132 96'%3E%3Cdefs%3E%3Clinear"
                "Gradient id='bodyfill' gradientUnits='userSpaceOnUse' x1='0' y1='-28' x2='0' y2='28'%3E "
                "%3Cstop offset='0' stop-color='%23FF9752'/%3E %3Cstop offset='0.45' stop-color='%23F26A2"
                "E'/%3E %3Cstop offset='1' stop-color='%23D2461A'/%3E %3C/linearGradient%3E %3ClinearGrad"
                "ient id='finfill' gradientUnits='userSpaceOnUse' x1='0' y1='-34' x2='0' y2='34'%3E %3Cst"
                "op offset='0' stop-color='%23F9853F'/%3E %3Cstop offset='1' stop-color='%23D8531F'/%3E %"
                "3C/linearGradient%3E %3CclipPath id='bodyclip'%3E %3Cpath d='M 52,1 C 40,-18 16,-28 -4,-"
                "27 C -16,-26 -25,-20 -30,-10 L -30,10 C -25,20 -16,26 -4,27 C 16,28 40,19 52,1 Z'/%3E %3"
                "C/clipPath%3E%3C/defs%3E%3Cg transform='rotate%28-7%29' opacity='1.00'%3E%3Cpath d='M -3"
                "0,-10 C -40,-16 -47,-20 -53,-22 C -56,-11 -56,11 -53,22 C -47,20 -40,16 -30,10 Z' fill='"
                "url%28%23finfill%29' stroke='%23152430' stroke-width='2'/%3E %3Cpath d='M 24,-24 C 14,-3"
                "4 -2,-38 -12,-36 C -20,-34 -25,-25 -27,-16 Z' fill='url%28%23finfill%29' stroke='%231524"
                "30' stroke-width='2'/%3E %3Cpath d='M 8,24 C 2,33 -8,35 -14,33 C -20,31 -24,23 -26,17 Z'"
                " fill='url%28%23finfill%29' stroke='%23152430' stroke-width='2'/%3E %3Cpath d='M 52,1 C "
                "40,-18 16,-28 -4,-27 C -16,-26 -25,-20 -30,-10 L -30,10 C -25,20 -16,26 -4,27 C 16,28 40"
                ",19 52,1 Z' fill='url%28%23bodyfill%29'/%3E %3Cg clip-path='url%28%23bodyclip%29'%3E %3C"
                "path d='M 30,-32 C 27,-14 27,10 24,34' fill='none' stroke='%23152430' stroke-width='21'/"
                "%3E %3Cpath d='M 8,-34 C 2,-16 12,8 4,34' fill='none' stroke='%23152430' stroke-width='2"
                "2'/%3E %3Cpath d='M -22,-32 C -25,-12 -25,12 -22,34' fill='none' stroke='%23152430' stro"
                "ke-width='14'/%3E %3Cpath d='M 30,-32 C 27,-14 27,10 24,34' fill='none' stroke='%23FBF1E"
                "6' stroke-width='13'/%3E %3Cpath d='M 8,-34 C 2,-16 12,8 4,34' fill='none' stroke='%23FB"
                "F1E6' stroke-width='14'/%3E %3Cpath d='M -22,-32 C -25,-12 -25,12 -22,34' fill='none' st"
                "roke='%23FBF1E6' stroke-width='8'/%3E %3Cellipse cx='6' cy='23' rx='30' ry='10' fill='%2"
                "38E2F12' opacity='0.30'/%3E %3Cellipse cx='2' cy='-24' rx='28' ry='8' fill='%23FFC08A' o"
                "pacity='0.22'/%3E %3C/g%3E %3Cpath d='M 52,1 C 40,-18 16,-28 -4,-27 C -16,-26 -25,-20 -3"
                "0,-10 L -30,10 C -25,20 -16,26 -4,27 C 16,28 40,19 52,1 Z' fill='none' stroke='%23152430"
                "' stroke-width='2.5'/%3E %3Cpath d='M 18,6 C 10,16 8,24 12,26 C 18,26 24,16 24,8 Z' fill"
                "='%23F9873F' stroke='%23152430' stroke-width='1.6' opacity='0.9'/%3E %3Ccircle cx='37' c"
                "y='-7' r='5.4' fill='%23F7EADF' stroke='%23152430' stroke-width='1.4'/%3E %3Ccircle cx='"
                "37' cy='-7' r='3' fill='%23131F2A'/%3E %3Ccircle cx='35.4' cy='-8.6' r='1.1' fill='%23FF"
                "FFFF'/%3E%3C/g%3E%3C/svg%3E"
                '") calc(${hx}% + ${hd} * 9999px) ${hy}% / min(${hs}px, 46%) auto no-repeat'
            ),
            "$part_2@detail": (
                'url("data:image/svg+xml,'
                "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='-66 -48 132 96'%3E%3Cdefs%3E%3Clinear"
                "Gradient id='bodyfill' gradientUnits='userSpaceOnUse' x1='0' y1='-28' x2='0' y2='28'%3E "
                "%3Cstop offset='0' stop-color='%23FF9752'/%3E %3Cstop offset='0.45' stop-color='%23F26A2"
                "E'/%3E %3Cstop offset='1' stop-color='%23D2461A'/%3E %3C/linearGradient%3E %3ClinearGrad"
                "ient id='finfill' gradientUnits='userSpaceOnUse' x1='0' y1='-34' x2='0' y2='34'%3E %3Cst"
                "op offset='0' stop-color='%23F9853F'/%3E %3Cstop offset='1' stop-color='%23D8531F'/%3E %"
                "3C/linearGradient%3E %3CclipPath id='bodyclip'%3E %3Cpath d='M 52,1 C 40,-18 16,-28 -4,-"
                "27 C -16,-26 -25,-20 -30,-10 L -30,10 C -25,20 -16,26 -4,27 C 16,28 40,19 52,1 Z'/%3E %3"
                "C/clipPath%3E%3C/defs%3E%3Cg transform='scale%28-1 1%29 rotate%287%29' opacity='1.00'%3E"
                "%3Cpath d='M -30,-10 C -40,-16 -47,-20 -53,-22 C -56,-11 -56,11 -53,22 C -47,20 -40,16 -"
                "30,10 Z' fill='url%28%23finfill%29' stroke='%23152430' stroke-width='2'/%3E %3Cpath d='M"
                " 24,-24 C 14,-34 -2,-38 -12,-36 C -20,-34 -25,-25 -27,-16 Z' fill='url%28%23finfill%29' "
                "stroke='%23152430' stroke-width='2'/%3E %3Cpath d='M 8,24 C 2,33 -8,35 -14,33 C -20,31 -"
                "24,23 -26,17 Z' fill='url%28%23finfill%29' stroke='%23152430' stroke-width='2'/%3E %3Cpa"
                "th d='M 52,1 C 40,-18 16,-28 -4,-27 C -16,-26 -25,-20 -30,-10 L -30,10 C -25,20 -16,26 -"
                "4,27 C 16,28 40,19 52,1 Z' fill='url%28%23bodyfill%29'/%3E %3Cg clip-path='url%28%23body"
                "clip%29'%3E %3Cpath d='M 30,-32 C 27,-14 27,10 24,34' fill='none' stroke='%23152430' str"
                "oke-width='21'/%3E %3Cpath d='M 8,-34 C 2,-16 12,8 4,34' fill='none' stroke='%23152430' "
                "stroke-width='22'/%3E %3Cpath d='M -22,-32 C -25,-12 -25,12 -22,34' fill='none' stroke='"
                "%23152430' stroke-width='14'/%3E %3Cpath d='M 30,-32 C 27,-14 27,10 24,34' fill='none' s"
                "troke='%23FBF1E6' stroke-width='13'/%3E %3Cpath d='M 8,-34 C 2,-16 12,8 4,34' fill='none"
                "' stroke='%23FBF1E6' stroke-width='14'/%3E %3Cpath d='M -22,-32 C -25,-12 -25,12 -22,34'"
                " fill='none' stroke='%23FBF1E6' stroke-width='8'/%3E %3Cellipse cx='6' cy='23' rx='30' r"
                "y='10' fill='%238E2F12' opacity='0.30'/%3E %3Cellipse cx='2' cy='-24' rx='28' ry='8' fil"
                "l='%23FFC08A' opacity='0.22'/%3E %3C/g%3E %3Cpath d='M 52,1 C 40,-18 16,-28 -4,-27 C -16"
                ",-26 -25,-20 -30,-10 L -30,10 C -25,20 -16,26 -4,27 C 16,28 40,19 52,1 Z' fill='none' st"
                "roke='%23152430' stroke-width='2.5'/%3E %3Cpath d='M 18,6 C 10,16 8,24 12,26 C 18,26 24,"
                "16 24,8 Z' fill='%23F9873F' stroke='%23152430' stroke-width='1.6' opacity='0.9'/%3E %3Cc"
                "ircle cx='37' cy='-7' r='5.4' fill='%23F7EADF' stroke='%23152430' stroke-width='1.4'/%3E"
                " %3Ccircle cx='37' cy='-7' r='3' fill='%23131F2A'/%3E %3Ccircle cx='35.4' cy='-8.6' r='1"
                ".1' fill='%23FFFFFF'/%3E%3C/g%3E%3C/svg%3E"
                '") calc(${hx}% + (1 - ${hd}) * 9999px) ${hy}% / min(${hs}px, 46%) auto no-repeat, '
                'url("data:image/svg+xml,'
                "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='-66 -48 132 96'%3E%3Cdefs%3E%3Clinear"
                "Gradient id='bodyfill' gradientUnits='userSpaceOnUse' x1='0' y1='-28' x2='0' y2='28'%3E "
                "%3Cstop offset='0' stop-color='%23FF9752'/%3E %3Cstop offset='0.45' stop-color='%23F26A2"
                "E'/%3E %3Cstop offset='1' stop-color='%23D2461A'/%3E %3C/linearGradient%3E %3ClinearGrad"
                "ient id='finfill' gradientUnits='userSpaceOnUse' x1='0' y1='-34' x2='0' y2='34'%3E %3Cst"
                "op offset='0' stop-color='%23F9853F'/%3E %3Cstop offset='1' stop-color='%23D8531F'/%3E %"
                "3C/linearGradient%3E %3CclipPath id='bodyclip'%3E %3Cpath d='M 52,1 C 40,-18 16,-28 -4,-"
                "27 C -16,-26 -25,-20 -30,-10 L -30,10 C -25,20 -16,26 -4,27 C 16,28 40,19 52,1 Z'/%3E %3"
                "C/clipPath%3E%3C/defs%3E%3Cg transform='rotate%28-5%29' opacity='0.92'%3E%3Cpath d='M -3"
                "0,-10 C -40,-16 -47,-20 -53,-22 C -56,-11 -56,11 -53,22 C -47,20 -40,16 -30,10 Z' fill='"
                "url%28%23finfill%29' stroke='%23152430' stroke-width='2'/%3E %3Cpath d='M 24,-24 C 14,-3"
                "4 -2,-38 -12,-36 C -20,-34 -25,-25 -27,-16 Z' fill='url%28%23finfill%29' stroke='%231524"
                "30' stroke-width='2'/%3E %3Cpath d='M 8,24 C 2,33 -8,35 -14,33 C -20,31 -24,23 -26,17 Z'"
                " fill='url%28%23finfill%29' stroke='%23152430' stroke-width='2'/%3E %3Cpath d='M 52,1 C "
                "40,-18 16,-28 -4,-27 C -16,-26 -25,-20 -30,-10 L -30,10 C -25,20 -16,26 -4,27 C 16,28 40"
                ",19 52,1 Z' fill='url%28%23bodyfill%29'/%3E %3Cg clip-path='url%28%23bodyclip%29'%3E %3C"
                "path d='M 30,-32 C 27,-14 27,10 24,34' fill='none' stroke='%23152430' stroke-width='21'/"
                "%3E %3Cpath d='M 8,-34 C 2,-16 12,8 4,34' fill='none' stroke='%23152430' stroke-width='2"
                "2'/%3E %3Cpath d='M -22,-32 C -25,-12 -25,12 -22,34' fill='none' stroke='%23152430' stro"
                "ke-width='14'/%3E %3Cpath d='M 30,-32 C 27,-14 27,10 24,34' fill='none' stroke='%23FBF1E"
                "6' stroke-width='13'/%3E %3Cpath d='M 8,-34 C 2,-16 12,8 4,34' fill='none' stroke='%23FB"
                "F1E6' stroke-width='14'/%3E %3Cpath d='M -22,-32 C -25,-12 -25,12 -22,34' fill='none' st"
                "roke='%23FBF1E6' stroke-width='8'/%3E %3Cellipse cx='6' cy='23' rx='30' ry='10' fill='%2"
                "38E2F12' opacity='0.30'/%3E %3Cellipse cx='2' cy='-24' rx='28' ry='8' fill='%23FFC08A' o"
                "pacity='0.22'/%3E %3C/g%3E %3Cpath d='M 52,1 C 40,-18 16,-28 -4,-27 C -16,-26 -25,-20 -3"
                "0,-10 L -30,10 C -25,20 -16,26 -4,27 C 16,28 40,19 52,1 Z' fill='none' stroke='%23152430"
                "' stroke-width='2.5'/%3E %3Cpath d='M 18,6 C 10,16 8,24 12,26 C 18,26 24,16 24,8 Z' fill"
                "='%23F9873F' stroke='%23152430' stroke-width='1.6' opacity='0.9'/%3E %3Ccircle cx='37' c"
                "y='-7' r='5.4' fill='%23F7EADF' stroke='%23152430' stroke-width='1.4'/%3E %3Ccircle cx='"
                "37' cy='-7' r='3' fill='%23131F2A'/%3E %3Ccircle cx='35.4' cy='-8.6' r='1.1' fill='%23FF"
                "FFFF'/%3E%3C/g%3E%3Crect x='-66' y='-48' width='132' height='96' fill='%230F3E57' opacit"
                "y='0.10'/%3E%3C/svg%3E"
                '") calc(${mx}% + (${md} + max(0, ${mp} - 2)) * 9999px) ${my}% / min(${ms}px, 34%) auto no-repeat, '
                'url("data:image/svg+xml,'
                "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='-66 -48 132 96'%3E%3Cdefs%3E%3Clinear"
                "Gradient id='bodyfill' gradientUnits='userSpaceOnUse' x1='0' y1='-28' x2='0' y2='28'%3E "
                "%3Cstop offset='0' stop-color='%23FF9752'/%3E %3Cstop offset='0.45' stop-color='%23F26A2"
                "E'/%3E %3Cstop offset='1' stop-color='%23D2461A'/%3E %3C/linearGradient%3E %3ClinearGrad"
                "ient id='finfill' gradientUnits='userSpaceOnUse' x1='0' y1='-34' x2='0' y2='34'%3E %3Cst"
                "op offset='0' stop-color='%23F9853F'/%3E %3Cstop offset='1' stop-color='%23D8531F'/%3E %"
                "3C/linearGradient%3E %3CclipPath id='bodyclip'%3E %3Cpath d='M 52,1 C 40,-18 16,-28 -4,-"
                "27 C -16,-26 -25,-20 -30,-10 L -30,10 C -25,20 -16,26 -4,27 C 16,28 40,19 52,1 Z'/%3E %3"
                "C/clipPath%3E%3C/defs%3E%3Cg transform='scale%28-1 1%29 rotate%285%29' opacity='0.92'%3E"
                "%3Cpath d='M -30,-10 C -40,-16 -47,-20 -53,-22 C -56,-11 -56,11 -53,22 C -47,20 -40,16 -"
                "30,10 Z' fill='url%28%23finfill%29' stroke='%23152430' stroke-width='2'/%3E %3Cpath d='M"
                " 24,-24 C 14,-34 -2,-38 -12,-36 C -20,-34 -25,-25 -27,-16 Z' fill='url%28%23finfill%29' "
                "stroke='%23152430' stroke-width='2'/%3E %3Cpath d='M 8,24 C 2,33 -8,35 -14,33 C -20,31 -"
                "24,23 -26,17 Z' fill='url%28%23finfill%29' stroke='%23152430' stroke-width='2'/%3E %3Cpa"
                "th d='M 52,1 C 40,-18 16,-28 -4,-27 C -16,-26 -25,-20 -30,-10 L -30,10 C -25,20 -16,26 -"
                "4,27 C 16,28 40,19 52,1 Z' fill='url%28%23bodyfill%29'/%3E %3Cg clip-path='url%28%23body"
                "clip%29'%3E %3Cpath d='M 30,-32 C 27,-14 27,10 24,34' fill='none' stroke='%23152430' str"
                "oke-width='21'/%3E %3Cpath d='M 8,-34 C 2,-16 12,8 4,34' fill='none' stroke='%23152430' "
                "stroke-width='22'/%3E %3Cpath d='M -22,-32 C -25,-12 -25,12 -22,34' fill='none' stroke='"
                "%23152430' stroke-width='14'/%3E %3Cpath d='M 30,-32 C 27,-14 27,10 24,34' fill='none' s"
                "troke='%23FBF1E6' stroke-width='13'/%3E %3Cpath d='M 8,-34 C 2,-16 12,8 4,34' fill='none"
                "' stroke='%23FBF1E6' stroke-width='14'/%3E %3Cpath d='M -22,-32 C -25,-12 -25,12 -22,34'"
                " fill='none' stroke='%23FBF1E6' stroke-width='8'/%3E %3Cellipse cx='6' cy='23' rx='30' r"
                "y='10' fill='%238E2F12' opacity='0.30'/%3E %3Cellipse cx='2' cy='-24' rx='28' ry='8' fil"
                "l='%23FFC08A' opacity='0.22'/%3E %3C/g%3E %3Cpath d='M 52,1 C 40,-18 16,-28 -4,-27 C -16"
                ",-26 -25,-20 -30,-10 L -30,10 C -25,20 -16,26 -4,27 C 16,28 40,19 52,1 Z' fill='none' st"
                "roke='%23152430' stroke-width='2.5'/%3E %3Cpath d='M 18,6 C 10,16 8,24 12,26 C 18,26 24,"
                "16 24,8 Z' fill='%23F9873F' stroke='%23152430' stroke-width='1.6' opacity='0.9'/%3E %3Cc"
                "ircle cx='37' cy='-7' r='5.4' fill='%23F7EADF' stroke='%23152430' stroke-width='1.4'/%3E"
                " %3Ccircle cx='37' cy='-7' r='3' fill='%23131F2A'/%3E %3Ccircle cx='35.4' cy='-8.6' r='1"
                ".1' fill='%23FFFFFF'/%3E%3C/g%3E%3Crect x='-66' y='-48' width='132' height='96' fill='%2"
                "30F3E57' opacity='0.10'/%3E%3C/svg%3E"
                '") calc(${mx}% + ((1 - ${md}) + max(0, ${mp} - 2)) * 9999px) ${my}% / min(${ms}px, 34%) auto no-repeat, '
                'url("data:image/svg+xml,'
                "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='-66 -48 132 96'%3E%3Cdefs%3E%3Clinear"
                "Gradient id='bodyfill' gradientUnits='userSpaceOnUse' x1='0' y1='-28' x2='0' y2='28'%3E "
                "%3Cstop offset='0' stop-color='%23FF9752'/%3E %3Cstop offset='0.45' stop-color='%23F26A2"
                "E'/%3E %3Cstop offset='1' stop-color='%23D2461A'/%3E %3C/linearGradient%3E %3ClinearGrad"
                "ient id='finfill' gradientUnits='userSpaceOnUse' x1='0' y1='-34' x2='0' y2='34'%3E %3Cst"
                "op offset='0' stop-color='%23F9853F'/%3E %3Cstop offset='1' stop-color='%23D8531F'/%3E %"
                "3C/linearGradient%3E %3CclipPath id='bodyclip'%3E %3Cpath d='M 52,1 C 40,-18 16,-28 -4,-"
                "27 C -16,-26 -25,-20 -30,-10 L -30,10 C -25,20 -16,26 -4,27 C 16,28 40,19 52,1 Z'/%3E %3"
                "C/clipPath%3E%3C/defs%3E%3Cg transform='rotate%286%29' opacity='0.55'%3E%3Cpath d='M -30"
                ",-10 C -40,-16 -47,-20 -53,-22 C -56,-11 -56,11 -53,22 C -47,20 -40,16 -30,10 Z' fill='u"
                "rl%28%23finfill%29' stroke='%23152430' stroke-width='2'/%3E %3Cpath d='M 24,-24 C 14,-34"
                " -2,-38 -12,-36 C -20,-34 -25,-25 -27,-16 Z' fill='url%28%23finfill%29' stroke='%2315243"
                "0' stroke-width='2'/%3E %3Cpath d='M 8,24 C 2,33 -8,35 -14,33 C -20,31 -24,23 -26,17 Z' "
                "fill='url%28%23finfill%29' stroke='%23152430' stroke-width='2'/%3E %3Cpath d='M 52,1 C 4"
                "0,-18 16,-28 -4,-27 C -16,-26 -25,-20 -30,-10 L -30,10 C -25,20 -16,26 -4,27 C 16,28 40,"
                "19 52,1 Z' fill='url%28%23bodyfill%29'/%3E %3Cg clip-path='url%28%23bodyclip%29'%3E %3Cp"
                "ath d='M 30,-32 C 27,-14 27,10 24,34' fill='none' stroke='%23152430' stroke-width='21'/%"
                "3E %3Cpath d='M 8,-34 C 2,-16 12,8 4,34' fill='none' stroke='%23152430' stroke-width='22"
                "'/%3E %3Cpath d='M -22,-32 C -25,-12 -25,12 -22,34' fill='none' stroke='%23152430' strok"
                "e-width='14'/%3E %3Cpath d='M 30,-32 C 27,-14 27,10 24,34' fill='none' stroke='%23FBF1E6"
                "' stroke-width='13'/%3E %3Cpath d='M 8,-34 C 2,-16 12,8 4,34' fill='none' stroke='%23FBF"
                "1E6' stroke-width='14'/%3E %3Cpath d='M -22,-32 C -25,-12 -25,12 -22,34' fill='none' str"
                "oke='%23FBF1E6' stroke-width='8'/%3E %3Cellipse cx='6' cy='23' rx='30' ry='10' fill='%23"
                "8E2F12' opacity='0.30'/%3E %3Cellipse cx='2' cy='-24' rx='28' ry='8' fill='%23FFC08A' op"
                "acity='0.22'/%3E %3C/g%3E %3Cpath d='M 52,1 C 40,-18 16,-28 -4,-27 C -16,-26 -25,-20 -30"
                ",-10 L -30,10 C -25,20 -16,26 -4,27 C 16,28 40,19 52,1 Z' fill='none' stroke='%23152430'"
                " stroke-width='2.5'/%3E %3Cpath d='M 18,6 C 10,16 8,24 12,26 C 18,26 24,16 24,8 Z' fill="
                "'%23F9873F' stroke='%23152430' stroke-width='1.6' opacity='0.9'/%3E %3Ccircle cx='37' cy"
                "='-7' r='5.4' fill='%23F7EADF' stroke='%23152430' stroke-width='1.4'/%3E %3Ccircle cx='3"
                "7' cy='-7' r='3' fill='%23131F2A'/%3E %3Ccircle cx='35.4' cy='-8.6' r='1.1' fill='%23FFF"
                "FFF'/%3E%3C/g%3E%3Crect x='-66' y='-48' width='132' height='96' fill='%230F3E57' opacity"
                "='0.28'/%3E%3C/svg%3E"
                '") calc(${fx}% + max(0, ${fp} - 2) * 9999px) ${fy}% / min(${fs}px, 22%) auto no-repeat'
            ),
            "$part_3": (
                'url("data:image/svg+xml,'
                "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 60 260'%3E%3Ccircle cx='30' cy='2"
                "2' r='2.4' fill='none' stroke='%23CFEFF5' stroke-width='1.1' opacity='0.44'/%3E%3Ccircle"
                " cx='29.2' cy='21.1' r='0.6' fill='%23EAFBFF' opacity='0.48'/%3E%3Ccircle cx='46' cy='58"
                "' r='4.9' fill='none' stroke='%23CFEFF5' stroke-width='1.1' opacity='0.42'/%3E%3Ccircle "
                "cx='43.9' cy='56.2' r='1.2' fill='%23EAFBFF' opacity='0.46'/%3E%3Ccircle cx='37' cy='94'"
                " r='3.9' fill='none' stroke='%23CFEFF5' stroke-width='1.1' opacity='0.40'/%3E%3Ccircle c"
                "x='35.5' cy='92.6' r='0.9' fill='%23EAFBFF' opacity='0.44'/%3E%3Ccircle cx='17' cy='130'"
                " r='2.9' fill='none' stroke='%23CFEFF5' stroke-width='1.1' opacity='0.38'/%3E%3Ccircle c"
                "x='16.4' cy='129.0' r='0.7' fill='%23EAFBFF' opacity='0.42'/%3E%3Ccircle cx='18' cy='166"
                "' r='5.4' fill='none' stroke='%23CFEFF5' stroke-width='1.1' opacity='0.36'/%3E%3Ccircle "
                "cx='15.8' cy='164.1' r='1.3' fill='%23EAFBFF' opacity='0.40'/%3E%3Ccircle cx='37' cy='20"
                "2' r='4.4' fill='none' stroke='%23CFEFF5' stroke-width='1.1' opacity='0.34'/%3E%3Ccircle"
                " cx='35.7' cy='200.4' r='1.1' fill='%23EAFBFF' opacity='0.38'/%3E%3Ccircle cx='46' cy='2"
                "38' r='3.4' fill='none' stroke='%23CFEFF5' stroke-width='1.1' opacity='0.32'/%3E%3Ccircl"
                "e cx='44.4' cy='236.8' r='0.8' fill='%23EAFBFF' opacity='0.36'/%3E%3C/svg%3E"
                '") ${bx}% ${by}% / ${bw}px auto repeat-y, '
                'url("data:image/svg+xml,'
                "%3Csvg xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' vie"
                "wBox='0 0 800 500' preserveAspectRatio='xMidYMax slice'%3E %3Cdefs%3E %3ClinearGradient "
                "id='water' x1='0' y1='0' x2='0' y2='1'%3E %3Cstop offset='0' stop-color='%231E6C8E'/%3E "
                "%3Cstop offset='0.24' stop-color='%23175978'/%3E %3Cstop offset='0.52' stop-color='%230F"
                "3E57'/%3E %3Cstop offset='0.78' stop-color='%230A2C3F'/%3E %3Cstop offset='1' stop-color"
                "='%23071D2A'/%3E %3C/linearGradient%3E %3ClinearGradient id='ray' gradientUnits='userSpa"
                "ceOnUse' x1='0' y1='-60' x2='0' y2='430'%3E %3Cstop offset='0' stop-color='%23CFEFF7' st"
                "op-opacity='0.22'/%3E %3Cstop offset='0.45' stop-color='%23AFE0EE' stop-opacity='0.08'/%"
                "3E %3Cstop offset='1' stop-color='%238FD0E6' stop-opacity='0'/%3E %3C/linearGradient%3E "
                "%3ClinearGradient id='tentfill' gradientUnits='userSpaceOnUse' x1='0' y1='0' x2='0' y2='"
                "-33'%3E %3Cstop offset='0' stop-color='%236B4560'/%3E %3Cstop offset='0.55' stop-color='"
                "%239C6580'/%3E %3Cstop offset='1' stop-color='%23D693A8'/%3E %3C/linearGradient%3E %3Cli"
                "nearGradient id='floor' gradientUnits='userSpaceOnUse' x1='0' y1='432' x2='0' y2='505'%3"
                "E %3Cstop offset='0' stop-color='%230B2836'/%3E %3Cstop offset='1' stop-color='%2305141D"
                "'/%3E %3C/linearGradient%3E %3Cfilter id='caustic' x='0' y='0' width='100%' height='100%"
                "' color-interpolation-filters='sRGB'%3E %3CfeTurbulence type='fractalNoise' baseFrequenc"
                "y='0.016 0.05' numOctaves='3' seed='9' stitchTiles='stitch' result='n'/%3E %3CfeColorMat"
                "rix in='n' type='matrix' values='0 0 0 0 0.82 0 0 0 0 0.95 0 0 0 0 1 1.5 0 0 0 -1.02' re"
                "sult='m'/%3E %3CfeComposite in='m' in2='SourceGraphic' operator='in'/%3E %3C/filter%3E %"
                "3ClinearGradient id='cfade' x1='0' y1='0' x2='0' y2='1'%3E %3Cstop offset='0' stop-color"
                "='%23FFFFFF' stop-opacity='1'/%3E %3Cstop offset='1' stop-color='%23000000' stop-opacity"
                "='0'/%3E %3C/linearGradient%3E %3Cmask id='cmask'%3E%3Crect x='0' y='0' width='800' heig"
                "ht='280' fill='url%28%23cfade%29'/%3E%3C/mask%3E %3Cg id='tent'%3E %3Cpath d='M 0,0 C -3"
                ".4,-13 -3.4,-25 0,-33 C 3.4,-25 3.4,-13 0,0 Z' fill='url%28%23tentfill%29'/%3E %3Ccircle"
                " cx='0' cy='-32' r='2.6' fill='%23F0A9BC' opacity='0.85'/%3E %3C/g%3E %3C/defs%3E %3Crec"
                "t x='0' y='0' width='800' height='500' fill='url%28%23water%29'/%3E %3Cg transform='skew"
                "X%28-13%29'%3E %3Crect x='150' y='-60' width='58' height='500' fill='url%28%23ray%29'/%3"
                "E %3Crect x='300' y='-60' width='26' height='500' fill='url%28%23ray%29'/%3E %3Crect x='"
                "430' y='-60' width='72' height='500' fill='url%28%23ray%29'/%3E %3Crect x='620' y='-60' "
                "width='34' height='500' fill='url%28%23ray%29'/%3E %3C/g%3E %3Crect x='0' y='0' width='8"
                "00' height='280' fill='%23FFFFFF' filter='url%28%23caustic%29' mask='url%28%23cmask%29' "
                "opacity='0.5'/%3E %3Cpath d='M -4,470 C 90,452 150,462 230,452 C 320,441 380,458 470,450"
                " C 560,442 660,456 804,442 L 804,504 L -4,504 Z' fill='url%28%23floor%29'/%3E %3Cpath d="
                "'M -4,470 C 90,452 150,462 230,452 C 320,441 380,458 470,450 C 560,442 660,456 804,442' "
                "fill='none' stroke='%232C7C93' stroke-width='2' opacity='0.30'/%3E %3Cg transform='trans"
                "late%28636 468%29 scale%281.05%29'%3E%3Cellipse cx='0' cy='4' rx='48' ry='15' fill='%235"
                "53650'/%3E%3Cuse xlink:href='%23tent' transform='translate%28-42.0 0%29 rotate%28-25.5%2"
                "9 scale%281.02 0.59%29'/%3E%3Cuse xlink:href='%23tent' transform='translate%28-36.0 0%29"
                " rotate%28-28.4%29 scale%280.86 0.71%29'/%3E%3Cuse xlink:href='%23tent' transform='trans"
                "late%28-30.0 0%29 rotate%28-25.3%29 scale%280.94 0.83%29'/%3E%3Cuse xlink:href='%23tent'"
                " transform='translate%28-24.0 0%29 rotate%28-13.5%29 scale%281.02 0.94%29'/%3E%3Cuse xli"
                "nk:href='%23tent' transform='translate%28-18.0 0%29 rotate%28-9.9%29 scale%280.86 1.01%2"
                "9'/%3E%3Cuse xlink:href='%23tent' transform='translate%28-12.0 0%29 rotate%28-13.0%29 sc"
                "ale%280.94 1.04%29'/%3E%3Cuse xlink:href='%23tent' transform='translate%28-6.0 0%29 rota"
                "te%28-6.1%29 scale%281.02 1.01%29'/%3E%3Cuse xlink:href='%23tent' transform='translate%2"
                "80.0 0%29 rotate%284.9%29 scale%280.86 0.94%29'/%3E%3Cuse xlink:href='%23tent' transform"
                "='translate%286.0 0%29 rotate%284.8%29 scale%280.94 0.83%29'/%3E%3Cuse xlink:href='%23te"
                "nt' transform='translate%2812.0 0%29 rotate%283.6%29 scale%281.02 0.71%29'/%3E%3Cuse xli"
                "nk:href='%23tent' transform='translate%2818.0 0%29 rotate%2813.6%29 scale%280.86 0.59%29"
                "'/%3E%3Cuse xlink:href='%23tent' transform='translate%2824.0 0%29 rotate%2821.9%29 scale"
                "%280.94 0.50%29'/%3E%3Cuse xlink:href='%23tent' transform='translate%2830.0 0%29 rotate%"
                "2819.4%29 scale%281.02 0.45%29'/%3E%3Cuse xlink:href='%23tent' transform='translate%2836"
                ".0 0%29 rotate%2821.4%29 scale%280.86 0.45%29'/%3E%3Cuse xlink:href='%23tent' transform="
                "'translate%2842.0 0%29 rotate%2833.1%29 scale%280.94 0.50%29'/%3E%3C/g%3E %3Cg transform"
                "='translate%28268 462%29 scale%281.45%29'%3E%3Cellipse cx='0' cy='4' rx='48' ry='15' fil"
                "l='%23553650'/%3E%3Cuse xlink:href='%23tent' transform='translate%28-42.0 0%29 rotate%28"
                "-30.0%29 scale%280.86 0.45%29'/%3E%3Cuse xlink:href='%23tent' transform='translate%28-36"
                ".0 0%29 rotate%28-20.8%29 scale%280.94 0.50%29'/%3E%3Cuse xlink:href='%23tent' transform"
                "='translate%28-30.0 0%29 rotate%28-22.7%29 scale%281.02 0.59%29'/%3E%3Cuse xlink:href='%"
                "23tent' transform='translate%28-24.0 0%29 rotate%28-21.8%29 scale%280.86 0.71%29'/%3E%3C"
                "use xlink:href='%23tent' transform='translate%28-18.0 0%29 rotate%28-10.4%29 scale%280.9"
                "4 0.83%29'/%3E%3Cuse xlink:href='%23tent' transform='translate%28-12.0 0%29 rotate%28-4."
                "6%29 scale%281.02 0.94%29'/%3E%3Cuse xlink:href='%23tent' transform='translate%28-6.0 0%"
                "29 rotate%28-7.8%29 scale%280.86 1.01%29'/%3E%3Cuse xlink:href='%23tent' transform='tran"
                "slate%280.0 0%29 rotate%28-3.1%29 scale%280.94 1.04%29'/%3E%3Cuse xlink:href='%23tent' t"
                "ransform='translate%286.0 0%29 rotate%288.6%29 scale%281.02 1.01%29'/%3E%3Cuse xlink:hre"
                "f='%23tent' transform='translate%2812.0 0%29 rotate%2810.6%29 scale%280.86 0.94%29'/%3E%"
                "3Cuse xlink:href='%23tent' transform='translate%2818.0 0%29 rotate%288.1%29 scale%280.94"
                " 0.83%29'/%3E%3Cuse xlink:href='%23tent' transform='translate%2824.0 0%29 rotate%2816.4%"
                "29 scale%281.02 0.71%29'/%3E%3Cuse xlink:href='%23tent' transform='translate%2830.0 0%29"
                " rotate%2826.4%29 scale%280.86 0.59%29'/%3E%3Cuse xlink:href='%23tent' transform='transl"
                "ate%2836.0 0%29 rotate%2825.2%29 scale%280.94 0.50%29'/%3E%3Cuse xlink:href='%23tent' tr"
                "ansform='translate%2842.0 0%29 rotate%2825.1%29 scale%281.02 0.45%29'/%3E%3C/g%3E %3Crec"
                "t x='0' y='0' width='800' height='500' fill='%230A2C3F' opacity='0.10'/%3E %3C/svg%3E"
                '") ${sx}% ${sy}% / cover no-repeat, '
                "linear-gradient(to bottom, #1E6C8E 0%, #175978 24%, #0F3E57 52%, "
                "#0A2C3F 78%, #071D2A 100%) center / cover no-repeat, "
                "#0F3E57"
            ),
            "--dgcv-special-text": "#FFD2BC",
            "--plaque-fill": "#0E2637",
            "--plaque-border": "#FFA47F",
            "--dgcv-table-shadow": "0 2px 14px rgba(2, 12, 20, 0.55)",
            "--dgcv-text-shadow": "none",
            "--dgcv-hover-transform": "translateY(-1px)",
            "--dgcv-hover-transition": "background-color 140ms ease, color 140ms ease, transform 140ms ease",
        },
    ),
    "koifish_pond": ThemeConfig(
        bg_primary="#0E1F1C",
        bg_surface="#162C27",
        bg_alt="#122622",
        bg_hover="#E8A93C",
        text_main="#DCE8DF",
        text_heading="#F0B75A",
        text_hover="#0E1F1C",
        text_alt="#B4C9BA",
        border_main="#63947B",
        border_alt="#284840",
        bg_action="#E2652A",
        text_on_action="#1A0703",
        bg_action_hover="#F0834A",
        bg_error="#DE5A46",
        text_on_error="#1A0604",
        bg_success="#63A85C",
        text_on_success="#04140A",
        font_family="'Iowan Old Style', 'Palatino Linotype', Palatino, Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "$randomized": (
                "rx=0:179 ry=0:179 rs=150:210 "
                "g1x=14:84 g1y=12:84 g1s=104:168 "
                "g2x=12:86 g2y=14:82 g2s=76:126 g2p=0:3 "
                "k1x=16:82 k1y=14:80 k1s=96:150 "
                "k2x=12:86 k2y=12:84 k2s=84:132 k2p=0:3 "
                "k3x=14:84 k3y=16:82 k3s=72:114 k3p=0:1 "
                "k4x=14:84 k4y=16:82 k4s=64:100 "
                "sx=40:60 sy=40:60"
            ),
            "$assemble": "--dgcv-special-background",
            "$part_1": "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='180' height='180' viewBox='0 0 180 180'%3E%3Cfilter id='r' x='0' y='0' width='100%' height='100%' color-interpolation-filters='sRGB'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.022 0.028' numOctaves='4' seed='13' stitchTiles='stitch' result='n'/%3E%3CfeColorMatrix in='n' type='matrix' values='0 0 0 0 0.78 0 0 0 0 0.93 0 0 0 0 0.86 1.05 0 0 0 -0.72'/%3E%3C/filter%3E%3Crect width='180' height='180' filter='url%28%23r%29'/%3E%3C/svg%3E\") ${rx}px ${ry}px / ${rs}px ${rs}px repeat, url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='-60 -60 120 120'%3E%3Cg transform='translate%28-16 -12%29 rotate%2824%29'%3E%3Cdefs%3E%3ClinearGradient id='pa' gradientUnits='userSpaceOnUse' x1='0' y1='-30' x2='0' y2='30'%3E%3Cstop offset='0' stop-color='%234E8A3E'/%3E%3Cstop offset='1' stop-color='%232F5E28'/%3E%3C/linearGradient%3E%3C/defs%3E%3Cpath d='M 0,0 L 28.7,-8.8 A 30 30 0 1 0 28.7,8.8 Z' fill='url%28%23pa%29' stroke='%2324491F' stroke-width='1.65'/%3E%3Cpath d='M 0,0 L 19.7,16.7' stroke='%237FB05E' stroke-width='1.50' opacity='0.45'/%3E%3Cpath d='M 0,0 L 1.4,25.8' stroke='%237FB05E' stroke-width='1.50' opacity='0.45'/%3E%3Cpath d='M 0,0 L -17.7,18.7' stroke='%237FB05E' stroke-width='1.50' opacity='0.45'/%3E%3Cpath d='M 0,0 L -25.8,0.0' stroke='%237FB05E' stroke-width='1.50' opacity='0.45'/%3E%3Cpath d='M 0,0 L -17.7,-18.7' stroke='%237FB05E' stroke-width='1.50' opacity='0.45'/%3E%3Cpath d='M 0,0 L 1.4,-25.8' stroke='%237FB05E' stroke-width='1.50' opacity='0.45'/%3E%3Cpath d='M 0,0 L 19.7,-16.7' stroke='%237FB05E' stroke-width='1.50' opacity='0.45'/%3E%3C/g%3E%3Cg transform='translate%2822 20%29 rotate%28196%29'%3E%3Cdefs%3E%3ClinearGradient id='pb' gradientUnits='userSpaceOnUse' x1='0' y1='-21' x2='0' y2='21'%3E%3Cstop offset='0' stop-color='%235C9A46'/%3E%3Cstop offset='1' stop-color='%2339702F'/%3E%3C/linearGradient%3E%3C/defs%3E%3Cpath d='M 0,0 L 20.1,-6.1 A 21 21 0 1 0 20.1,6.1 Z' fill='url%28%23pb%29' stroke='%232A5224' stroke-width='1.16'/%3E%3Cpath d='M 0,0 L 13.8,11.7' stroke='%238CBB68' stroke-width='1.05' opacity='0.45'/%3E%3Cpath d='M 0,0 L 1.0,18.0' stroke='%238CBB68' stroke-width='1.05' opacity='0.45'/%3E%3Cpath d='M 0,0 L -12.4,13.1' stroke='%238CBB68' stroke-width='1.05' opacity='0.45'/%3E%3Cpath d='M 0,0 L -18.1,0.0' stroke='%238CBB68' stroke-width='1.05' opacity='0.45'/%3E%3Cpath d='M 0,0 L -12.4,-13.1' stroke='%238CBB68' stroke-width='1.05' opacity='0.45'/%3E%3Cpath d='M 0,0 L 1.0,-18.0' stroke='%238CBB68' stroke-width='1.05' opacity='0.45'/%3E%3Cpath d='M 0,0 L 13.8,-11.7' stroke='%238CBB68' stroke-width='1.05' opacity='0.45'/%3E%3C/g%3E%3C/svg%3E\") calc(${g1x}% - (min(${g1s}px, 40%) / 2)) calc(${g1y}% - (min(${g1s}px, 40%) / 2)) / min(${g1s}px, 40%) min(${g1s}px, 40%) no-repeat, url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='-60 -60 120 120'%3E%3Cg transform='translate%28-10 8%29 rotate%28128%29'%3E%3Cdefs%3E%3ClinearGradient id='pc' gradientUnits='userSpaceOnUse' x1='0' y1='-27' x2='0' y2='27'%3E%3Cstop offset='0' stop-color='%238A5230'/%3E%3Cstop offset='1' stop-color='%235A3220'/%3E%3C/linearGradient%3E%3C/defs%3E%3Cpath d='M 0,0 L 25.8,-7.9 A 27 27 0 1 0 25.8,7.9 Z' fill='url%28%23pc%29' stroke='%23402314' stroke-width='1.49'/%3E%3Cpath d='M 0,0 L 17.7,15.0' stroke='%23B87A48' stroke-width='1.35' opacity='0.45'/%3E%3Cpath d='M 0,0 L 1.3,23.2' stroke='%23B87A48' stroke-width='1.35' opacity='0.45'/%3E%3Cpath d='M 0,0 L -16.0,16.9' stroke='%23B87A48' stroke-width='1.35' opacity='0.45'/%3E%3Cpath d='M 0,0 L -23.2,0.0' stroke='%23B87A48' stroke-width='1.35' opacity='0.45'/%3E%3Cpath d='M 0,0 L -16.0,-16.9' stroke='%23B87A48' stroke-width='1.35' opacity='0.45'/%3E%3Cpath d='M 0,0 L 1.3,-23.2' stroke='%23B87A48' stroke-width='1.35' opacity='0.45'/%3E%3Cpath d='M 0,0 L 17.7,-15.0' stroke='%23B87A48' stroke-width='1.35' opacity='0.45'/%3E%3C/g%3E%3Cg transform='translate%2824 -20%29 rotate%28302%29'%3E%3Cdefs%3E%3ClinearGradient id='pd' gradientUnits='userSpaceOnUse' x1='0' y1='-18' x2='0' y2='18'%3E%3Cstop offset='0' stop-color='%236E7C2E'/%3E%3Cstop offset='1' stop-color='%2347521C'/%3E%3C/linearGradient%3E%3C/defs%3E%3Cpath d='M 0,0 L 17.2,-5.3 A 18 18 0 1 0 17.2,5.3 Z' fill='url%28%23pd%29' stroke='%23333C14' stroke-width='0.99'/%3E%3Cpath d='M 0,0 L 11.8,10.0' stroke='%239BAA4E' stroke-width='0.90' opacity='0.45'/%3E%3Cpath d='M 0,0 L 0.8,15.5' stroke='%239BAA4E' stroke-width='0.90' opacity='0.45'/%3E%3Cpath d='M 0,0 L -10.6,11.2' stroke='%239BAA4E' stroke-width='0.90' opacity='0.45'/%3E%3Cpath d='M 0,0 L -15.5,0.0' stroke='%239BAA4E' stroke-width='0.90' opacity='0.45'/%3E%3Cpath d='M 0,0 L -10.6,-11.2' stroke='%239BAA4E' stroke-width='0.90' opacity='0.45'/%3E%3Cpath d='M 0,0 L 0.8,-15.5' stroke='%239BAA4E' stroke-width='0.90' opacity='0.45'/%3E%3Cpath d='M 0,0 L 11.8,-10.0' stroke='%239BAA4E' stroke-width='0.90' opacity='0.45'/%3E%3C/g%3E%3C/svg%3E\") calc(${g2x}% - (min(${g2s}px, 36%) / 2) + max(0, ${g2p} - 2) * 9999px) calc(${g2y}% - (min(${g2s}px, 36%) / 2)) / min(${g2s}px, 36%) min(${g2s}px, 36%) no-repeat, url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='-58 -58 116 116'%3E%3Cdefs%3E%3ClinearGradient id='bo' gradientUnits='userSpaceOnUse' x1='0' y1='-44' x2='0' y2='30'%3E%3Cstop offset='0' stop-color='%23F59A4A'/%3E%3Cstop offset='1' stop-color='%23C6400F'/%3E%3C/linearGradient%3E%3ClinearGradient id='to' gradientUnits='userSpaceOnUse' x1='0' y1='28' x2='0' y2='54'%3E%3Cstop offset='0' stop-color='%23C6400F' stop-opacity='0.95'/%3E%3Cstop offset='1' stop-color='%23EE8A44' stop-opacity='0.42'/%3E%3C/linearGradient%3E%3ClinearGradient id='fo' gradientUnits='userSpaceOnUse' x1='-10' y1='-16' x2='-23' y2='-2'%3E%3Cstop offset='0' stop-color='%23C6400F' stop-opacity='0.85'/%3E%3Cstop offset='1' stop-color='%23EE8A44' stop-opacity='0.38'/%3E%3C/linearGradient%3E%3ClinearGradient id='ho' gradientUnits='userSpaceOnUse' x1='10' y1='-16' x2='23' y2='-2'%3E%3Cstop offset='0' stop-color='%23C6400F' stop-opacity='0.85'/%3E%3Cstop offset='1' stop-color='%23EE8A44' stop-opacity='0.38'/%3E%3C/linearGradient%3E%3CclipPath id='co'%3E%3Cpath d='M 0,-44 C 7,-40 12,-27 12,-10 C 12,6 8,22 4,30 L -4,30 C -8,22 -12,6 -12,-10 C -12,-27 -7,-40 0,-44 Z'/%3E%3C/clipPath%3E%3C/defs%3E%3Cg transform='rotate%28-18%29' opacity='1'%3E%3Cpath d='M 4,26 C 9,36 15,46 13,52 C 7,50 2,43 0,37 C -2,43 -7,50 -13,52 C -15,46 -9,36 -4,26 Z' fill='url%28%23to%29'/%3E%3Cpath d='M -11,-18 C -18,-14 -23,-6 -21,-1 C -16,-5 -12,-11 -10,-15 Z' fill='url%28%23fo%29'/%3E%3Cpath d='M 11,-18 C 18,-14 23,-6 21,-1 C 16,-5 12,-11 10,-15 Z' fill='url%28%23ho%29'/%3E%3Cpath d='M 0,-44 C 7,-40 12,-27 12,-10 C 12,6 8,22 4,30 L -4,30 C -8,22 -12,6 -12,-10 C -12,-27 -7,-40 0,-44 Z' fill='url%28%23bo%29'/%3E%3Cg clip-path='url%28%23co%29'%3E%3Cellipse cx='-3' cy='-22' rx='7' ry='10' transform='rotate%2820 -3 -22%29' fill='%23FFD9A8' opacity='0.55'/%3E%3Cellipse cx='4' cy='4' rx='8' ry='12' transform='rotate%28-14 4 4%29' fill='%23B02F0A' opacity='0.50'/%3E%3Cpath d='M 0,-30 C 3,-16 3,4 1,20 C -1,4 -1,-16 0,-30 Z' fill='%233A1206' opacity='0.16'/%3E%3Cellipse cx='0' cy='30' rx='9' ry='11' fill='%23C6400F' opacity='0.55'/%3E%3C/g%3E%3Ccircle cx='-4.8' cy='-35' r='1.7' fill='%231F0A03' opacity='0.75'/%3E%3Ccircle cx='4.8' cy='-35' r='1.7' fill='%231F0A03' opacity='0.75'/%3E%3C/g%3E%3C/svg%3E\") calc(${k1x}% - (min(${k1s}px, 42%) / 2)) calc(${k1y}% - (min(${k1s}px, 42%) / 2)) / min(${k1s}px, 42%) min(${k1s}px, 42%) no-repeat",
            "$part_2@detail": "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='-58 -58 116 116'%3E%3Cdefs%3E%3ClinearGradient id='bk' gradientUnits='userSpaceOnUse' x1='0' y1='-44' x2='0' y2='30'%3E%3Cstop offset='0' stop-color='%23FBF6EC'/%3E%3Cstop offset='1' stop-color='%23E2DACB'/%3E%3C/linearGradient%3E%3ClinearGradient id='tk' gradientUnits='userSpaceOnUse' x1='0' y1='28' x2='0' y2='54'%3E%3Cstop offset='0' stop-color='%23E2DACB' stop-opacity='0.95'/%3E%3Cstop offset='1' stop-color='%23F7F1E5' stop-opacity='0.42'/%3E%3C/linearGradient%3E%3ClinearGradient id='fk' gradientUnits='userSpaceOnUse' x1='-10' y1='-16' x2='-23' y2='-2'%3E%3Cstop offset='0' stop-color='%23E2DACB' stop-opacity='0.85'/%3E%3Cstop offset='1' stop-color='%23F7F1E5' stop-opacity='0.38'/%3E%3C/linearGradient%3E%3ClinearGradient id='hk' gradientUnits='userSpaceOnUse' x1='10' y1='-16' x2='23' y2='-2'%3E%3Cstop offset='0' stop-color='%23E2DACB' stop-opacity='0.85'/%3E%3Cstop offset='1' stop-color='%23F7F1E5' stop-opacity='0.38'/%3E%3C/linearGradient%3E%3CclipPath id='ck'%3E%3Cpath d='M 0,-44 C 7,-40 12,-27 12,-10 C 12,6 8,22 4,30 L -4,30 C -8,22 -12,6 -12,-10 C -12,-27 -7,-40 0,-44 Z'/%3E%3C/clipPath%3E%3C/defs%3E%3Cg transform='rotate%28152%29' opacity='1'%3E%3Cpath d='M 4,26 C 9,36 15,46 13,52 C 7,50 2,43 0,37 C -2,43 -7,50 -13,52 C -15,46 -9,36 -4,26 Z' fill='url%28%23tk%29'/%3E%3Cpath d='M -11,-18 C -18,-14 -23,-6 -21,-1 C -16,-5 -12,-11 -10,-15 Z' fill='url%28%23fk%29'/%3E%3Cpath d='M 11,-18 C 18,-14 23,-6 21,-1 C 16,-5 12,-11 10,-15 Z' fill='url%28%23hk%29'/%3E%3Cpath d='M 0,-44 C 7,-40 12,-27 12,-10 C 12,6 8,22 4,30 L -4,30 C -8,22 -12,6 -12,-10 C -12,-27 -7,-40 0,-44 Z' fill='url%28%23bk%29'/%3E%3Cg clip-path='url%28%23ck%29'%3E%3Cellipse cx='-2' cy='-26' rx='8' ry='11' transform='rotate%2816 -2 -26%29' fill='%23D8402A' opacity='0.92'/%3E%3Cellipse cx='3' cy='2' rx='9' ry='13' transform='rotate%28-12 3 2%29' fill='%23D8402A' opacity='0.88'/%3E%3Cellipse cx='-4' cy='20' rx='5' ry='7' transform='rotate%288 -4 20%29' fill='%23D8402A' opacity='0.75'/%3E%3Cpath d='M 0,-30 C 3,-16 3,4 1,20 C -1,4 -1,-16 0,-30 Z' fill='%234A2418' opacity='0.16'/%3E%3Cellipse cx='0' cy='30' rx='9' ry='11' fill='%23E2DACB' opacity='0.55'/%3E%3C/g%3E%3Ccircle cx='-4.8' cy='-35' r='1.7' fill='%23241009' opacity='0.75'/%3E%3Ccircle cx='4.8' cy='-35' r='1.7' fill='%23241009' opacity='0.75'/%3E%3C/g%3E%3C/svg%3E\") calc(${k2x}% - (min(${k2s}px, 38%) / 2) + max(0, ${k2p} - 2) * 9999px) calc(${k2y}% - (min(${k2s}px, 38%) / 2)) / min(${k2s}px, 38%) min(${k2s}px, 38%) no-repeat, url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='-58 -58 116 116'%3E%3Cdefs%3E%3ClinearGradient id='bg' gradientUnits='userSpaceOnUse' x1='0' y1='-44' x2='0' y2='30'%3E%3Cstop offset='0' stop-color='%23F7CE6A'/%3E%3Cstop offset='1' stop-color='%23D8992A'/%3E%3C/linearGradient%3E%3ClinearGradient id='tg' gradientUnits='userSpaceOnUse' x1='0' y1='28' x2='0' y2='54'%3E%3Cstop offset='0' stop-color='%23D8992A' stop-opacity='0.95'/%3E%3Cstop offset='1' stop-color='%23F2C158' stop-opacity='0.42'/%3E%3C/linearGradient%3E%3ClinearGradient id='fg' gradientUnits='userSpaceOnUse' x1='-10' y1='-16' x2='-23' y2='-2'%3E%3Cstop offset='0' stop-color='%23D8992A' stop-opacity='0.85'/%3E%3Cstop offset='1' stop-color='%23F2C158' stop-opacity='0.38'/%3E%3C/linearGradient%3E%3ClinearGradient id='hg' gradientUnits='userSpaceOnUse' x1='10' y1='-16' x2='23' y2='-2'%3E%3Cstop offset='0' stop-color='%23D8992A' stop-opacity='0.85'/%3E%3Cstop offset='1' stop-color='%23F2C158' stop-opacity='0.38'/%3E%3C/linearGradient%3E%3CclipPath id='cg'%3E%3Cpath d='M 0,-44 C 7,-40 12,-27 12,-10 C 12,6 8,22 4,30 L -4,30 C -8,22 -12,6 -12,-10 C -12,-27 -7,-40 0,-44 Z'/%3E%3C/clipPath%3E%3C/defs%3E%3Cg transform='rotate%2864%29' opacity='1'%3E%3Cpath d='M 4,26 C 9,36 15,46 13,52 C 7,50 2,43 0,37 C -2,43 -7,50 -13,52 C -15,46 -9,36 -4,26 Z' fill='url%28%23tg%29'/%3E%3Cpath d='M -11,-18 C -18,-14 -23,-6 -21,-1 C -16,-5 -12,-11 -10,-15 Z' fill='url%28%23fg%29'/%3E%3Cpath d='M 11,-18 C 18,-14 23,-6 21,-1 C 16,-5 12,-11 10,-15 Z' fill='url%28%23hg%29'/%3E%3Cpath d='M 0,-44 C 7,-40 12,-27 12,-10 C 12,6 8,22 4,30 L -4,30 C -8,22 -12,6 -12,-10 C -12,-27 -7,-40 0,-44 Z' fill='url%28%23bg%29'/%3E%3Cg clip-path='url%28%23cg%29'%3E%3Cellipse cx='0' cy='-14' rx='8' ry='13' transform='rotate%2810 0 -14%29' fill='%23FFEBB0' opacity='0.55'/%3E%3Cellipse cx='2' cy='14' rx='7' ry='10' transform='rotate%28-18 2 14%29' fill='%23B87716' opacity='0.45'/%3E%3Cpath d='M 0,-30 C 3,-16 3,4 1,20 C -1,4 -1,-16 0,-30 Z' fill='%234A320A' opacity='0.16'/%3E%3Cellipse cx='0' cy='30' rx='9' ry='11' fill='%23D8992A' opacity='0.55'/%3E%3C/g%3E%3Ccircle cx='-4.8' cy='-35' r='1.7' fill='%23241804' opacity='0.75'/%3E%3Ccircle cx='4.8' cy='-35' r='1.7' fill='%23241804' opacity='0.75'/%3E%3C/g%3E%3C/svg%3E\") calc(${k3x}% - (min(${k3s}px, 34%) / 2) + ${k3p} * 9999px) calc(${k3y}% - (min(${k3s}px, 34%) / 2)) / min(${k3s}px, 34%) min(${k3s}px, 34%) no-repeat, url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='-58 -58 116 116'%3E%3Cdefs%3E%3ClinearGradient id='bu' gradientUnits='userSpaceOnUse' x1='0' y1='-44' x2='0' y2='30'%3E%3Cstop offset='0' stop-color='%233C4249'/%3E%3Cstop offset='1' stop-color='%2320262C'/%3E%3C/linearGradient%3E%3ClinearGradient id='tu' gradientUnits='userSpaceOnUse' x1='0' y1='28' x2='0' y2='54'%3E%3Cstop offset='0' stop-color='%2320262C' stop-opacity='0.95'/%3E%3Cstop offset='1' stop-color='%23333940' stop-opacity='0.42'/%3E%3C/linearGradient%3E%3ClinearGradient id='fu' gradientUnits='userSpaceOnUse' x1='-10' y1='-16' x2='-23' y2='-2'%3E%3Cstop offset='0' stop-color='%2320262C' stop-opacity='0.85'/%3E%3Cstop offset='1' stop-color='%23333940' stop-opacity='0.38'/%3E%3C/linearGradient%3E%3ClinearGradient id='hu' gradientUnits='userSpaceOnUse' x1='10' y1='-16' x2='23' y2='-2'%3E%3Cstop offset='0' stop-color='%2320262C' stop-opacity='0.85'/%3E%3Cstop offset='1' stop-color='%23333940' stop-opacity='0.38'/%3E%3C/linearGradient%3E%3CclipPath id='cu'%3E%3Cpath d='M 0,-44 C 7,-40 12,-27 12,-10 C 12,6 8,22 4,30 L -4,30 C -8,22 -12,6 -12,-10 C -12,-27 -7,-40 0,-44 Z'/%3E%3C/clipPath%3E%3C/defs%3E%3Cg transform='rotate%28-108%29' opacity='0.88'%3E%3Cpath d='M 4,26 C 9,36 15,46 13,52 C 7,50 2,43 0,37 C -2,43 -7,50 -13,52 C -15,46 -9,36 -4,26 Z' fill='url%28%23tu%29'/%3E%3Cpath d='M -11,-18 C -18,-14 -23,-6 -21,-1 C -16,-5 -12,-11 -10,-15 Z' fill='url%28%23fu%29'/%3E%3Cpath d='M 11,-18 C 18,-14 23,-6 21,-1 C 16,-5 12,-11 10,-15 Z' fill='url%28%23hu%29'/%3E%3Cpath d='M 0,-44 C 7,-40 12,-27 12,-10 C 12,6 8,22 4,30 L -4,30 C -8,22 -12,6 -12,-10 C -12,-27 -7,-40 0,-44 Z' fill='url%28%23bu%29'/%3E%3Cg clip-path='url%28%23cu%29'%3E%3Cellipse cx='-3' cy='-24' rx='7' ry='10' transform='rotate%2818 -3 -24%29' fill='%23F0EADC' opacity='0.85'/%3E%3Cellipse cx='4' cy='6' rx='8' ry='12' transform='rotate%28-16 4 6%29' fill='%23F0EADC' opacity='0.78'/%3E%3Cpath d='M 0,-30 C 3,-16 3,4 1,20 C -1,4 -1,-16 0,-30 Z' fill='%2312161A' opacity='0.16'/%3E%3Cellipse cx='0' cy='30' rx='9' ry='11' fill='%2320262C' opacity='0.55'/%3E%3C/g%3E%3Ccircle cx='-4.8' cy='-35' r='1.7' fill='%230A0D10' opacity='0.75'/%3E%3Ccircle cx='4.8' cy='-35' r='1.7' fill='%230A0D10' opacity='0.75'/%3E%3C/g%3E%3C/svg%3E\") calc(${k4x}% - (min(${k4s}px, 30%) / 2) + (1 - ${k3p}) * 9999px) calc(${k4y}% - (min(${k4s}px, 30%) / 2)) / min(${k4s}px, 30%) min(${k4s}px, 30%) no-repeat",
            "$part_3": "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 800 500' preserveAspectRatio='xMidYMid slice'%3E%3Cdefs%3E%3ClinearGradient id='water' gradientUnits='userSpaceOnUse' x1='0' y1='0' x2='0' y2='500'%3E%3Cstop offset='0' stop-color='%2316302A'/%3E%3Cstop offset='0.45' stop-color='%23102420'/%3E%3Cstop offset='1' stop-color='%23081613'/%3E%3C/linearGradient%3E%3CradialGradient id='sun' cx='0.5' cy='0.5' r='0.5'%3E%3Cstop offset='0' stop-color='%239FD8B0' stop-opacity='0.13'/%3E%3Cstop offset='1' stop-color='%239FD8B0' stop-opacity='0'/%3E%3C/radialGradient%3E%3Cfilter id='silt' x='0' y='0' width='100%' height='100%' color-interpolation-filters='sRGB'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.009 0.013' numOctaves='4' seed='21' stitchTiles='stitch' result='n'/%3E%3CfeColorMatrix in='n' type='matrix' values='0 0 0 0 0.16 0 0 0 0 0.26 0 0 0 0 0.20 0.55 0 0 0 -0.18' result='m'/%3E%3CfeComposite in='m' in2='SourceGraphic' operator='in'/%3E%3C/filter%3E%3Cfilter id='caustic' x='0' y='0' width='100%' height='100%' color-interpolation-filters='sRGB'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.013 0.017' numOctaves='3' seed='5' stitchTiles='stitch' result='n'/%3E%3CfeColorMatrix in='n' type='matrix' values='0 0 0 0 0.72 0 0 0 0 0.92 0 0 0 0 0.82 1.35 0 0 0 -0.98' result='m'/%3E%3CfeComposite in='m' in2='SourceGraphic' operator='in'/%3E%3C/filter%3E%3C/defs%3E%3Crect x='0' y='0' width='800' height='500' fill='url%28%23water%29'/%3E%3Crect x='0' y='0' width='800' height='500' fill='%23FFFFFF' filter='url%28%23silt%29' opacity='0.5'/%3E%3Cellipse cx='300' cy='190' rx='330' ry='210' fill='url%28%23sun%29'/%3E%3Cellipse cx='128' cy='392' rx='74' ry='46' fill='%2320342B' opacity='0.55'/%3E%3Cellipse cx='624' cy='118' rx='92' ry='54' fill='%231C3028' opacity='0.5'/%3E%3Cellipse cx='452' cy='438' rx='58' ry='34' fill='%2322362C' opacity='0.45'/%3E%3Crect x='0' y='0' width='800' height='500' fill='%23FFFFFF' filter='url%28%23caustic%29' opacity='0.34'/%3E%3C/svg%3E\") ${sx}% ${sy}% / cover no-repeat, linear-gradient(to bottom, #16302A 0%, #102420 45%, #081613 100%) center / cover no-repeat, #102420",
            "--dgcv-special-text": "#F0B75A",
            "--plaque-fill": "#162C27",
            "--plaque-border": "#F0B75A",
            "--dgcv-table-shadow": "0 2px 14px rgba(3, 10, 8, 0.55)",
            "--dgcv-text-shadow": "none",
            "--dgcv-hover-transform": "translateY(-1px)",
            "--dgcv-hover-transition": "background-color 150ms ease, color 150ms ease, transform 150ms ease",
        },
    ),
    "chalkboard_green": ThemeConfig(
        bg_primary="#1A2420",
        bg_surface="#212F29",
        bg_alt="#1D2925",
        bg_hover="#F2D675",
        text_main="#E9E6DA",
        text_heading="#F6DE8C",
        text_hover="#1A2420",
        text_alt="#C9D2C6",
        border_main="#9C7A50",
        border_alt="#634B32",
        bg_action="#E8748F",
        text_on_action="#241016",
        bg_action_hover="#F196AC",
        bg_error="#D9605A",
        text_on_error="#230A08",
        bg_success="#7FB06B",
        text_on_success="#0E1D0A",
        font_family="'Chalkboard SE', 'Comic Sans MS', 'Marker Felt', cursive",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "$randomized": "axdx=12:86 axdy=14:78 axsz=56:92 cndx=12:86 cndy=14:78 cnsz=40:64 sx=42:58 sy=44:56",
            "--dgcv-special-background": "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140' viewBox='0 0 140 140'%3E%3Cfilter id='d' x='0' y='0' width='100%' height='100%' color-interpolation-filters='sRGB'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' seed='7' stitchTiles='stitch' result='n'/%3E%3CfeColorMatrix in='n' type='matrix' values='0 0 0 0 0.957 0 0 0 0 0.945 0 0 0 0 0.902 0.26 0 0 0 -0.14'/%3E%3C/filter%3E%3Crect width='140' height='140' filter='url%28%23d%29'/%3E%3C/svg%3E\") 0 0 / 140px 140px repeat, url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='-45 -45 90 90'%3E%3Cg stroke='%23F4F1E6' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round' fill='none' opacity='0.40'%3E%3Cpath d='M 0.0,0.0 L 0.0,-38.0'/%3E%3Cpath d='M 3.3,-31.8 L 0.0,-38.0 L -3.3,-31.8'/%3E%3Cpath d='M 0.0,0.0 L 34.0,-6.0'/%3E%3Cpath d='M 28.5,-1.7 L 34.0,-6.0 L 27.3,-8.2'/%3E%3Cpath d='M 0.0,0.0 L -26.0,18.0'/%3E%3Cpath d='M -22.8,11.8 L -26.0,18.0 L -19.0,17.2'/%3E%3C/g%3E%3C/svg%3E\") calc(${axdx}% - (min(${axsz}px, 42%) / 2)) calc(${axdy}% - (min(${axsz}px, 42%) / 2)) / min(${axsz}px, 42%) min(${axsz}px, 42%) no-repeat, url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='-20 -26 40 52'%3E%3Cg stroke='%23F4F1E6' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round' fill='none' opacity='0.40'%3E%3Cpath d='M 0,-24 L -13.0,4.0'/%3E%3Cpath d='M 0,-24 L 13.0,4.0'/%3E%3Cellipse cx='0' cy='4' rx='13' ry='4.2'/%3E%3C/g%3E%3C/svg%3E\") calc(${cndx}% - (min(${cnsz}px, 30%) / 2)) calc(${cndy}% - (min(${cnsz}px, 30%) * 0.65)) / min(${cnsz}px, 30%) calc(min(${cnsz}px, 30%) * 1.3) no-repeat, url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 800 500' preserveAspectRatio='xMidYMid slice'%3E%3Cdefs%3E%3ClinearGradient id='slate' x1='0' y1='0' x2='0' y2='1'%3E%3Cstop offset='0' stop-color='%2326342D'/%3E%3Cstop offset='0.5' stop-color='%231E2A24'/%3E%3Cstop offset='1' stop-color='%23161F1A'/%3E%3C/linearGradient%3E%3CradialGradient id='smudge' cx='0.5' cy='0.5' r='0.5'%3E%3Cstop offset='0' stop-color='%23F4F1E6' stop-opacity='0.07'/%3E%3Cstop offset='0.6' stop-color='%23F4F1E6' stop-opacity='0.03'/%3E%3Cstop offset='1' stop-color='%23F4F1E6' stop-opacity='0'/%3E%3C/radialGradient%3E%3ClinearGradient id='streak' x1='0' y1='0' x2='1' y2='0'%3E%3Cstop offset='0' stop-color='%23F4F1E6' stop-opacity='0'/%3E%3Cstop offset='0.5' stop-color='%23F4F1E6' stop-opacity='0.05'/%3E%3Cstop offset='1' stop-color='%23F4F1E6' stop-opacity='0'/%3E%3C/linearGradient%3E%3C/defs%3E%3Crect x='0' y='0' width='800' height='500' fill='url%28%23slate%29'/%3E%3Cellipse cx='210' cy='140' rx='170' ry='95' fill='url%28%23smudge%29'/%3E%3Cellipse cx='560' cy='330' rx='210' ry='110' fill='url%28%23smudge%29'/%3E%3Crect x='40' y='250' width='420' height='16' fill='url%28%23streak%29'/%3E%3Crect x='300' y='400' width='500' height='12' fill='url%28%23streak%29'/%3E%3C/svg%3E\") ${sx}% ${sy}% / cover no-repeat, linear-gradient(to bottom, #26342D 0%, #1E2A24 50%, #161F1A 100%) center / cover no-repeat, #1E2A24",
            "--dgcv-special-text": "#F6DE8C",
            "--dgcv-table-shadow": "0 2px 10px rgba(4, 8, 6, 0.45)",
            "--dgcv-text-shadow": "none",
            "--dgcv-hover-transform": "none",
            "--dgcv-hover-transition": "background-color 110ms ease, color 110ms ease",
        },
    ),
}


def get_dgcv_themes(
    show_themes: Union[bool, str, List[str]] = False,
) -> Union[List[str], str]:
    theme_names = sorted(THEME_REGISTRY.keys())

    if not show_themes:
        return theme_names

    display_names = theme_names
    if isinstance(show_themes, str):
        display_names = [show_themes] if show_themes in THEME_REGISTRY else []
    elif isinstance(show_themes, list):
        display_names = [name for name in show_themes if name in THEME_REGISTRY]

    ui_bg = "#1a1a1a"
    ui_text = "#eceff4"
    ui_border = "#333333"

    html_output = [
        f"""
<div class='dgcv-gallery-container' style='background-color: {ui_bg}; color: {ui_text}; padding: 30px; font-family: system-ui, sans-serif;'>
    <div style='display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid {ui_border}; margin-bottom: 30px; padding-bottom: 10px;'>
        <h1 style='margin:0;'><code>dgcv</code> Theme Registry</h1>
        <span style='opacity: 0.6;'>{len(display_names)} Themes Loaded</span>
    </div>

    <style>
        .dgcv-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); 
            gap: 25px; 
        }}
        .theme-card {{ 
            background: #252525; 
            border: 1px solid {ui_border}; 
            border-radius: 10px; 
            padding: 15px; 
            display: flex; 
            flex-direction: column;
            transition: border-color 0.3s ease;
        }}
        .theme-card:hover {{ border-color: #555; }}
        .card-header {{ 
            margin-bottom: 12px; 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
        }}
        .swatch-grid {{ 
            display: grid; 
            grid-template-columns: repeat(4, 1fr); 
            gap: 8px; 
        }}
        .swatch {{ 
            height: 40px; 
            display: flex; 
            flex-direction: column;
            align-items: center; 
            justify-content: center; 
            font-size: 10px; 
            font-weight: bold; 
            border-radius: 4px;
            text-transform: uppercase;
            overflow: hidden;
        }}
        .swatch-label {{ font-size: 9px; opacity: 0.7; margin-bottom: 2px; text-transform: none; }}
        .hover-zone {{
            grid-column: span 4;
            height: 50px;
            margin-top: 8px;
            cursor: help;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 6px;
            font-size: 11px;
            letter-spacing: 1px;
            font-weight: 900;
        }}
    </style>
    <div class="dgcv-grid">
"""
    ]

    for name in display_names:
        style_id = f"card-{name.replace('_', '-')}"

        raw_css, theme_data = get_style(name, return_theme_data=True)
        font = theme_data.font_family.replace("inherit", "system")
        border_radius = int(
            theme_data.custom_css_vars.get("--dgcv-border-radius", "12px").replace(
                "px", ""
            )
        )
        hover_zone_text = (
            "var(--dgcv-special-text,var(--dgcv-text-heading))"
            if "--dgcv-special-background" in raw_css
            else "var(--dgcv-text-main)"
        )
        scoped_css = raw_css.replace(":root", f".{style_id}")

        html_output.append(f"""
<style>
    {scoped_css}

    .{style_id} .hover-zone {{
        background: var(--dgcv-special-background, var(--dgcv-bg-primary));
        color: {hover_zone_text};
        box-shadow: var(--dgcv-table-shadow, none);
        border: var(--dgcv-border-width, 1px) solid var(--dgcv-border-main);
        border-image: var(--dgcv-border-image, none);
        border-radius: {border_radius};
        transition: var(--dgcv-hover-transition, all 0.2s ease);
        text-shadow: var(--dgcv-text-shadow, none);
    }}
    .{style_id} .hover-zone:hover {{
        background: var(--dgcv-bg-hover) !important;
        color: var(--dgcv-text-hover) !important;
        transform: var(--dgcv-hover-transform, none);
        box-shadow: var(--dgcv-table-shadow, none);
    }}

    .{style_id} .swatch-head {{
        background: var(--dgcv-bg-surface); 
        color: var(--dgcv-text-heading); 
        text-shadow: var(--dgcv-text-shadow, none);
    }}
    .{style_id} .swatch-main {{
        background: var(--dgcv-bg-primary); 
        color: var(--dgcv-text-main); 
        border: 1px solid var(--dgcv-border-main);
    }}
    .{style_id} .swatch-alt {{
        background: var(--dgcv-bg-alt); 
        color: var(--dgcv-text-alt);
    }}
    .{style_id} .swatch-hov {{
        background: var(--dgcv-bg-hover); 
        color: var(--dgcv-text-hover);
    }}
    .{style_id} .swatch-act {{
        background: var(--dgcv-bg-action); 
        color: var(--dgcv-text-on-action);
    }}
    .{style_id} .swatch-suc {{
        background: var(--dgcv-bg-success); 
        color: var(--dgcv-text-on-success);
    }}
    .{style_id} .swatch-err {{
        background: var(--dgcv-bg-error); 
        color: var(--dgcv-text-on-error);
    }}
    .{style_id} .swatch-brd {{
        background: var(--dgcv-bg-primary); 
        color: var(--dgcv-text-main);
        border: 2px solid var(--dgcv-border-alt);
    }}
</style>

<div class="theme-card {style_id}" style="font-family: var(--dgcv-font-family);">
    <div class="card-header">
        <strong style="color: #fff;">{name}</strong>
        <span style="font-family: var(--dgcv-font-family); font-size: 9px; opacity: 0.5;"><span style="font-size: 9px; opacity: 0.5;">{"fonts: " if "," in font else "font: "}</span>{font}</span>
    </div>

    <div class="swatch-grid">
        <div class="swatch swatch-head">HEAD</div>
        <div class="swatch swatch-main">MAIN</div>
        <div class="swatch swatch-alt">ALT</div>
        <div class="swatch swatch-hov">HOV</div>
        <div class="swatch swatch-act">ACT</div>
        <div class="swatch swatch-suc">SUC</div>
        <div class="swatch swatch-err">ERR</div>
        <div class="swatch swatch-brd">BRD-A</div>
        <div class="hover-zone">HOVER TEST</div>
    </div>
</div>
""")

    html_output.append("</div></div>")
    from ._config import _try_wrap_html

    return _try_wrap_html("".join(html_output))


def _set_dgcv_default_theme(theme: str):
    global dgcv_display_theme
    dgcv_display_theme = theme


def get_legible_hex(h: float, L: float, s: float) -> str:
    r, g, b = colorsys.hls_to_rgb(h, L, s)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def get_random_theme(vibrancy: float = 0.2) -> ThemeConfig:
    is_dark_mode = random.choice([True, False])
    base_hue = random.random()
    sat_base = max(0.1, min(1.0, vibrancy))
    sat_accent = max(0.2, min(1.0, vibrancy + 0.4))
    if is_dark_mode:
        bg_l, surface_l, alt_l = 0.08, 0.14, 0.18
        text_l = 0.95
    else:
        bg_l, surface_l, alt_l = 0.98, 0.94, 0.90
        text_l = 0.05

    bg_primary = get_legible_hex(base_hue, bg_l, sat_base * 0.4)
    bg_surface = get_legible_hex(base_hue, surface_l, sat_base * 0.5)
    bg_alt = get_legible_hex(base_hue, alt_l, sat_base * 0.5)
    accent_hue = (base_hue + random.uniform(0.3, 0.7)) % 1.0
    bg_hover = get_legible_hex(accent_hue, 0.5, sat_accent)
    text_heading = get_legible_hex(accent_hue, 0.6 if is_dark_mode else 0.3, sat_accent)

    custom_css_vars = {
        "--dgcv-border-width": "1px",
        "--dgcv-hover-transform": "scale(1.02) translateY(-2px)",
        "--dgcv-hover-transition": "all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275)",
    }

    rarity = random.random()
    if rarity < 0.05:
        bg_primary = "linear-gradient(to bottom, #2a2a72, #009ffd)"
        custom_css_vars["--dgcv-table-shadow"] = "0 0 15px rgba(247, 216, 75, 0.9)"
        custom_css_vars["--dgcv-special-background"] = bg_primary
        text_main = "#ffffff"
        border_main = "#f7d84b"
    elif rarity < 0.10:
        bg_primary = get_legible_hex(base_hue, bg_l, sat_base * 0.4)
        custom_css_vars["--dgcv-special-background"] = (
            "radial-gradient(circle, #1a1a2e, #0f3460)"
        )
        custom_css_vars["--dgcv-table-shadow"] = "0 0 20px rgba(34, 166, 179, 0.8)"
        custom_css_vars["--dgcv-border-width"] = "2px"
        border_main = "#22a6b3"
        text_main = get_legible_hex(base_hue, text_l, 0.1)
    elif rarity < 0.15:
        bg_primary = get_legible_hex(base_hue, bg_l, sat_base * 0.4)
        custom_css_vars["--dgcv-special-background"] = (
            "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
        )
        custom_css_vars["--dgcv-table-shadow"] = "0 10px 30px rgba(0,0,0,0.5)"
        custom_css_vars["--dgcv-hover-transform"] = "rotate(1deg) scale(1.01)"
        text_heading = "#a1ffce"
        text_main = get_legible_hex(base_hue, text_l, 0.1)
        border_main = get_legible_hex(base_hue, 0.45, sat_base)
    else:
        text_main = get_legible_hex(base_hue, text_l, 0.1)
        border_main = get_legible_hex(base_hue, 0.45, sat_base)

    out = ThemeConfig(
        bg_primary=bg_primary,
        bg_surface=bg_surface,
        bg_alt=bg_alt,
        bg_hover=bg_hover,
        text_main=text_main,
        text_heading=text_heading,
        text_hover="#ffffff" if is_dark_mode else "#000000",
        text_alt=get_legible_hex(base_hue, text_l, 0.2),
        border_main=border_main,
        border_alt=get_legible_hex(base_hue, 0.55, sat_base * 0.5),
        font_family=random.choice(
            [
                "'Courier New', monospace",
                "Georgia, serif",
                "Impact, sans-serif",
                "Verdana",
            ]
        ),
        bg_action=bg_hover,
        text_on_action="#ffffff",
        bg_action_hover=get_legible_hex(accent_hue, 0.4, sat_accent),
        bg_error="#ff4b2b",
        text_on_error="#ffffff",
        bg_success="#00ff87",
        text_on_success="#000000",
        custom_css_vars=custom_css_vars,
    )
    if get_dgcv_settings_registry().get("DEBUG", False):
        print(out.registry_format("random"))
    return out


def _theme_from_mapping(data: Dict, theme_name: str = "<raw theme>") -> ThemeConfig:
    known = {f.name for f in fields(ThemeConfig)}
    required = {
        f.name
        for f in fields(ThemeConfig)
        if f.default is MISSING and f.default_factory is MISSING
    }

    supplied, custom_css_vars, unknown = {}, {}, []
    for key, value in data.items():
        if key in ("name", "description"):
            continue
        if key == "custom_css_vars":
            if not isinstance(value, dict):
                raise ValueError(
                    f"theme {theme_name!r}: 'custom_css_vars' must be a dict, "
                    f"got {type(value).__name__}"
                )
            custom_css_vars.update(value)
        elif key.startswith("--") or key.startswith("$"):
            custom_css_vars[key] = value
        elif key in known:
            supplied[key] = value
        else:
            unknown.append(key)

    if unknown:
        raise ValueError(
            f"theme {theme_name!r}: unrecognized keys {sorted(unknown)}; expected "
            f"ThemeConfig fields, '--' custom properties, or '$' stubs"
        )
    missing = required - set(supplied)
    if missing:
        raise ValueError(
            f"theme {theme_name!r}: missing required fields {sorted(missing)}"
        )

    supplied["custom_css_vars"] = custom_css_vars
    return ThemeConfig(**supplied)


def _coerce_theme_data(theme, theme_name: str = "<raw theme>"):
    if isinstance(theme, ThemeConfig):
        return theme
    if isinstance(theme, dict):
        return _theme_from_mapping(theme, theme_name)
    return None


def get_style(
    theme_name: Union[str, ThemeConfig, Dict],
    *args,
    return_theme_data: bool = False,
    minimal=False,
    **kwargs,
) -> str:
    """
    Returns a CSS ``:root { }`` block of dgcv theme variables for the given theme.

    Looks up ``theme_name`` in the THEME_REGISTRY, or accepts raw theme data in its
    place, and emits all theme fields and custom variables as scoped CSS custom
    properties via ``ThemeConfig.to_css_string``.

    Parameters:
    -----------
    theme_name: str | ThemeConfig | dict
        Name of the theme to look up. A light autocomplete (via the python difflib)
        will replace malformed str names to nearby themes if they exist.
        If ``theme_name`` is ``"random"``, a theme is selected at random from the
        registry. If ``theme_name`` or a close match is not found in the registry
        and is not ``"random"``, falls back silently to dgcv settings defualts.

        Raw theme data may be supplied instead of a name, which renders a theme
        without registering it. A ``ThemeConfig`` is used as given. A dict may pair
        ``ThemeConfig`` field names with values and carry custom properties either
        under a nested ``custom_css_vars`` key or as top level keys beginning with
        ``--`` (or ``$``, for the assembly and randomization stubs), so that a
        single flat JSON object is a complete theme. The keys ``name`` and
        ``description`` are allowed as metadata and ignored; any other unrecognized
        key, or a missing required field, raises ``ValueError``. Raw themes pass
        through the same assembly and randomization pipeline as registered ones.

        Raw theme data is also honored when it is the value of the ``theme`` entry
        of the dgcv settings registry, which is how it reaches renderers that only
        forward theme names.

    Returns:
    --------
    str
        a ``:root { }`` CSS block with all theme variables indented and
        ready for injection into a ``<style>`` tag.
    """
    default_dark, default_light = "matte_slate_soft", "paper_graphite"  # hard coded
    aliases = {
        "chalkboard": "chalkboard_green",
        "gruv": "gruvbox_dark",
        "dark": default_dark,
        "light": default_light,
    }
    theme_data = _coerce_theme_data(theme_name)
    if theme_data is not None:
        theme_name = (
            theme_name.get("name", "<raw theme>")
            if isinstance(theme_name, dict)
            else "<raw theme>"
        )
    else:
        theme_name = theme_name.lower()
        if difflib.get_close_matches(theme_name, ["shuffle"], n=1, cutoff=0.6):
            theme_name = random.choice(get_dgcv_themes())
        theme_name = aliases.get(theme_name, theme_name)
        if theme_name not in THEME_REGISTRY and theme_name != "random":
            close = difflib.get_close_matches(
                theme_name, THEME_REGISTRY.keys(), n=1, cutoff=0.6
            )
            if close:
                theme_name = close[0]
            else:
                fallback = get_dgcv_settings_registry().get("theme", default_dark)
                theme_data = _coerce_theme_data(fallback)
                theme_name = (
                    "<raw theme>"
                    if theme_data is not None
                    else aliases.get(fallback, default_dark)
                )
        if theme_data is None:
            theme_data = (
                get_random_theme()
                if theme_name == "random"
                else THEME_REGISTRY[theme_name]
            )
    theme_data = assemble_parts(theme_data, minimal=minimal, theme_name=theme_name)
    theme_data = apply_randomization(theme_data, theme_name, minimal=minimal)
    if return_theme_data:
        return f":root {{\n    {theme_data.to_css_string()}\n}}", theme_data
    return f":root {{\n    {theme_data.to_css_string()}\n}}"


# -----------------------------------------------------------------------------
# randomization
# -----------------------------------------------------------------------------


RANDOMIZED_KEY = "$randomized"
ASSEMBLE_KEY = "$assemble"

_PART_RE = re.compile(r"^\$part_(\d+)(?:@(\w+))?$")

_SLOT_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)=(-?\d+):(-?\d+)$")
_PHI_CONJUGATE = 0.6180339887498949

_rng = random.Random()
_cursors: Dict[str, float] = {}


class ThemeRandomizationError(ValueError):
    """Raised when a theme's randomization stub or tokens are malformed."""


class ThemeAssemblyError(ValueError):
    """Raised when a theme's part stubs are malformed."""


def assemble_parts(theme_data, minimal: bool = False, theme_name: str = "<theme>"):
    target = theme_data.custom_css_vars.get(ASSEMBLE_KEY)
    if not target:
        return theme_data
    target = target.strip()

    parts, rest, seen = [], {}, set()
    for key, value in theme_data.custom_css_vars.items():
        if key == ASSEMBLE_KEY:
            continue
        match = _PART_RE.match(key)
        if match is None:
            if key.startswith("$part"):
                raise ThemeAssemblyError(
                    f"theme {theme_name!r}: malformed part key {key!r}; "
                    f"expected $part_N or $part_N@detail"
                )
            rest[key] = value
            continue
        index, tag = int(match.group(1)), match.group(2)
        if tag not in (None, "detail"):
            raise ThemeAssemblyError(
                f"theme {theme_name!r}: unknown part tag {tag!r} on {key!r}; "
                f"only @detail is understood"
            )
        if index in seen:
            raise ThemeAssemblyError(
                f"theme {theme_name!r}: duplicate part index {index}"
            )
        seen.add(index)
        if tag == "detail" and minimal:
            continue
        parts.append((index, value))

    if not parts:
        raise ThemeAssemblyError(
            f"theme {theme_name!r}: {ASSEMBLE_KEY} names {target!r} but no parts survive"
        )
    if target in rest:
        raise ThemeAssemblyError(
            f"theme {theme_name!r}: {target!r} is both assembled from parts and set directly"
        )

    joined = ", ".join(value.strip().strip(",").strip() for _, value in sorted(parts))
    rest[target] = joined
    return replace(theme_data, custom_css_vars=rest)


def _parse_slots(spec: str, theme_name: str) -> Dict[str, Tuple[int, int]]:
    slots: Dict[str, Tuple[int, int]] = {}
    for token in spec.split():
        match = _SLOT_RE.match(token)
        if match is None:
            raise ThemeRandomizationError(
                f"theme {theme_name!r}: malformed slot {token!r} in {RANDOMIZED_KEY}; "
                f"expected name=lo:hi with integer bounds"
            )
        name, lo, hi = match.group(1), int(match.group(2)), int(match.group(3))
        if lo > hi:
            raise ThemeRandomizationError(
                f"theme {theme_name!r}: slot {name!r} has lo {lo} greater than hi {hi}"
            )
        if name in slots:
            raise ThemeRandomizationError(
                f"theme {theme_name!r}: slot {name!r} declared twice"
            )
        slots[name] = (lo, hi)
    if not slots:
        raise ThemeRandomizationError(
            f"theme {theme_name!r}: {RANDOMIZED_KEY} is present but declares no slots"
        )
    return slots


def _roll(key: str, lo: int, hi: int) -> int:
    cursor = _cursors.get(key)
    if cursor is None:
        cursor = _rng.random()
    cursor = (cursor + _PHI_CONJUGATE) % 1.0
    _cursors[key] = cursor
    return lo + int(cursor * (hi - lo + 1))


def apply_randomization(theme_data, theme_name: str = "<theme>", minimal: bool = False):
    spec = theme_data.custom_css_vars.get(RANDOMIZED_KEY)
    if not spec:
        return theme_data

    slots = _parse_slots(spec, theme_name)
    if minimal:
        override = theme_data.custom_css_vars.get(RANDOMIZED_KEY + "@minimal")
        if override:
            slots.update(_parse_slots(override, theme_name))
    values = {
        name: _roll(f"{theme_name}:{name}", lo, hi) for name, (lo, hi) in slots.items()
    }

    resolved = {}
    for key, value in theme_data.custom_css_vars.items():
        if key.startswith("$"):
            continue
        if "$" not in value:
            resolved[key] = value
            continue
        try:
            resolved[key] = Template(value).substitute(values)
        except KeyError as exc:
            raise ThemeRandomizationError(
                f"theme {theme_name!r}: {key} references undeclared slot {exc.args[0]!r}; "
                f"declared slots are {sorted(slots)}"
            ) from exc
        except ValueError as exc:
            raise ThemeRandomizationError(
                f"theme {theme_name!r}: {key} contains a malformed token "
                f"(use ${{name}}, and $$ for a literal dollar sign)"
            ) from exc

    return replace(theme_data, custom_css_vars=resolved)


def check_theme_randomization(theme_data, theme_name: str = "<theme>") -> None:
    spec = theme_data.custom_css_vars.get(RANDOMIZED_KEY)
    if not spec:
        for key, value in theme_data.custom_css_vars.items():
            if "$" in value.replace("$$", ""):
                raise ThemeRandomizationError(
                    f"theme {theme_name!r}: {key} uses a $token but the theme has no "
                    f"{RANDOMIZED_KEY} stub, so it will be emitted literally"
                )
        return

    slots = _parse_slots(spec, theme_name)
    override = theme_data.custom_css_vars.get(RANDOMIZED_KEY + "@minimal")
    if override:
        slots.update(_parse_slots(override, theme_name))
    used = set()
    for key, value in theme_data.custom_css_vars.items():
        if key.startswith(RANDOMIZED_KEY) or key == ASSEMBLE_KEY:
            continue
        identifiers = set(Template(value).get_identifiers())  # Python 3.11+
        unknown = identifiers - set(slots)
        if unknown:
            raise ThemeRandomizationError(
                f"theme {theme_name!r}: {key} references undeclared slot(s) {sorted(unknown)}"
            )
        used |= identifiers
    unused = set(slots) - used
    if unused:
        raise ThemeRandomizationError(
            f"theme {theme_name!r}: slot(s) {sorted(unused)} declared but never referenced"
        )


def preview_assembled(
    theme_data, minimal: bool = False, theme_name: str = "<theme>"
) -> str:
    data = apply_randomization(
        assemble_parts(theme_data, minimal=minimal, theme_name=theme_name),
        theme_name,
        minimal=minimal,
    )
    return data.custom_css_vars[theme_data.custom_css_vars[ASSEMBLE_KEY].strip()]
