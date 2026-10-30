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
    "appalachian": ThemeConfig(
        bg_primary="#f0f8ff",
        bg_surface="#2e8b57",
        bg_alt="#b0e0e6",
        bg_hover="#5f9ea0",
        text_main="#2f4f4f",
        text_heading="#fff",
        text_hover="#0b2e1c",
        border_main="#2e8b57",
        border_alt="#87cefa",
        bg_action="#1e90ff",
        text_on_action="#fff",
        bg_action_hover="#00bfff",
        bg_error="#d32f2f",
        text_on_error="#fff",
        bg_success="#388e3c",
        text_on_success="#fff",
        font_family="Georgia,serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-special-background": "linear-gradient(160deg,#5f9ea0,#52898b,#457375)",
            "--dgcv-table-shadow": "0 0 12px #2e8b5759",
            "--dgcv-text-shadow": "0 1px 0 #ffffff80",
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
        text_on_action="#fff",
        bg_action_hover="#9d5fff",
        bg_error="#ff4b2b",
        text_on_error="#fff",
        bg_success="#00e676",
        text_on_success="#0a0e1a",
        font_family="Georgia,serif",
        custom_css_vars={
            "--dgcv-special-background": "linear-gradient(170deg,#0a0e1a,#0d3320 35%,#0a1a3a 65%,#1a0a2e)",
            "--dgcv-table-shadow": "0 0 20px #00b4a066,0 0 60px #7b2fff33",
            "--dgcv-text-shadow": "0 0 8px #00e5cc99",
        },
    ),
    "ballpoint": ThemeConfig(
        bg_primary="#fbfbf7",
        bg_surface="#dbe7f5",
        bg_alt="#f3f4ee",
        bg_hover="#fff3a0",
        text_main="#1f3a8f",
        text_heading="#1f3a8f",
        text_hover="#1f3a8f",
        border_main="#2a4bb5",
        border_alt="#c5d3e8",
        bg_action="#1f3a8f",
        text_on_action="#fbfbf7",
        bg_action_hover="#2a4bb5",
        bg_error="#b91c1c",
        text_on_error="#fbfbf7",
        bg_success="#2e7d4f",
        text_on_success="#fbfbf7",
        font_family="Bradley Hand,Segoe Print,Comic Sans MS,cursive",
        custom_css_vars={
            "--dgcv-special-background": "linear-gradient(90deg,#0000 9px,#e57373 9px 10.5px,#0000 10.5px),repeating-linear-gradient(#0000 0 23px,#b9cfe8 23px 24px),#fbfbf7",
            "--dgcv-border-radius": "3px",
            "--dgcv-table-shadow": "0 1px 3px #1f3a8f26",
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
        border_alt="#414868",
        bg_action="#3d59a1",
        text_on_action="#fff",
        bg_action_hover="#7dcfff",
        bg_error="#f7768e",
        text_on_error="#24283b",
        bg_success="#9ece6a",
        text_on_success="#24283b",
        font_family="Georgia,serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-table-shadow": "0 0 10px #7dcfff80",
        },
    ),
    "blue_gray": ThemeConfig(
        bg_primary="#2e3440",
        bg_surface="#2e3440",
        bg_alt="#3b4252",
        bg_hover="#3d4658",
        text_main="#d8dee9",
        text_heading="#eceff4",
        text_hover="#8fbcbb",
        border_main="#4c566a",
        border_alt="#434c5e",
        bg_action="#5e81ac",
        text_on_action="#eceff4",
        bg_action_hover="#81a1c1",
        bg_error="#bf616a",
        text_on_error="#eceff4",
        bg_success="#a3be8c",
        text_on_success="#2e3440",
        font_family="'Inter',sans-serif",
    ),
    "blueprint": ThemeConfig(
        bg_primary="#002b4f",
        bg_surface="#036",
        bg_alt="#036",
        bg_hover="#369",
        text_main="#fff",
        text_heading="#fff",
        text_hover="#fff",
        border_main="#ccc",
        border_alt="#e6e6e6",
        bg_action="#4da6ff",
        text_on_action="#002b4f",
        bg_action_hover="#80bfff",
        bg_error="#ff4d4d",
        text_on_error="#fff",
        bg_success="#4dff4d",
        text_on_success="#002b4f",
        font_family="Roboto Mono,monospace",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-special-background": "repeating-linear-gradient(0deg,#0000 0 19px,#ffffff26 19px 20px),repeating-linear-gradient(90deg,#0000 0 19px,#ffffff26 19px 20px),#036",
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
        text_on_error="#fff",
        bg_success="#2e7d32",
        text_on_success="#fff",
        font_family="'Courier New',monospace",
        custom_css_vars={
            "--dgcv-special-background": """url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg'><filter id='b'><feTurbulence type='fractalNoise' baseFrequency='.008 .8' numOctaves='3' seed='2'/><feColorMatrix values='0 0 0 0 .55 0 0 0 0 .58 0 0 0 0 .6 0 0 0 .9 -.25'/></filter><rect width='100%' height='100%' filter='url(%23b)'/></svg>"),linear-gradient(180deg,#d4d9dd,#9aa4ad 40%,#b8bfc5 60%,#cfd4d8 80%,#a8b0b8)""",
            "--dgcv-table-shadow": "0 2px 8px #0006,inset 0 1px 0 #ffffff4c",
            "--dgcv-text-shadow": "0 1px 0 #fff6",
        },
    ),
    "chalkboard_green": ThemeConfig(
        bg_primary="#1a2420",
        bg_surface="#212f29",
        bg_alt="#1d2925",
        bg_hover="#f2d675",
        text_main="#e9e6da",
        text_heading="#f6de8c",
        text_hover="#1a2420",
        text_alt="#c9d2c6",
        border_main="#9c7a50",
        border_alt="#634b32",
        bg_action="#e8748f",
        text_on_action="#241016",
        bg_action_hover="#f196ac",
        bg_error="#d9605a",
        text_on_error="#230a08",
        bg_success="#7fb06b",
        text_on_success="#0e1d0a",
        font_family="'Chalkboard SE','Comic Sans MS','Marker Felt',cursive",
        custom_css_vars={
            "$randomized": "axdx=12:86 axdy=14:78 axsz=56:92 cndx=12:86 cndy=14:78 cnsz=40:64",
            "--dgcv-special-background": """url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg'><filter id='d'><feTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='2'/><feColorMatrix values='0 0 0 0 .96 0 0 0 0 .95 0 0 0 0 .9 .26 0 0 0 -.14'/></filter><rect width='100%' height='100%' filter='url(%23d)'/></svg>"),url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='-45 -45 90 90'><path d='M0 0v-38m3 6L0-38l-3 6M0 0l34-6m-6 4L34-6l-7-2M0 0l-26 18m3-6L-26 18l7-1' stroke='%23fff6' stroke-width='2.2' fill='none'/></svg>") ${axdx}% ${axdy}%/auto min(${axsz}px,60%) no-repeat,url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='-20 -26 40 52'><path d='M-13 4L0-24l13 28a13 4 0 1 0-26 0a13 4 0 1 0 26 0' stroke='%23fff6' stroke-width='2.2' fill='none'/></svg>") ${cndx}% ${cndy}%/auto min(${cnsz}px,60%) no-repeat,radial-gradient(40% 30% at ${axdx}% ${axdy}%,#fff1,#0000),radial-gradient(40% 30% at ${cndx}% ${cndy}%,#fff1,#0000),linear-gradient(#26342d,#1e2a24,#161f1a)""",
            "--dgcv-table-shadow": "0 2px 10px #04080673",
        },
    ),
    "clownfish_reef": ThemeConfig(
        bg_primary="#0a1e2b",
        bg_surface="#123044",
        bg_alt="#0d2433",
        bg_hover="#f5714c",
        text_main="#dce9f0",
        text_heading="#ffa47f",
        text_hover="#0a1e2b",
        text_alt="#a8c4d2",
        border_main="#4380a0",
        border_alt="#1f4860",
        bg_action="#ec5f34",
        text_on_action="#160800",
        bg_action_hover="#ff7e52",
        bg_error="#e0596e",
        text_on_error="#1a0409",
        bg_success="#2c9c7a",
        text_on_success="#04180f",
        font_family="'Avenir Next', Avenir, 'Century Gothic', 'Trebuchet MS', sans-serif",
        custom_css_vars={
            "$randomized": "nx=0:199 ny=0:199 ns=180:240 hx=0:70 hy=0:12 hs=90:150 mx=0:52 my=0:30 ms=60:95 mp=0:3 bx=4:96 by=0:99 bw=46:74 as=60:100 sx=0:100",
            "$assemble": "--dgcv-special-background",
            "$part_1": """url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='200' height='200'><filter id='f'><feTurbulence type='fractalNoise' baseFrequency='.012 .07' numOctaves='2' seed='9' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 .82 0 0 0 0 .95 0 0 0 0 1 1.3 0 0 0 -.85'/></filter><rect width='200' height='200' filter='url(%23f)'/></svg>") ${nx}px ${ny}px/${ns}px repeat, url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='-59 -40 113 78' stroke='%23152430' stroke-width='2'><path d='M-31-7C-41-12-49-15-55-16C-57-5-55 17-50 27C-45 25-38 20-29 13ZM21-26C10-35-6-38-16-35C-23-32-27-22-29-13ZM10 23C5 33-4 36-10 34C-17 33-21 25-24 20Z' fill='%23e0561f'/><path d='M52-4C38-22 13-30-7-26C-19-24-27-17-31-7L-29 13C-23 23-13 28-1 27C19 26 42 15 52-4Z' fill='%23f4702f'/><path d='M20-25C24-24 28-22 32-20L33 16C29 18 25 21 21 22ZM-2-27C2-27 7-27 11-27L14 25C10 26 5 27 1 27ZM-29-11C-27-15-25-17-22-20L-17 24C-20 22-23 20-26 17Z' fill='%23fbf1e6'/><path d='M19 4C12 15 10 23 15 25C21 24 26 13 25 5Z' fill='%23f98a42'/><circle cx='36' cy='-11' r='5.2' fill='%23f7eadf' stroke-width='1.4'/><circle cx='36' cy='-11' r='2.4' fill='%23152430'/><circle cx='34' cy='-12' r='1.1' fill='%23fff' stroke='none'/></svg>") calc(${hx}% + 250px) ${hy}%/min(${hs}px,16%) no-repeat""",
            "$part_2@detail": """url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='-54 -41 113 78' stroke='%23152430' stroke-width='2'><path d='M29-13C38-19 45-24 51-27C55-16 57 6 55 17C49 16 41 12 31 7ZM-26-22C-17-33-1-38 9-37C17-36 23-27 26-18ZM-6 25C1 33 11 34 17 32C23 29 26 21 27 15Z' fill='%23e0561f'/><path d='M-52 6C-41-14-18-26 2-27C14-27 23-22 29-13L31 7C27 18 18 25 6 27C-13 29-38 22-52 6Z' fill='%23f4702f'/><path d='M-24-21C-28-19-32-16-36-13L-29 22C-25 24-20 25-16 26ZM-3-27C-7-26-12-25-16-24L-9 27C-4 27 0 27 4 27ZM26-17C24-19 21-22 18-24L21 20C24 18 27 15 29 12Z' fill='%23fbf1e6'/><path d='M-17 8C-9 17-6 25-10 27C-16 27-23 18-23 10Z' fill='%23f98a42'/><circle cx='-37' cy='-4' r='5.2' fill='%23f7eadf' stroke-width='1.4'/><circle cx='-37' cy='-4' r='2.4' fill='%23152430'/><circle cx='-36' cy='-5' r='1.1' fill='%23fff' stroke='none'/></svg>") calc(${mx}% + 300px + max(0,${mp} - 2)*9999px) ${my}%/min(${ms}px,12%) no-repeat""",
            "$part_3": """url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 60 260' fill='none' stroke='%23cfeff5' stroke-width='1.1' opacity='.4'><circle cx='30' cy='22' r='2.4'/><circle cx='46' cy='58' r='4.9'/><circle cx='37' cy='94' r='3.9'/><circle cx='17' cy='130' r='2.9'/><circle cx='18' cy='166' r='5.4'/><circle cx='37' cy='202' r='4.4'/><circle cx='46' cy='238' r='3.4'/></svg>") ${bx}% ${by}%/${bw}px repeat-y, url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 120 60' fill='none' stroke-linecap='round'><path id='t' d='M18 52Q14.8 36.1 1.9 23.1M25 52Q22.2 33.7 10.8 18.8M32 52Q30 32.8 21.9 17.1M39 52Q37.6 33.1 32 17.7M46 52Q45 31.9 40.9 15.4M53 52Q52 30.5 48.2 12.9M60 52Q60.4 28 62 8.5M67 52Q67.1 30.3 67.7 12.6M74 52Q75.5 29.9 81.4 11.9M81 52Q82.3 31.4 87.4 14.5M88 52Q89.3 31.4 94.6 14.5M95 52Q97.1 33 105.4 17.5M102 52Q104.3 36.7 113.7 24.1' stroke='%239c6580' stroke-width='5'/><use href='%23t' stroke='%23d693a8' stroke-width='2'/><ellipse cx='60' cy='55' rx='50' ry='8' fill='%2346304a' stroke='none'/></svg>") 100% 100%/auto min(${as}px,32%) no-repeat, linear-gradient(#0000 20%,#0f3e57 55%,#071d2a), linear-gradient(104deg,#0000 7%,#cfeff738 8% 11%,#0000 12% 21%,#cfeff724 22% 23%,#0000 24% 32%,#cfeff738 33% 36%,#0000 37% 57%,#cfeff738 58% 61%,#0000 62% 71%,#cfeff724 72% 73%,#0000 74% 82%,#cfeff738 83% 86%,#0000 87%) ${sx}% 0/200% 100%, linear-gradient(#1e6c8e,#175978 24%,#0f3e57 52%,#0a2c3f 78%,#071d2a), #0f3e57""",
            "--dgcv-special-text": "#ffd2bc",
            "--plaque-fill": "#0e2637",
            "--plaque-border": "#ffa47f",
            "--dgcv-table-shadow": "0 2px 14px #020c148c",
            "--dgcv-text-shadow": "0 0 3px #0a1e2b,0 0 2px #0a1e2b",
        },
    ),
    "coffee_shop": ThemeConfig(
        bg_primary="#fffaf0",
        bg_surface="#3e2723",
        bg_alt="#f5f5dc",
        bg_hover="#deb887",
        text_main="#3e2723",
        text_heading="#fff",
        text_hover="#3e2723",
        border_main="#8b4513",
        border_alt="#eaddc5",
        bg_action="#5d4037",
        text_on_action="#fff",
        bg_action_hover="#8d6e63",
        bg_error="#b71c1c",
        text_on_error="#fff",
        bg_success="#33691e",
        text_on_success="#fff",
        font_family="Georgia,serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-border-radius": "8px",
            "--dgcv-table-shadow": "0 0 10px #3e272333",
        },
    ),
    "dark_high_contrast": ThemeConfig(
        bg_primary="#1e1e1e",
        bg_surface="#000",
        bg_alt="#333",
        bg_hover="#4d4d4d",
        text_main="#fff",
        text_heading="#fff",
        text_hover="#fff",
        border_main="#fff",
        border_alt="#666",
        bg_action="#fff",
        text_on_action="#000",
        bg_action_hover="#ccc",
        bg_error="#f00",
        text_on_error="#fff",
        bg_success="#0f0",
        text_on_success="#000",
    ),
    "dark_modern": ThemeConfig(
        bg_primary="#1c1c1c",
        bg_surface="#2c2c2c",
        bg_alt="#3a3a3a",
        bg_hover="#484848",
        text_main="#f5f5f5",
        text_heading="#f5f5f5",
        text_hover="#7fdbff",
        border_main="#444",
        border_alt="#555",
        bg_action="#0074d9",
        text_on_action="#fff",
        bg_action_hover="#39cccc",
        bg_error="#ff4136",
        text_on_error="#fff",
        bg_success="#2ecc40",
        text_on_success="#1c1c1c",
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
        font_family="Georgia,serif",
        custom_css_vars={
            "--dgcv-special-text": "#e4f2f8",
            "--dgcv-special-background": "linear-gradient(135deg,#03101c,#06243f 18%,#0c4a78 36%,#137060 52%,#1f9460 66%,#2eb0a0 78%,#cdeef0 90%,#03101c)",
            "--dgcv-table-shadow": "0 0 20px #1c6aa073,0 0 60px #2ea86e38,inset 0 1px 0 #e4f2f81f",
            "--dgcv-text-shadow": "0 1px 2px #000c,0 0 6px #0008",
            "--dgcv-border-image": "linear-gradient(135deg,#1c6aa0,#1f9460,#2eb0a0,#4cc8b4,#2eb0a0,#1f9460,#1c6aa0) 1",
            "--dgcv-border-radius": "0px",
        },
    ),
    "fire_gradient": ThemeConfig(
        bg_primary="#0f0700",
        bg_surface="#1a0c00",
        bg_alt="#120900",
        bg_hover="#f60",
        text_main="#f5ddc8",
        text_heading="#fa5",
        text_hover="#0f0700",
        text_alt="#e0c0a0",
        border_main="#a30",
        border_alt="#2a1200",
        bg_action="#a30",
        text_on_action="#f5ddc8",
        bg_action_hover="#f60",
        bg_error="#f20",
        text_on_error="#0f0700",
        bg_success="#364",
        text_on_success="#f5ddc8",
        font_family="Georgia,serif",
        custom_css_vars={
            "--dgcv-special-background": "linear-gradient(135deg,#0f0700,#2a0a00 12%,#4a1500 24%,#6a2200 35%,#a30 48%,#c50 58%,#f60 68%,#fa0 78%,#fc4 86%,#0f0700)",
            "--dgcv-table-shadow": "0 0 20px #a309,0 0 60px #f603,inset 0 1px 0 #ffc8641f",
            "--plaque-fill": "#1a0c00",
            "--plaque-border": "#fa5",
            "--dgcv-text-shadow": "0 0 12px #ff780099",
            "--dgcv-border-image": "linear-gradient(135deg,#f20,#f60,#fa0,#fc4,#f60,#f20) 1",
            "--dgcv-border-radius": "0px",
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
        border_alt="#40916c",
        bg_action="#40916c",
        text_on_action="#fff",
        bg_action_hover="#52b788",
        bg_error="#d90429",
        text_on_error="#fff",
        bg_success="#74c69d",
        text_on_success="#081c15",
        font_family="Georgia,serif",
        custom_css_vars={
            "--dgcv-special-background": "linear-gradient(170deg,#1b4332,#2d6a4f,#1b4332)",
            "--dgcv-table-shadow": "0 0 16px #1b433299",
            "--dgcv-text-shadow": "0 1px 2px #0006",
        },
    ),
    "graph_paper": ThemeConfig(
        bg_primary="#f2faff",
        bg_surface="#fff",
        bg_alt="#fff",
        bg_hover="#e6f7ff",
        text_main="#000",
        text_heading="#000",
        text_hover="#000",
        border_main="#ccc",
        border_alt="#e6e6e6",
        bg_action="#06c",
        text_on_action="#fff",
        bg_action_hover="#3385ff",
        bg_error="#c00",
        text_on_error="#fff",
        bg_success="#090",
        text_on_success="#fff",
        font_family="Roboto,sans-serif",
        custom_css_vars={
            "--dgcv-special-background": "repeating-linear-gradient(0deg,#0000 0 19px,#c5e0f5 19px 20px),repeating-linear-gradient(90deg,#0000 0 19px,#c5e0f5 19px 20px),#f2faff",
        },
    ),
    "gruvbox_dark": ThemeConfig(
        bg_primary="#282828",
        bg_surface="#1d2021",
        bg_alt="#32302f",
        bg_hover="#3c3836",
        text_main="#ebdbb2",
        text_heading="#fabd2f",
        text_hover="#8ec07c",
        text_alt="#83a598",
        border_main="#504945",
        border_alt="#665c54",
        bg_action="#458588",
        text_on_action="#ebdbb2",
        bg_action_hover="#83a598",
        bg_error="#cc241d",
        text_on_error="#ebdbb2",
        bg_success="#98971a",
        text_on_success="#1d2021",
        font_family="monospace",
    ),
    "gruvbox_light": ThemeConfig(
        bg_primary="#fbf1c7",
        bg_surface="#ebdbb2",
        bg_alt="#f2e5bc",
        bg_hover="#e3d4b9",
        text_main="#3c3836",
        text_heading="#9d0006",
        text_hover="#076678",
        border_main="#d5c4a1",
        border_alt="#e8d8b0",
        bg_action="#076678",
        text_on_action="#fbf1c7",
        bg_action_hover="#458588",
        bg_error="#9d0006",
        text_on_error="#fbf1c7",
        bg_success="#79740e",
        text_on_success="#fbf1c7",
        font_family="Georgia,serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-table-shadow": "0 0 10px #0000000d",
        },
    ),
    "kagome": ThemeConfig(
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
        text_on_action="#fff",
        bg_action_hover="#9c6422",
        bg_error="#b23a2a",
        text_on_error="#fff",
        bg_success="#5f7a2e",
        text_on_success="#fff",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-special-text": "#36280f",
            "--dgcv-special-background": """url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='16.17' height='28'><clipPath id='c'><path id='s' d='M-9 4.5h34v5h-34zm0 14h34v5h-34z'/></clipPath><g id='a' fill='%23d97c4a' transform='rotate(120 8.1 14)'><use href='%23s'/></g><use href='%23s' fill='%239ba858' transform='rotate(60 8.1 14)'/><use href='%23s' fill='%23ddb13f'/><use href='%23a' clip-path='url(%23c)'/></svg>"),#f5edda""",
            "--plaque-fill": "#efe6cf",
            "--plaque-border": "#ddb13f",
            "--dgcv-table-shadow": "0 1px 4px #36280f29,inset 0 1px 0 #fff8",
        },
    ),
    "kagome_dark": ThemeConfig(
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
        text_on_action="#fff",
        bg_action_hover="#7e5aa6",
        bg_error="#b53a48",
        text_on_error="#fff",
        bg_success="#2c8059",
        text_on_success="#fff",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-special-text": "#f4ead9",
            "--dgcv-special-background": """url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='16.17' height='28'><clipPath id='c'><path id='s' d='M-9 4.5h34v5h-34zm0 14h34v5h-34z'/></clipPath><g id='a' fill='%238f6a29' transform='rotate(120 8.1 14)'><use href='%23s'/></g><use href='%23s' fill='%231c7068' transform='rotate(60 8.1 14)'/><use href='%23s' fill='%23a23d54'/><use href='%23a' clip-path='url(%23c)'/></svg>"),#201a28""",
            "--plaque-fill": "#2a2336",
            "--plaque-border": "#a23d54",
            "--dgcv-table-shadow": "0 1px 6px #0007,inset 0 1px 0 #fff1",
        },
    ),
    "kagome_green": ThemeConfig(
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
        text_on_action="#fff",
        bg_action_hover="#287d76",
        bg_error="#c8463f",
        text_on_error="#fff",
        bg_success="#237a4c",
        text_on_success="#fff",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-special-text": "#0f2d2f",
            "--dgcv-special-background": """url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='16.17' height='28'><clipPath id='c'><path id='s' d='M-9 4.5h34v5h-34zm0 14h34v5h-34z'/></clipPath><g id='a' fill='%239aa6e4' transform='rotate(120 8.1 14)'><use href='%23s'/></g><use href='%23s' fill='%23f6977a' transform='rotate(60 8.1 14)'/><use href='%23s' fill='%234fb1a8'/><use href='%23a' clip-path='url(%23c)'/></svg>"),#eef5f4""",
            "--plaque-fill": "#e2efed",
            "--plaque-border": "#4fb1a8",
            "--dgcv-table-shadow": "0 1px 4px #14363829,inset 0 1px 0 #fff8",
        },
    ),
    "kagome_light": ThemeConfig(
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
        text_on_action="#fff",
        bg_action_hover="#8c6ab2",
        bg_error="#c84a5a",
        text_on_error="#fff",
        bg_success="#2a8055",
        text_on_success="#fff",
        font_family="Georgia, serif",
        custom_css_vars={
            "--dgcv-special-text": "#3a2e44",
            "--dgcv-special-background": """url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='16.17' height='28'><clipPath id='c'><path id='s' d='M-9 4.5h34v5h-34zm0 14h34v5h-34z'/></clipPath><g id='a' fill='%23b0bce8' transform='rotate(120 8.1 14)'><use href='%23s'/></g><use href='%23s' fill='%23a8d8b8' transform='rotate(60 8.1 14)'/><use href='%23s' fill='%23f4b89a'/><use href='%23a' clip-path='url(%23c)'/></svg>"),#f6f1ea""",
            "--plaque-fill": "#ece8f2",
            "--plaque-border": "#f4b89a",
            "--dgcv-table-shadow": "0 1px 4px #6a4a8a24,inset 0 1px 0 #fff8",
        },
    ),
    "koifish_pond": ThemeConfig(
        bg_primary="#0e1f1c",
        bg_surface="#162c27",
        bg_alt="#122622",
        bg_hover="#e8a93c",
        text_main="#dce8df",
        text_heading="#f0b75a",
        text_hover="#0e1f1c",
        text_alt="#b4c9ba",
        border_main="#63947b",
        border_alt="#284840",
        bg_action="#e2652a",
        text_on_action="#1a0703",
        bg_action_hover="#f0834a",
        bg_error="#de5a46",
        text_on_error="#1a0604",
        bg_success="#63a85c",
        text_on_success="#04140a",
        font_family="'Iowan Old Style','Palatino Linotype',Palatino,Georgia,serif",
        custom_css_vars={
            "$randomized": "rx=0:239 ry=0:239 rs=200:280 gx=4:38 gy=0:40 gs=80:120 k1x=48:92 k1y=0:30 k1s=100:150 k2x=30:70 k2y=0:50 k2s=84:130 sx=40:60 sy=40:60",
            "$assemble": "--dgcv-special-background",
            "$part_1": """url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='240' height='240'><filter id='r' x='0' y='0' width='100%' height='100%'><feTurbulence type='fractalNoise' baseFrequency='.0165 .021' numOctaves='4' seed='13' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 .78 0 0 0 0 .93 0 0 0 0 .86 1.05 0 0 0 -.72'/></filter><rect width='240' height='240' filter='url(%23r)'/></svg>") ${rx}px ${ry}px/${rs}px repeat, url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='-50 -40 100 80' stroke-width='1.5'><defs><path id='p' d='M0 0 28.7-8.8A30 30 0 1 0 28.7 8.8Z'/><path id='v' d='M20 17 0 0 1 26M-18 19 0 0-26 0M-18-19 0 0 1-26M0 0 20-17' fill='none' opacity='.45'/></defs><g transform='translate(-18-8)rotate(24)' fill='%234e8a3e' stroke='%2324491f'><use href='%23p'/><use href='%23v' stroke='%237fb05e'/></g><g transform='translate(24 16)rotate(128)scale(.73)' fill='%238a5230' stroke='%23402314'><use href='%23p'/><use href='%23v' stroke='%23b87a48'/></g></svg>") calc(${gx}% + max(0px,260px - 100%)*99) ${gy}%/min(${gs}px,44%) no-repeat, url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='-58 -58 116 116'><g transform='rotate(-18)'><path d='M4 26c5 10 11 20 9 26-6-2-11-9-13-15-2 6-7 13-13 15-2-6 4-16 9-26ZM-11-18c-7 4-12 12-10 17 5-4 9-10 11-14ZM11-18c7 4 12 12 10 17-5-4-9-10-11-14Z' fill='%23d4541c' opacity='.75'/><path d='M0-44c7 4 12 17 12 34 0 16-4 32-8 40h-8c-4-8-8-24-8-40 0-17 5-30 12-34Z' fill='%23ec7a2e'/><ellipse cx='-3' cy='-24' rx='5' ry='9' transform='rotate(20-3-24)' fill='%23ffd9a8' opacity='.55'/><ellipse cx='3' cy='6' rx='6' ry='11' transform='rotate(-14 3 6)' fill='%23b02f0a' opacity='.5'/><g fill='%231f0a03' opacity='.75'><circle cx='-5' cy='-35' r='1.7'/><circle cx='5' cy='-35' r='1.7'/></g></g></svg>") calc(${k1x}% + max(0px,260px - 100%)*99) ${k1y}%/min(${k1s}px,42%) no-repeat""",
            "$part_2@detail": """url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='-58 -58 116 116'><g transform='rotate(152)'><path d='M4 26c5 10 11 20 9 26-6-2-11-9-13-15-2 6-7 13-13 15-2-6 4-16 9-26ZM-11-18c-7 4-12 12-10 17 5-4 9-10 11-14ZM11-18c7 4 12 12 10 17-5-4-9-10-11-14Z' fill='%23e2dacb' opacity='.75'/><path d='M0-44c7 4 12 17 12 34 0 16-4 32-8 40h-8c-4-8-8-24-8-40 0-17 5-30 12-34Z' fill='%23f6f0e2'/><g fill='%23d8402a' opacity='.9'><ellipse cx='-2' cy='-26' rx='6' ry='9' transform='rotate(16-2-26)'/><ellipse cx='3' cy='2' rx='7' ry='11' transform='rotate(-12 3 2)'/><ellipse cx='-3' cy='22' rx='3.5' ry='5'/></g><g fill='%23241009' opacity='.75'><circle cx='-5' cy='-35' r='1.7'/><circle cx='5' cy='-35' r='1.7'/></g></g></svg>") calc(${k2x}% + max(0px,260px - 100%)*99) ${k2y}%/min(${k2s}px,38%) no-repeat""",
            "$part_3": "radial-gradient(at ${sx}% ${sy}%,#9fd8b030,#0000 55%),linear-gradient(#16302a,#102420 45%,#081613)",
            "--plaque-fill": "#162c27",
            "--plaque-border": "#f0b75a",
            "--dgcv-table-shadow": "0 2px 14px #030a088c",
        },
    ),
    "lunar": ThemeConfig(
        bg_primary="#dfe7ec",
        bg_surface="#394b59",
        bg_alt="#f5f8fa",
        bg_hover="#cbd6e2",
        text_main="#333",
        text_heading="#b0c4de",
        text_hover="#333",
        border_main="#394b59",
        border_alt="#e1e8ed",
        bg_action="#5c7080",
        text_on_action="#fff",
        bg_action_hover="#738694",
        bg_error="#db3737",
        text_on_error="#fff",
        bg_success="#0f9960",
        text_on_success="#fff",
    ),
    "matte_amber": ThemeConfig(
        bg_primary="#0f0c07",
        bg_surface="#181208",
        bg_alt="#120e06",
        bg_hover="#c82",
        text_main="#f0e4c8",
        text_heading="#e8c070",
        text_hover="#0f0c07",
        text_alt="#d8cc99",
        border_main="#7a5510",
        border_alt="#251a08",
        bg_action="#7a5510",
        text_on_action="#f0e4c8",
        bg_action_hover="#c82",
        bg_error="#c32",
        text_on_error="#f0e4c8",
        bg_success="#364",
        text_on_success="#f0e4c8",
        font_family="Georgia,serif",
        custom_css_vars={
            "--dgcv-special-background": "linear-gradient(135deg,#0f0c07,#201508,#0f0c07)",
            "--dgcv-table-shadow": "inset 0 1px 0 #f0c8640d",
        },
    ),
    "matte_green": ThemeConfig(
        bg_primary="#070f0a",
        bg_surface="#0a1810",
        bg_alt="#081209",
        bg_hover="#2a5",
        text_main="#d0eedd",
        text_heading="#6c8",
        text_hover="#070f0a",
        text_alt="#a8d8bb",
        border_main="#1a5530",
        border_alt="#102018",
        bg_action="#1a5530",
        text_on_action="#d0eedd",
        bg_action_hover="#2a5",
        bg_error="#c32",
        text_on_error="#d0eedd",
        bg_success="#2a5",
        text_on_success="#070f0a",
        font_family="Georgia,serif",
        custom_css_vars={
            "--dgcv-special-background": "linear-gradient(135deg,#070f0a,#0c2015,#070f0a)",
            "--dgcv-table-shadow": "inset 0 1px 0 #64dc8c0d",
        },
    ),
    "matte_slate_soft": ThemeConfig(
        bg_primary="#111214",
        bg_surface="#18191d",
        bg_alt="#141518",
        bg_hover="#5c7eb1",
        text_main="#c8cdd6",
        text_heading="#c8cdd6",
        text_hover="#111214",
        text_alt="#8892a0",
        border_main="#2a2d34",
        border_alt="#1e2026",
        bg_action="#2a2d34",
        text_on_action="#c8cdd6",
        bg_action_hover="#57a",
        bg_error="#c34",
        text_on_error="#c8cdd6",
        bg_success="#2a6644",
        text_on_success="#c8cdd6",
        font_family="Georgia,serif",
        custom_css_vars={
            "--dgcv-special-background": "linear-gradient(135deg,#18191d,#1c1e24,#18191d)",
            "--dgcv-table-shadow": "inset 0 1px 0 #96b4dc0a",
        },
    ),
    "matte_violet": ThemeConfig(
        bg_primary="#0a0a0f",
        bg_surface="#100e18",
        bg_alt="#0d0b14",
        bg_hover="#96f",
        text_main="#e0d8f0",
        text_heading="#c8b8f0",
        text_hover="#0a0a0f",
        text_alt="#c0b0e8",
        border_main="#53a",
        border_alt="#1a1428",
        bg_action="#53a",
        text_on_action="#fff",
        bg_action_hover="#74c",
        bg_error="#c24",
        text_on_error="#fff",
        bg_success="#4a7",
        text_on_success="#0a0a0f",
        font_family="Georgia,serif",
        custom_css_vars={
            "--dgcv-special-background": "linear-gradient(135deg,#0a0a0f,#120a20 40%,#1a1030 70%,#0a0a0f)",
            "--dgcv-table-shadow": "inset 0 1px 0 #b48cff0d",
        },
    ),
    "meadow_day": ThemeConfig(
        bg_primary="#fff",
        bg_surface="#ece9d8",
        bg_alt="#f2f0e4",
        bg_hover="#316ac5",
        text_main="#000",
        text_heading="#0a246a",
        text_hover="#fff",
        text_alt="#171713",
        border_main="#63819f",
        border_alt="#c8c4b4",
        bg_action="#245edc",
        text_on_action="#fff",
        bg_action_hover="#0b3fa8",
        bg_error="#b0362b",
        text_on_error="#fff",
        bg_success="#3b7a28",
        text_on_success="#fff",
        font_family="Tahoma,Verdana,sans-serif",
        custom_css_vars={
            "--dgcv-special-background": """url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 800 500'><linearGradient id='t' x2='0' y2='1'><stop stop-color='%23b3ce73'/><stop offset='.15' stop-color='%238cb656'/><stop offset='1' stop-color='%233c5a28'/></linearGradient><linearGradient id='n' x2='0' y2='1'><stop stop-color='%2360843a'/><stop offset='1' stop-color='%23344c22'/></linearGradient><linearGradient id='f' x2='0' y2='1'><stop offset='.7' stop-color='%23fff'/><stop offset='1' stop-color='%23fff' stop-opacity='0'/></linearGradient><filter id='c'><feTurbulence type='fractalNoise' baseFrequency='.0055 .019' numOctaves='5' seed='3'/><feColorMatrix values='0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 -2.9 0 0 0 1.62'/><feComposite in2='SourceGraphic' operator='in'/></filter><filter id='m'><feTurbulence type='fractalNoise' baseFrequency='.014 .045' numOctaves='4'/><feColorMatrix values='0 0 0 0 .03 0 0 0 0 .07 0 0 0 0 .01 .7 0 0 0 -.2'/><feComposite in2='SourceGraphic' operator='in'/><feBlend in2='SourceGraphic'/></filter><rect width='800' height='340' fill='url(%23f)' filter='url(%23c)'/><path d='M-4 352C88 338 200 330 300 330S560 336 804 349V504H-4Z' fill='%2397ae7e'/><path d='M-4 318C52 296 122 278 212 277S420 300 520 320S720 360 804 378V504H-4Z' fill='url(%23t)' stroke='%23cfe19a' stroke-width='2' filter='url(%23m)'/><path d='M-4 463C130 443 280 417 400 395S690 344 804 320V504H-4Z' fill='url(%23n)' stroke='%2392b05f' stroke-width='2' filter='url(%23m)'/></svg>") 50% 58%/cover no-repeat,linear-gradient(#3a80c8,#5f9fd6 40%,#dcebf5),#5f9fd6""",
            "--plaque-fill": "#ece9d8",
            "--plaque-border": "#0a246a",
            "--dgcv-table-shadow": "0 2px 6px #0a246a47",
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
        font_family="Tahoma,'Trebuchet MS',Verdana,Geneva,sans-serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "--dgcv-special-background": """url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 800 500'><linearGradient id='c' x2='0' y2='1'><stop offset='.5' stop-color='%236a4660'/><stop offset='1' stop-color='%23f0a46a'/></linearGradient><linearGradient id='t' x2='0' y2='1'><stop stop-color='%234e4c2c'/><stop offset='.35' stop-color='%2330351e'/><stop offset='1' stop-color='%231a2210'/></linearGradient><filter id='f'><feTurbulence type='fractalNoise' baseFrequency='.0055 .019' numOctaves='5' seed='3'/><feColorMatrix values='1.2 0 0 0 .55 1.2 0 0 0 .55 1.2 0 0 0 .55 -2.9 0 0 0 1.62' result='m'/><feComposite in='SourceGraphic' in2='m' operator='arithmetic' k1='1'/></filter><rect width='800' height='352' fill='url(%23c)' filter='url(%23f)'/><path d='M-4 318C52 296 122 278 212 277 332 276 420 300 520 320 630 342 720 360 804 378V504H-4z' fill='url(%23t)' stroke='%23ffce928c' stroke-width='2'/><path d='M-4 463C130 443 280 417 400 395 530 371 690 344 804 320V504H-4z' fill='%232c2c18' stroke='%23f7bc7c73' stroke-width='2'/></svg>") center 58%/cover,radial-gradient(50% 60% at 40% 100%,#f8a86880,#0000),linear-gradient(#262b50,#3c3660 30%,#684a60 55%,#8c6060 75%,#9a6a60),#3a3560""",
            "--plaque-fill": "#1D2739",
            "--plaque-border": "#F2C078",
            "--dgcv-table-shadow": "0 2px 10px #0000008c",
        },
    ),
    "monet": ThemeConfig(
        bg_primary="#f7f9e4",
        bg_surface="#2a5d67",
        bg_alt="#c6f3d8",
        bg_hover="#d0ece7",
        text_main="#2a5d67",
        text_heading="#fff",
        text_hover="#2a5d67",
        border_main="#2a5d67",
        border_alt="#b2dfcc",
        bg_action="#1a3d44",
        text_on_action="#fff",
        bg_action_hover="#3b7a86",
        bg_error="#d9534f",
        text_on_error="#fff",
        bg_success="#5cb85c",
        text_on_success="#fff",
        custom_css_vars={
            "--dgcv-table-shadow": "0 2px 8px #2a5d6733",
        },
    ),
    "newspaper": ThemeConfig(
        bg_primary="#f4f4f4",
        bg_surface="#fafafa",
        bg_alt="#fafafa",
        bg_hover="#e0e0e0",
        text_main="#000",
        text_heading="#000",
        text_hover="#000",
        border_main="#333",
        border_alt="#808080",
        bg_action="#000",
        text_on_action="#fff",
        bg_action_hover="#4d4d4d",
        bg_error="#8b0000",
        text_on_error="#fff",
        bg_success="#006400",
        text_on_success="#fff",
        font_family="Georgia,serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-table-shadow": "0px 2px 5px #0000001a",
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
        text_on_action="#fff",
        bg_action_hover="#0090c8",
        bg_error="#ff4b2b",
        text_on_error="#fff",
        bg_success="#00e676",
        text_on_success="#020d1a",
        font_family="Georgia,serif",
        custom_css_vars={
            "--dgcv-special-background": "radial-gradient(ellipse at top,#0a2a4a,#041424 40%,#020810)",
            "--dgcv-table-shadow": "0 0 30px #0077a880,0 0 80px #00e5ff26",
            "--dgcv-text-shadow": "0 0 10px #00e5ff80",
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
        border_main="#64a",
        border_alt="#1a1428",
        bg_action="#24c",
        text_on_action="#fff",
        bg_action_hover="#88b0b4",
        bg_error="#ff4b2b",
        text_on_error="#fff",
        bg_success="#00e676",
        text_on_success="#0a0a0f",
        font_family="Georgia,serif",
        custom_css_vars={
            "--dgcv-special-background": "linear-gradient(135deg,#0a0a0f,#1a0a2e 15%,#0a1a3a 30%,#0a2a1a 45%,#1a2a0a 55%,#2a1a0a 65%,#2a0a1a 75%,#1a0a2e 88%,#0a0a0f)",
            "--dgcv-table-shadow": "0 0 20px #64a9,0 0 60px #2244cc4c,inset 0 1px 0 #c896ff26",
            "--dgcv-text-shadow": "0 0 12px #b478ffb2",
            "--dgcv-border-image": "linear-gradient(135deg,#f0f,#04f,#0fc,#fa0,#f0f) 1",
            "--dgcv-border-radius": "0px",
        },
    ),
    "paper_graphite": ThemeConfig(
        bg_primary="#f5f4f2",
        bg_surface="#eceae6",
        bg_alt="#f0eeeb",
        bg_hover="#69707e",
        text_main="#1c1c1e",
        text_heading="#2c2c30",
        text_hover="#f5f4f2",
        text_alt="#4a4a50",
        border_main="#b0aaa0",
        border_alt="#d8d4ce",
        bg_action="#4a4a50",
        text_on_action="#f5f4f2",
        bg_action_hover="#6b7280",
        bg_error="#c34",
        text_on_error="#f5f4f2",
        bg_success="#3a6644",
        text_on_success="#f5f4f2",
        font_family="Georgia,serif",
        custom_css_vars={
            "--dgcv-special-background": "linear-gradient(135deg,#eceae6,#e4e0da,#eceae6)",
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
        bg_error="#c34",
        text_on_error="#faf7f0",
        bg_success="#3a6644",
        text_on_success="#faf7f0",
        font_family="Georgia,serif",
        custom_css_vars={
            "--dgcv-special-background": "linear-gradient(135deg,#f0ebe0,#e8e0d0,#f0ebe0)",
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
        bg_error="#c34",
        text_on_error="#f2f5f0",
        bg_success="#3a6644",
        text_on_success="#f2f5f0",
        font_family="Georgia,serif",
        custom_css_vars={
            "--dgcv-special-background": "linear-gradient(135deg,#e4ece0,#d8e8d0,#e4ece0)",
        },
    ),
    "picasso_blue": ThemeConfig(
        bg_primary="#eaf2f8",
        bg_surface="#154360",
        bg_alt="#d6eaf8",
        bg_hover="#aed6f1",
        text_main="#154360",
        text_heading="#fff",
        text_hover="#154360",
        border_main="#154360",
        border_alt="#85c1e9",
        bg_action="#1f618d",
        text_on_action="#fff",
        bg_action_hover="#2980b9",
        bg_error="#c0392b",
        text_on_error="#fff",
        bg_success="#27ae60",
        text_on_success="#fff",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-table-shadow": "0 2px 10px #15436033",
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
        text_on_action="#fff",
        bg_action_hover="#3a70b0",
        bg_error="#c0453c",
        text_on_error="#fff",
        bg_success="#2f7d4f",
        text_on_success="#fff",
        font_family="Georgia,serif",
        custom_css_vars={
            "--dgcv-special-text": "#e6e8ea",
            "--dgcv-special-background": "repeating-linear-gradient(45deg,#ffffff0f 0 1px,#0000000f 1px 2px,#0000 2px 4px),repeating-linear-gradient(90deg,#27406a80 0 18px,#161a2280 18px 26px,#234a3080 26px 44px,#161a2280 44px 52px),repeating-linear-gradient(0deg,#27406a 0 18px,#161a22 18px 26px,#234a30 26px 44px,#161a22 44px 52px),#161a22",
            "--plaque-fill": "#1e2530",
            "--plaque-border": "#86abca",
            "--dgcv-table-shadow": "0 1px 6px #00000073,inset 0 1px 0 #ffffff0f",
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
        text_on_action="#fff",
        bg_action_hover="#9c6230",
        bg_error="#a83321",
        text_on_error="#fff",
        bg_success="#5a7029",
        text_on_success="#fff",
        font_family="Georgia,serif",
        custom_css_vars={
            "--dgcv-special-text": "#2e2414",
            "--dgcv-special-background": "repeating-linear-gradient(45deg,#ffffff0f 0 1px,#0000000f 1px 2px,#0000 2px 4px),repeating-linear-gradient(90deg,#e7d9bf80 0 14px,#9a915280 14px 26px,#f3ebda80 26px 34px,#c88a6380 34px 46px,#f3ebda80 46px 52px,#b87a4a80 52px 54px,#f3ebda80 54px 60px),repeating-linear-gradient(0deg,#e7d9bf 0 14px,#9a9152 14px 26px,#f3ebda 26px 34px,#c88a63 34px 46px,#f3ebda 46px 52px,#b87a4a 52px 54px,#f3ebda 54px 60px),#e7d9bf",
            "--plaque-fill": "#ece0c6",
            "--plaque-border": "#8a4a20",
            "--dgcv-table-shadow": "0 1px 4px #2e241424,inset 0 1px 0 #ffffff8c",
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
        text_on_action="#fff",
        bg_action_hover="#a93e52",
        bg_error="#c0453c",
        text_on_error="#fff",
        bg_success="#2f7d4f",
        text_on_success="#fff",
        font_family="Georgia,serif",
        custom_css_vars={
            "--dgcv-special-text": "#f0e6d8",
            "--dgcv-special-background": "repeating-linear-gradient(45deg,#ffffff0f 0 1px,#0000000f 1px 2px,#0000 2px 4px),repeating-linear-gradient(90deg,#5e203080 0 18px,#18101680 18px 24px,#1f3a2880 24px 42px,#18101680 42px 48px,#8a652680 48px 50px,#18101680 50px 54px),repeating-linear-gradient(0deg,#5e2030 0 18px,#181016 18px 24px,#1f3a28 24px 42px,#181016 42px 48px,#8a6526 48px 50px,#181016 50px 54px),#181016",
            "--plaque-fill": "#241820",
            "--plaque-border": "#cf9442",
            "--dgcv-table-shadow": "0 1px 6px #00000080,inset 0 1px 0 #ffffff0d",
        },
    ),
    "purples": ThemeConfig(
        bg_primary="#2d2b55",
        bg_surface="#3f3f7a",
        bg_alt="#4a4a8a",
        bg_hover="#6a5acd",
        text_main="#f5f5f5",
        text_heading="#e6e6fa",
        text_hover="#f5f5f5",
        border_main="#6a5acd",
        border_alt="#7b68ee",
        bg_action="#8a2be2",
        text_on_action="#fff",
        bg_action_hover="#9370db",
        bg_error="#ff1493",
        text_on_error="#fff",
        bg_success="#00fa9a",
        text_on_success="#2d2b55",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-special-background": "linear-gradient(160deg,#2d2b55,#3f3f7a,#2d2b55)",
            "--dgcv-table-shadow": "0 0 20px #6a5acdb2,0 0 60px #8a2be24c",
            "--dgcv-text-shadow": "0 0 8px #6a5acd80",
        },
    ),
    "quilt_applique": ThemeConfig(
        bg_primary="#0f2420",
        bg_surface="#163530",
        bg_alt="#122a26",
        bg_hover="#f0a878",
        text_main="#f0ece0",
        text_heading="#f0a878",
        text_hover="#0f2420",
        text_alt="#a8c0b8",
        border_main="#c87a5a",
        border_alt="#2a4c44",
        bg_action="#3e9b7a",
        text_on_action="#062018",
        bg_action_hover="#54b292",
        bg_error="#a8382c",
        text_on_error="#fbe6de",
        bg_success="#2f6b4a",
        text_on_success="#e2f0e2",
        font_family="'Bookman Old Style', Bookman, 'Palatino Linotype', Palatino, Georgia, serif",
        custom_css_vars={
            "--dgcv-border-width": "1px",
            "$randomized": "wx=0:139 wy=0:139 ws=110:170 mx=18:82 my=20:80 ms=150:230 sx=10:88 sy=14:86 ss=130:200 px=8:90 py=10:88 ps=64:104 pp=0:3 gx=0:159 gy=0:159 gs=150:200",
            "$assemble": "--dgcv-special-background",
            "$part_1": """url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='140' height='140'><filter id='w' x='0' y='0' width='100%' height='100%'><feTurbulence type='fractalNoise' baseFrequency='.06 .85' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 .93 0 0 0 0 .89 0 0 0 0 .79 .3 0 0 0 -.15'/></filter><rect width='140' height='140' filter='url(%23w)'/></svg>") ${wx}px ${wy}px/${ws}px ${ws}px,url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='-60 -60 120 120' stroke='%23f2e6ce' stroke-width='1.5' stroke-dasharray='3 2.8' stroke-opacity='.7' stroke-linejoin='round'><defs><path id='p' d='M0-10C9-20 12-32 0-47-12-32-9-20 0-10zM10 0C20 9 32 12 47 0 32-12 20-9 10 0zM0 10C-9 20-12 32 0 47 12 32 9 20 0 10zM-10 0C-20-9-32-12-47 0-32 12-20 9-10 0z'/><radialGradient id='g' gradientUnits='userSpaceOnUse' r='47'><stop offset='.2' stop-color='%23dc6a50'/><stop offset='1' stop-color='%23f09274'/></radialGradient></defs><circle r='51' fill='none' stroke-width='.9' stroke-opacity='.26'/><g fill='%234a1f38' stroke-opacity='.55'><use href='%23p' transform='rotate(22.5)scale(.78)'/><use href='%23p' transform='rotate(67.5)scale(.78)'/></g><g fill='url(%23g)'><use href='%23p'/><use href='%23p' transform='rotate(45)'/></g><circle r='13.5' fill='%23dcab4f'/><circle r='5.5' fill='%233a162c' stroke-width='1.1'/></svg>") calc(${mx}% + max(0px,162px - 100%)*99) ${my}%/min(${ms}px,46%) no-repeat""",
            "$part_2@detail": """url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='-60 -40 120 80' stroke='%23f2e6ce' stroke-width='1.4' stroke-linejoin='round'><defs><path id='l' d='M0 0C8-9 22-11 31-3 22 7 8 7 0 0z'/><path id='s' d='M-52 26C-32 20-14 6 0-12 11-25 24-28 38-27' fill='none' stroke-linecap='round'/></defs><use href='%23s' stroke='%234e9268' stroke-width='5'/><g stroke-dasharray='2.8 2.6' stroke-opacity='.6'><use href='%23s'/><g fill='%236bb587'><use href='%23l' transform='translate(-38 22)rotate(38)scale(.85)'/><use href='%23l' transform='translate(-24 15)rotate(215)scale(.75)'/><use href='%23l' transform='translate(-8 0)rotate(52)scale(.9)'/><use href='%23l' transform='translate(6-16)rotate(228)scale(.8)'/><use href='%23l' transform='translate(22-25)rotate(28)scale(.68)'/></g><path d='M0-26C11-14 11-4 0 0-11-4-11-14 0-26z' transform='translate(38-27)rotate(72)scale(.72)' fill='%23e69a5e'/></g></svg>") calc(${sx}% + max(0px,174px - 100%)*99) ${sy}%/min(${ss}px,42%) no-repeat,url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='-40 -40 80 80' stroke='%23f2e6ce' stroke-width='1.5' stroke-dasharray='3 2.6' stroke-opacity='.65'><circle r='34' fill='none' stroke-width='.9' stroke-opacity='.22'/><path d='M0-26C11-14 11-4 0 0-11-4-11-14 0-26zM26 0C14 11 4 11 0 0 4-11 14-11 26 0zM0 26C-11 14-11 4 0 0 11 4 11 14 0 26zM-26 0C-14-11-4-11 0 0-4 11-14 11-26 0z' fill='%23dcab4f' stroke-linejoin='round'/><circle r='5' fill='%235a2844' stroke-width='1.1'/></svg>") calc(${px}% + max(0,${pp} - 2)*9999px + max(0px,234px - 100%)*99) ${py}%/min(${ps}px,22%) no-repeat""",
            "$part_3": """radial-gradient(122% 98% at 50% 38%,#ffe1b412,#ffe1b405 34%,#06120f1f 66%,#06120f6b),url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><path d='M12 12h68v68H12z' fill='%2314322c'/><path d='M92 12h68v68H92z' fill='%23123029'/><path d='M12 92h68v68H12z' fill='%2316352e'/><path d='M92 92h68v68H92z' fill='%23112823'/><g stroke='%23f2e6ce' fill='none'><path d='M6 0v160M86 0v160M0 6h160M0 86h160' stroke-dasharray='4.5 4' opacity='.22'/><path d='M17 17h58v58H17zM97 17h58v58H97zM17 97h58v58H17zM97 97h58v58H97z' stroke-width='.9' stroke-dasharray='3.4 3.6' opacity='.16'/></g></svg>") ${gx}px ${gy}px/${gs}px ${gs}px,#0c1f1b""",
            "--dgcv-special-text": "#f4e9d2",
            "--plaque-fill": "#122e28",
            "--plaque-border": "#f0a878",
            "--dgcv-table-shadow": "0 2px 12px #040e0c8c",
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
        text_on_action="#fff",
        bg_action_hover="#3e6ec0",
        bg_error="#c83a5a",
        text_on_error="#fff",
        bg_success="#2a9e7a",
        text_on_success="#08201a",
        font_family="Georgia,serif",
        custom_css_vars={
            "--dgcv-special-text": "#eef2f8",
            "--dgcv-special-background": "repeating-linear-gradient(60deg,#155e5e 0 14px,#1e4a82 14px 28px,#36347e 28px 42px,#4e2e72 42px 56px,#6e2a64 56px 70px,#4e2e72 70px 84px,#36347e 84px 98px,#1e4a82 98px 112px)",
            "--plaque-fill": "#1e2742",
            "--plaque-border": "#54cfc4",
            "--dgcv-table-shadow": "0 0 20px #6e2a6466,0 0 60px #2e5aa840,inset 0 1px 0 #eef2f81f",
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
        text_on_action="#fff",
        bg_action_hover="#356cb0",
        bg_error="#b5302a",
        text_on_error="#fff",
        bg_success="#2f7d52",
        text_on_success="#fff",
        font_family="Georgia,serif",
        custom_css_vars={
            "--dgcv-special-text": "#0e2138",
            "--dgcv-special-background": "repeating-linear-gradient(90deg,#3a6ea59e 0 9px,#0000 9px 18px),repeating-linear-gradient(0deg,#3a6ea59e 0 9px,#0000 9px 18px),#fcfaf5",
            "--plaque-fill": "#eaf1f9",
            "--plaque-border": "#244c86",
            "--dgcv-table-shadow": "0 1px 4px #12203529,inset 0 1px 0 #fff9",
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
        font_family="Georgia,serif",
        custom_css_vars={
            "--dgcv-special-text": "#f2e8d8",
            "--dgcv-special-background": "repeating-linear-gradient(45deg,#ffffff0d 0 1px,#0000 1px 4px) 0 0/4px 4px,repeating-linear-gradient(-45deg,#0000000f 0 1px,#0000 1px 5px) 0 0/5px 5px,repeating-linear-gradient(0deg,#8a7048 0 1px,#0000 1px 22px) 0 0/22px 22px,repeating-linear-gradient(90deg,#8a7048 0 1px,#0000 1px 22px) 0 0/22px 22px,conic-gradient(#7a2a2a 0 25%,#1f5a52 25% 50%,#243a6a 50% 75%,#6a4a1e 75% 100%) 0 0/22px 22px,#7a2a2a",
            "--plaque-fill": "#2c241c",
            "--plaque-border": "#e0a83e",
        },
    ),
    "rembrandt": ThemeConfig(
        bg_primary="#fffaf0",
        bg_surface="#523d2e",
        bg_alt="#f7e6d5",
        bg_hover="#d4b996",
        text_main="#523d2e",
        text_heading="#fff",
        text_hover="#523d2e",
        border_main="#6c4f3d",
        border_alt="#ebd4b3",
        bg_action="#38291f",
        text_on_action="#fff",
        bg_action_hover="#6c4f3d",
        bg_error="#a52a2a",
        text_on_error="#fff",
        bg_success="#556b2f",
        text_on_success="#fff",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-special-background": "linear-gradient(170deg,#fffaf0,#f7e6d5 60%,#fffaf0)",
            "--dgcv-special-text": "#523d2e",
            "--dgcv-table-shadow": "0 4px 16px #523d2e4c,inset 0 0 40px #523d2e0d",
            "--dgcv-text-shadow": "0 1px 0 #ffffff80",
        },
    ),
    "sakura": ThemeConfig(
        bg_primary="#f8f1ea",
        bg_surface="#b0476e",
        bg_alt="#f2e7db",
        bg_hover="#f3c4ce",
        text_main="#4e3228",
        text_heading="#fff",
        text_hover="#4e3228",
        text_alt="#6e4a3a",
        border_main="#c0986e",
        border_alt="#e0c8ac",
        bg_action="#b0506c",
        text_on_action="#fff",
        bg_action_hover="#b85470",
        bg_error="#be3a2c",
        text_on_error="#fff",
        bg_success="#4a7c3e",
        text_on_success="#fff",
        font_family="Palatino,serif",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-special-background": "linear-gradient(135deg,#46291f,#5a382a 28%,#6e4636 50%,#7a4a42 72%,#8a5848)",
            "--dgcv-table-shadow": "0 0 18px #6e463438,inset 0 1px 0 #ffffff80",
            "--dgcv-text-shadow": "0 1px 0 #ffffff80",
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
        font_family="Georgia,serif",
        custom_css_vars={
            "--dgcv-special-background": "linear-gradient(135deg,#161009,#4a3216 14%,#9a6a26 28%,#e8b860 42%,#fcd98a 56%,#e8e0d0 68%,#b8d0cc 80%,#6aa8a4 90%,#161009)",
            "--dgcv-table-shadow": "0 0 20px #cc9a3873,0 0 60px #6aa8a440,inset 0 1px 0 #fdf3da26",
            "--plaque-fill": "#251a10",
            "--plaque-border": "#6aa8a4",
            "--dgcv-text-shadow": "0 1px 2px #000c,0 0 6px #0008",
            "--dgcv-border-image": "linear-gradient(135deg,#cc9a38,#e8b860,#e8e0d0,#6aa8a4,#cc9a38) 1",
            "--dgcv-border-radius": "0px",
        },
    ),
    "slate_and_copper": ThemeConfig(
        bg_primary="#2f3e46",
        bg_surface="#2f3e46",
        bg_alt="#354f52",
        bg_hover="#b87333",
        text_main="#cad2c5",
        text_heading="#cad2c5",
        text_hover="#0e1d25",
        border_main="#b87333",
        border_alt="#52796f",
        bg_action="#b87333",
        text_on_action="#fff",
        bg_action_hover="#cd7f32",
        bg_error="#e07a5f",
        text_on_error="#fff",
        bg_success="#81b29a",
        text_on_success="#2f3e46",
        font_family="Georgia,serif",
        custom_css_vars={
            "--dgcv-table-shadow": "0 2px 8px #b873334c",
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
        text_on_action="#fff",
        bg_action_hover="#2e8ae0",
        bg_error="#e23b3b",
        text_on_error="#fff",
        bg_success="#2bc070",
        text_on_success="#16204e",
        font_family="Georgia,serif",
        custom_css_vars={
            "--dgcv-special-background": "radial-gradient(1.5px 1.5px at 12% 20%,#f7d84b,#0000),radial-gradient(2px 2px at 28% 12%,#f7d84b,#0000),radial-gradient(1px 1px at 45% 28%,#fff,#0000),radial-gradient(2px 2px at 62% 16%,#f7d84b,#0000),radial-gradient(1.5px 1.5px at 78% 24%,#fff,#0000),radial-gradient(1px 1px at 88% 12%,#f7d84b,#0000),radial-gradient(2px 2px at 18% 46%,#f7d84b,#0000),radial-gradient(1px 1px at 38% 58%,#fff,#0000),radial-gradient(1.5px 1.5px at 55% 50%,#f7d84b,#0000),radial-gradient(1px 1px at 72% 62%,#fff,#0000),radial-gradient(2px 2px at 30% 78%,#f7d84b,#0000),radial-gradient(1.5px 1.5px at 50% 86%,#fff,#0000),radial-gradient(1px 1px at 84% 80%,#f7d84b,#0000),radial-gradient(circle at 82% 78%,#f7d84b8c,#f7d84b1f 14%,#0000 32%),linear-gradient(to bottom,#12123a,#1a1c4e 30%,#262a6e 55%,#2e4a96)",
            "--dgcv-table-shadow": "0 0 20px #f7d84b8c,0 0 60px #2e5aa859,inset 0 1px 0 #f7d84b2e",
            "--dgcv-text-shadow": "0 0 10px #f7d84b80",
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
        text_on_action="#fff",
        bg_action_hover="#e91e63",
        bg_error="#b71c1c",
        text_on_error="#fff",
        bg_success="#00e676",
        text_on_success="#000",
        font_family="Georgia,serif",
        custom_css_vars={
            "--dgcv-special-background": "linear-gradient(160deg,#ff6b35,#f7431c 20%,#c2185b 50%,#7b1fa2 80%,#4a148c)",
            "--dgcv-table-shadow": "0 4px 24px #c2185b80,0 0 60px #ff6b3533",
            "--dgcv-text-shadow": "0 1px 4px #00000080",
        },
    ),
    "van_gogh": ThemeConfig(
        bg_primary="#f7f2e7",
        bg_surface="#1c6ea4",
        bg_alt="#fff7d1",
        bg_hover="#ffd700",
        text_main="#1c6ea4",
        text_heading="#fff",
        text_hover="#12649a",
        border_main="#1c6ea4",
        border_alt="#104e76",
        bg_action="#e6b800",
        text_on_action="#1c6ea4",
        bg_action_hover="#cca300",
        bg_error="#c30",
        text_on_error="#fff",
        bg_success="#396",
        text_on_success="#fff",
        custom_css_vars={
            "--dgcv-border-width": "2px",
            "--dgcv-special-background": "linear-gradient(160deg,#f7f2e7,#fff7d1,#f7f2e7)",
            "--dgcv-special-text": "#1c6ea4",
            "--dgcv-table-shadow": "0 4px 12px #1c6ea440",
            "--dgcv-text-shadow": "0 1px 0 #ffffff80",
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
