"""Tools the model can call. Phase H2 grows this package; H1 ships only echo."""

from .base import Tool, ToolResult
from .echo import EchoTool
from .registry import ToolRegistry

__all__ = ["EchoTool", "Tool", "ToolRegistry", "ToolResult"]
