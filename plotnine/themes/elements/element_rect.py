"""
Theme elements used to decorate the graph.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .element_base import element_base

if TYPE_CHECKING:
    from typing import Any, Optional, Sequence

    from plotnine.typing import TupleFloat3, TupleFloat4


class element_rect(element_base):
    """
    Theme element: Rectangle

    Used for backgrounds and borders

    Parameters
    ----------
    fill : str | tuple
        Rectangle background color
    color : str | tuple
        Line color
    colour : str | tuple
        Alias of color
    size : float
        Line thickness
    kwargs : dict
        Parameters recognised by
        :class:`matplotlib.patches.Rectangle`. In some cases
        you can use the fancy parameters from
        :class:`matplotlib.patches.FancyBboxPatch`.
    """

    def __init__(
        self,
        fill: Optional[str | TupleFloat3 | TupleFloat4] = None,
        color: Optional[str | TupleFloat3 | TupleFloat4] = None,
        size: Optional[float] = None,
        linetype: Optional[str | Sequence[int]] = None,
        colour: Optional[str | TupleFloat3 | TupleFloat4] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.properties.update(**kwargs)

        color = color if color else colour
        if fill:
            self.properties["facecolor"] = fill
        if color:
            self.properties["edgecolor"] = color
        if size:
            self.properties["linewidth"] = size
        if linetype:
            self.properties["linestyle"] = linetype
