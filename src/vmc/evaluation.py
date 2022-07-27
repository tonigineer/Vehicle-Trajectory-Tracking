"""File contains class to visualize results."""

import random

import numpy as np
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from dataclasses import dataclass, asdict


@dataclass
class ScatterEntry:
    """Container for scatter plot information."""

    x: np.ndarray
    y: np.ndarray
    name: str
    x_label: str = "[-]"
    y_label: str = "[-]"


@dataclass
class SubplotLayout:
    """Layout configuration for a subplot page.

    Title is mandatory, all others settings are optional.

    templates
    ---------
        `plotly`, `plotly_white`, `plotly_dark`,
        `ggplot2`, `seaborn`, `simple_white`
    """

    title: str
    template: str = 'plotly_dark'

    # TODO: implement font
    # font: dict = {
    #     'family': "Courier New, monospace",
    #     'size': 22,
    #     'color': "RebeccaPurple"
    # }

    legend_title: str = 'Legend'
    updatemenus: None = None
    hovermode: str = 'closest'


class Evaluation:
    """Master class to visualize results."""

    @staticmethod
    def __scatter_plot(x, y, name, mode='lines', color=None, showlegend=True):
        """Return on scatter plot that can be appended to fig data."""
        return go.Scatter(
            x=x,
            y=y,
            mode=mode,
            line=dict(
                width=2,
                color=color,
            ),
            name=name,
            showlegend=showlegend
        )

    @classmethod
    def subplots(cls, data, layout: SubplotLayout, *, columns=3):
        """Create page with subplots.

        Parameter
        ---------
            data : list
                List of ScatterEntry items.

            columns : int
                Number of columns.
        """
        num_plots = len(data)
        fig = make_subplots(rows=int(np.ceil(num_plots/columns)), cols=columns,
                            subplot_titles=tuple([d.name for d in data]))

        for i in range(num_plots):
            r, c = (i+columns) // columns, i % columns + 1

            fig.add_trace(
                cls.__scatter_plot(
                    x=data[i].x, y=data[i].y,
                    name=data[i].name,
                    showlegend=False
                ),
                row=r, col=c,
            )
            fig.update_xaxes(title_text=data[i].x_label, row=r, col=c),
            fig.update_yaxes(title_text=data[i].y_label, row=r, col=c)

        fig.update_layout(**asdict(layout))
        fig.show()


if __name__ == "__main__":
    # Show plots with random data.
    sl = SubplotLayout('Test plot')
    data = [
        ScatterEntry(
            np.array(random.sample(range(10, 30), 5)),
            np.array(range(1, 6)),
            'Test signal',
            'time [s]',
            'Signal [-]'
        ) for _ in range(9)]

    Evaluation.subplots(data, sl, columns=3)
