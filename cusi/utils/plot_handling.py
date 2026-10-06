import seaborn as sns
import pandas as pd
from typing import Any, Callable, Optional
from cusi.utils.parameter_handling import load_parameters
from cusi.utils.log_handling import log_error, log_info, log_warn, log_dict
import matplotlib.pyplot as plt
import os


class Plotter:
    """Plot styling, sizing and saving; plotting logic lives in plot functions. Use show(), not plt.show()."""

    COLOURS = (
        []
    )

    DEFAULTS = {
        "font_size": 16, 
        "labels_font_size": 19,
        "xtick_font_size": 19,
        "ytick_font_size": 15,
        "legend_font_size": 16,
        "title_font_size": 20
    }

    def __init__(self, parameters: Optional[dict[str, Any]] = None) -> None:
        self.parameters = load_parameters(parameters)
        self.size_params = {}
        sns.set_style("whitegrid")
        plt.rcParams["font.serif"] = ["Times New Roman"]
        self.default_plt_params = plt.rcParams.copy()
        self.default_plt_params["font.size"] = self.DEFAULTS["font_size"]
        self.default_plt_params["axes.labelsize"] = self.DEFAULTS["labels_font_size"]
        self.default_plt_params["xtick.labelsize"] = self.DEFAULTS["xtick_font_size"]
        self.default_plt_params["ytick.labelsize"] = self.DEFAULTS["ytick_font_size"]
        self.default_plt_params["axes.titlesize"] = self.DEFAULTS["title_font_size"]
        
        
        self.set_size_parameters()

    def set_size_parameters(
        self,
        scaler: float = 1,
        font_size: Optional[float] = None,
        labels_font_size: Optional[float] = None,
        xtick_font_size: Optional[float] = None,
        ytick_font_size: Optional[float] = None,
        legend_font_size: Optional[float] = None,
        title_font_size: Optional[float] = None,
    ) -> None:
        """Set font sizes times ``scaler``; None falls back to the default for that setting."""
        if font_size is None:
            font_size = self.default_plt_params["font.size"]
        plt.rcParams.update({"font.size": font_size * scaler})
        if labels_font_size is None:
            labels_font_size = self.default_plt_params["axes.labelsize"]
        plt.rcParams.update({"axes.labelsize": labels_font_size * scaler})
        if xtick_font_size is None:
            xtick_font_size = self.default_plt_params["xtick.labelsize"]
        plt.rcParams.update({"xtick.labelsize": xtick_font_size * scaler})
        if ytick_font_size is None:
            ytick_font_size = self.default_plt_params["ytick.labelsize"]
        plt.rcParams.update({"ytick.labelsize": ytick_font_size * scaler})
        if title_font_size is None:
            title_font_size = self.default_plt_params["axes.titlesize"]
        if legend_font_size is None:
            legend_font_size = self.DEFAULTS["legend_font_size"]
        plt.rcParams.update({"axes.titlesize": title_font_size * scaler})
        self.size_params["font_size"] = font_size * scaler
        self.size_params["labels_font_size"] = labels_font_size * scaler
        self.size_params["xtick_font_size"] = xtick_font_size * scaler
        self.size_params["ytick_font_size"] = ytick_font_size * scaler
        self.size_params["title_font_size"] = title_font_size * scaler
        self.size_params["legend_font_size"] = legend_font_size * scaler
        return

    def set_size_default(self, scaler: float = 1) -> None:
        self.set_size_parameters(
            scaler=scaler,
            font_size=None,
            labels_font_size=None,
            xtick_font_size=None,
            ytick_font_size=None,
            legend_font_size=None,
            title_font_size=None,
        )
        return

    def get_size_input_number(self, key_name: str) -> float:
        """Prompt until a positive float is entered for ``key_name``; empty input keeps the current value."""
        while True:
            got = input(
                f"Enter the size for {key_name} (current value is {self.size_params[key_name]}, hit enter to keep current value ):"
            )
            got = got.strip()
            if got.strip() == "":
                return self.size_params[key_name]
            try:
                got = float(got)
                if got <= 0:
                    log_warn(
                        f"Got {got}, but it must be greater than 0",
                        parameters=self.parameters,
                    )
                    continue
                return got
            except ValueError:
                log_warn(
                    f"Got {got}, but it must be a number", parameters=self.parameters
                )
                continue

    def test_sizes(self, plot_func: Callable[[], None]) -> None:
        """Interactively re-render ``plot_func`` with new font sizes until the user accepts."""
        self.set_size_default()
        done = False
        while not done:
            log_info(f"Plot with sizes: ", parameters=self.parameters)
            log_dict(self.size_params, n_indents=1, parameters=self.parameters)
            plot_func()
            plt.show()
            keepgoing = input(
                "Do you want to keep trying different sizes? (only y will keep going):"
            )
            if keepgoing.lower().strip() == "y":
                for key in self.size_params:
                    self.size_params[key] = self.get_size_input_number(key)
                self.set_size_parameters_from_dict(self.size_params)
            else:
                done = True
                break
        return

    def show(self, save_path: Optional[str] = None) -> None:
        """Save as .pdf and .png under figure_dir (shown unless figure_skip_show), or just show if no save_path."""
        if save_path is not None:
            if os.path.abspath(save_path).startswith(
                os.path.abspath(self.parameters["figure_dir"])
            ):
                figure_path = save_path
            else:
                figure_path = self.parameters["figure_dir"] + f"/{save_path}"
        else:
            figure_path = None
        if figure_path is not None:
            figure_dir = os.path.dirname(figure_path)
            if not os.path.exists(figure_dir):
                os.makedirs(figure_dir)
            figure_path = figure_path.replace(".pdf", "").replace(".png", "")
            plt.savefig(f"{figure_path}.pdf")
            plt.savefig(f"{figure_path}.png")
            log_info(f"Saved figure to {figure_path}.pdf", parameters=self.parameters)
            if not self.parameters["figure_skip_show"]:
                plt.show()
            else:
                plt.clf()
        else:
            plt.show()
        return

    def get_stacked_bar_plot_func(
        self,
        df: pd.DataFrame,
        x_col: str,
        stacked_cols: list[str],
        colours: list[Any],
        skip_col: Optional[str] = None,
        skip_text_y_dip: float = 15,
        skip_text_rotation: float = 20,
        x_tick_rotation: float = 45,
        y_label: str = "Percentage (%)",
        tight_layout: bool = True,
    ) -> Callable[[], None]:
        """Zero-argument stacked bar plot function; rows must be sorted by (skip_col, x_col), skip_col groups bars."""
        parameters = self.parameters
        for col in (
            stacked_cols + [x_col] + ([skip_col] if skip_col is not None else [])
        ):
            if col not in df.columns:
                log_error(
                    f"Column {col} is not in the dataframe with columns {list(df.columns)}. Please check your input.",
                    parameters=parameters,
                )
        if len(colours) != len(stacked_cols):
            log_error(
                f"Length of colours {len(colours)} does not match length of stacked_cols {len(stacked_cols)}. Please check your input.",
                parameters=parameters,
            )

        def plot_func():
            fig, ax = plt.subplots(figsize=(12, 6))

            x_positions = []
            current_x = 0
            last_skip = None

            for i, row in df.iterrows():
                if skip_col is not None:
                    if last_skip is not None and row[skip_col] != last_skip:
                        current_x += 1.0  # gap between skip_col groups

                x_positions.append(current_x)

                bottom = 0
                for col, color in zip(stacked_cols, colours):
                    val = row[col]
                    ax.bar(
                        current_x,
                        val,
                        bottom=bottom,
                        color=color,
                        edgecolor="white",
                        width=0.8,
                    )
                    bottom += val
                if skip_col is not None:
                    last_skip = row[skip_col]
                current_x += 1

            ax.set_xticks(x_positions)
            ax.set_xticklabels(df[x_col], rotation=x_tick_rotation, ha="right")

            if skip_col is not None:
                df["x_pos"] = x_positions
                for method, group in df.groupby(skip_col, sort=False):
                    mid_point = group["x_pos"].mean()
                    ax.text(
                        mid_point,
                        -skip_text_y_dip,
                        method,
                        ha="center",
                        fontweight="bold",
                        rotation=skip_text_rotation,
                        fontsize=self.size_params["labels_font_size"],
                    )

            ax.set_ylabel(y_label)
            ax.legend(
                stacked_cols,
                loc="lower left",
                bbox_to_anchor=(0, 1, 1, 0.1),
                mode="expand",
                ncol=len(stacked_cols),
                borderaxespad=0,
                frameon=False,
                fontsize=self.size_params["legend_font_size"],
            )
            if tight_layout:
                plt.tight_layout(rect=[0, 0, 1, 0.95])

        return plot_func
