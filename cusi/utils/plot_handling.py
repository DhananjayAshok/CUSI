import seaborn as sns
from typing import Any, Optional
from cusi.utils.parameter_handling import load_parameters
from cusi.utils.log_handling import log_info
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

