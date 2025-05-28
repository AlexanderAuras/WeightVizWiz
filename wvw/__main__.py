from collections import OrderedDict
import os
from pathlib import Path
import sys
from typing import Any, Literal, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PySide6.QtCore import QCoreApplication, QFile, QIODeviceBase, Qt, Slot
from PySide6.QtGui import QImage
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox
import torch
from torch import Tensor

from wvw.convert_image import numpy_to_qimg


matplotlib.use("Agg")
torch.set_grad_enabled(False)


current_weights: OrderedDict[str, Tensor] = cast(OrderedDict[str, Tensor], None)


def load_weights(main_window: Any) -> None:
    global current_weights
    # path = QFileDialog.getOpenFileName(main_window, "Load weights", str(Path(__file__).parents[1] / "resources"), "PyTorch weights (*.pt *.pth);;All files (*)")[0]
    path = QFileDialog.getOpenFileName(main_window, "Load weights", "/home/alexander/Documents/Projects/fno_unet/runs/64x64_few", "PyTorch weights (*.pt *.pth);;All files (*)")[0]
    if path == "":
        return
    try:
        current_weights = torch.load(path, map_location="cpu")
        main_window.action_show_weight.setEnabled(True)
    except Exception:
        _ = QMessageBox.critical(main_window, "Failed to load weights", "General error", QMessageBox.StandardButton.Ok)
        return


def viz_params(weight: Tensor, bias: Tensor | None, highlight_mode: Literal["value", "sign"]) -> QImage:
    plt.subplots(1, 2 if bias is not None else 1, width_ratios=[1, 0.2] if bias is not None else [1])
    plt.subplot(1, 2 if bias is not None else 1, 1)
    plt.tick_params(axis="both", which="both", top=False, bottom=False, left=False, right=False, labelbottom=False, labelleft=False)
    if highlight_mode == "value":
        plt.imshow(weight.numpy(), cmap="gray")
    elif highlight_mode == "sign":
        plt.imshow(weight.sign().numpy(), cmap="gray", vmin=-1.0, vmax=1.0)
    norm_weight = (weight - weight.min()) / (weight.max() - weight.min())
    for y in range(weight.shape[0]):
        for x in range(weight.shape[1]):
            if highlight_mode == "value":
                text_color = "white" if norm_weight[y, x].item() <= 0.5 else "black"
            elif highlight_mode == "sign":
                text_color = "white" if weight[y, x].sign().item() <= 0.0 else "black"
            plt.text(x, y, f"{weight[y, x].item():.3f}", horizontalalignment="center", verticalalignment="center", color=text_color, fontsize=8)
    if bias is not None:
        plt.subplot(1, 2, 2)
        plt.tick_params(axis="both", which="both", top=False, bottom=False, left=False, right=False, labelbottom=False, labelleft=False)
        if highlight_mode == "value":
            plt.imshow(bias[None, None].numpy(), cmap="gray")
        elif highlight_mode == "sign":
            plt.imshow(bias[None, None].sign().numpy(), cmap="gray", vmin=-1.0, vmax=1.0)
        if highlight_mode == "value":
            text_color = "white" if bias.item() <= 0.5 else "black"
        elif highlight_mode == "sign":
            text_color = "white" if bias.sign().item() <= 0.0 else "black"
        plt.text(0, 0, f"{bias.item():.3f}", horizontalalignment="center", verticalalignment="center", color=text_color, fontsize=8)
    plt.gcf().canvas.draw()
    image = np.frombuffer(plt.gcf().canvas.tostring_argb(), dtype=np.uint8)  # pyright: ignore [reportAttributeAccessIssue]
    image = image.reshape(plt.gcf().canvas.get_width_height()[::-1] + (4,))
    image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
    plt.close()
    return numpy_to_qimg(np.roll(image, -1, axis=0))


def viz_params_pca(weights: Tensor, biases: Tensor | None, highlight_mode: Literal["value", "sign"]) -> QImage:
    tmp = weights.flatten(end_dim=1).flatten(start_dim=1)
    tmp /= tmp.mean(dim=0, keepdim=True)
    princ_comps = torch.linalg.svd(tmp, full_matrices=False)[2].T.reshape(-1, *weights.shape[2:])
    plt.subplots(weights.shape[-1], weights.shape[-2])
    for i in range(princ_comps.shape[0]):
        plt.subplot(weights.shape[-1], weights.shape[-2], i + 1)
        plt.tick_params(axis="both", which="both", top=False, bottom=False, left=False, right=False, labelbottom=False, labelleft=False)
        plt.xlabel(f"Mean: {princ_comps[i].mean().item(): .3f}\nStd: {princ_comps[i].std().item():.3f}", fontsize=8)
        if highlight_mode == "value":
            plt.imshow(princ_comps[i].numpy(), cmap="gray")
        elif highlight_mode == "sign":
            plt.imshow(princ_comps[i].sign().numpy(), cmap="gray", vmin=-1.0, vmax=1.0)
        norm_princ_comps = (princ_comps[i] - princ_comps[i].min()) / (princ_comps[i].max() - princ_comps[i].min())
        for y in range(princ_comps[i].shape[0]):
            for x in range(princ_comps[i].shape[1]):
                if highlight_mode == "value":
                    text_color = "white" if norm_princ_comps[y, x].item() <= 0.5 else "black"
                elif highlight_mode == "sign":
                    text_color = "white" if princ_comps[i, y, x].sign().item() <= 0.0 else "black"
                plt.text(x, y, f"{norm_princ_comps[y, x].item():.3f}", horizontalalignment="center", verticalalignment="center", color=text_color, fontsize=8)
    plt.tight_layout()
    plt.gcf().canvas.draw()
    image = np.frombuffer(plt.gcf().canvas.tostring_argb(), dtype=np.uint8)  # pyright: ignore [reportAttributeAccessIssue]
    image = image.reshape(plt.gcf().canvas.get_width_height()[::-1] + (4,))
    image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
    plt.close()
    return numpy_to_qimg(np.roll(image, -1, axis=0))


def update_tab(tab: Any, weight_name: str, pca: bool = False) -> None:
    weight = current_weights[weight_name + ".weight"][tab.slider_out.value(), tab.slider_in.value()]
    if weight_name + ".bias" in current_weights:
        bias = current_weights[weight_name + ".bias"][tab.slider_out.value()]
    else:
        bias = None
    tab.ele_wmoment1.setText(f"{weight.mean().item():.3f}")
    tab.ele_wmoment2.setText(f"{weight.std().item():.3f}")
    if bias is not None:
        tab.ele_bmoment1.setText(f"{bias.mean().item():.3f}")
    tab.viz.image = viz_params(weight, bias, highlight_mode=tab.highlight_mode.text())
    if pca:
        tab.viz.image = viz_params_pca(
            current_weights[weight_name + ".weight"],
            current_weights[weight_name + ".bias"] if weight_name + ".bias" in current_weights else None,
            highlight_mode=tab.highlight_mode.text(),
        )


def format_2dtensor_str(tensor: Tensor) -> str:
    result = ""
    for row in tensor:
        for entry in row:
            result += f"{entry.item(): .3f} "
        result += "\n"
    return result


def show_weight(main_window: Any) -> None:
    global current_weights
    weight_names = {x.removesuffix(".weight").removesuffix(".bias") for x in current_weights.keys()}
    weight_names = sorted(weight_names)
    weight_name, ok = QInputDialog.getItem(main_window, "Select weight", "Please select a weight to show", weight_names)
    if weight_name == "" or not ok:
        return
    try:
        ui_file = QFile(Path(__file__).parents[1] / "resources" / "tab.ui")
        ui_file.open(QIODeviceBase.OpenModeFlag.ReadOnly)
        tab = cast(Any, QUiLoader().load(ui_file))
        ui_file.close()
    except Exception:
        _ = QMessageBox.critical(main_window, "Failed to load tab UI", "General error", QMessageBox.StandardButton.Ok)
        return
    wmean = current_weights[weight_name + ".weight"].flatten(end_dim=1).mean(dim=0)
    wstd = current_weights[weight_name + ".weight"].flatten(end_dim=1).std(dim=0)
    tab.layer_wmoment1.setText(format_2dtensor_str(wmean))
    tab.layer_wmoment2.setText(format_2dtensor_str(wstd))
    bmean = current_weights[weight_name + ".bias"].mean(dim=0)
    bstd = current_weights[weight_name + ".bias"].std(dim=0)
    tab.layer_bmoment1.setText(f"{bmean.item(): .3f}")
    tab.layer_bmoment2.setText(f"{bstd.item(): .3f}")
    tab.slider_out.setMaximum(current_weights[weight_name + ".weight"].shape[0] - 1)
    tab.slider_in.setMaximum(current_weights[weight_name + ".weight"].shape[1] - 1)
    tab.highlight_value.toggled.connect(Slot()(lambda _: tab.highlight_mode.setText("value")))
    tab.highlight_sign.toggled.connect(Slot()(lambda _: tab.highlight_mode.setText("sign")))
    tab.highlight_mode.textChanged.connect(Slot()(lambda _: update_tab(tab, weight_name, tab.pca_button.isChecked())))
    tab.slider_in.valueChanged.connect(Slot()(lambda _: update_tab(tab, weight_name, tab.pca_button.isChecked())))
    tab.slider_out.valueChanged.connect(Slot()(lambda _: update_tab(tab, weight_name, tab.pca_button.isChecked())))
    tab.pca_button.toggled.connect(Slot()(lambda checked: update_tab(tab, weight_name, checked)))
    update_tab(tab, weight_name, tab.pca_button.isChecked())
    main_window.tabWidget.addTab(tab, weight_name)


def main() -> None:
    os.environ["PYSIDE_DESIGNER_PLUGINS"] = str(Path(__file__).parent)
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    application = QApplication(sys.argv)
    try:
        ui_file = QFile(Path(__file__).parents[1] / "resources" / "wvw.ui")
        ui_file.open(QIODeviceBase.OpenModeFlag.ReadOnly)
        main_window = cast(Any, QUiLoader().load(ui_file))
        ui_file.close()
    except NotImplementedError:
        print("Failed to load UI")
        return

    main_window.action_load_weights.triggered.connect(Slot()(lambda _: load_weights(main_window)))
    main_window.action_show_weight.triggered.connect(Slot()(lambda _: show_weight(main_window)))
    main_window.show()
    application.exec()
    os._exit(0)  # BUG Needed due to a bug in PySide6


if __name__ == "__main__":
    main()
