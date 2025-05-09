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


def kernel_to_qimage(kernel: Tensor, highlight_mode: Literal["value", "sign"]) -> QImage:
    plt.figure()
    plt.subplot(111)
    if highlight_mode == "value":
        plt.imshow(kernel.numpy(), cmap="gray")
    elif highlight_mode == "sign":
        plt.imshow(kernel.sign().numpy(), cmap="gray")
    norm_kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())
    for y in range(kernel.shape[0]):
        for x in range(kernel.shape[1]):
            text_color = "white" if norm_kernel[y, x].item() <= 0.5 else "black"
            plt.text(x, y, f"{kernel[y, x]:.3f}", horizontalalignment="center", verticalalignment="center", color=text_color, fontsize=8)
    plt.gcf().canvas.draw()
    image = np.frombuffer(plt.gcf().canvas.tostring_argb(), dtype=np.uint8)  # pyright: ignore [reportAttributeAccessIssue]
    image = image.reshape(plt.gcf().canvas.get_width_height()[::-1] + (4,))
    image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
    plt.close()
    return numpy_to_qimg(np.roll(image, -1, axis=0))


def update_tab(tab: Any, weight_name: str) -> None:
    kernel = current_weights[weight_name][tab.slider_out.value(), tab.slider_in.value()]
    tab.kernel_moment1.setText(f"{kernel.mean().item():.3f}")
    tab.kernel_moment2.setText(f"{kernel.var().item():.3f}")
    tab.kernel.image = kernel_to_qimage(kernel, highlight_mode=tab.highlight_mode.text())


def format_2dtensor_str(tensor: Tensor) -> str:
    result = ""
    for row in tensor:
        for entry in row:
            result += f"{entry.item(): .3f} "
        result += "\n"
    return result


def show_weight(main_window: Any) -> None:
    global current_weights
    weight_names = list(current_weights.keys())
    weight_names = [name for name in weight_names if current_weights[name].ndim == 4]
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
    mean = current_weights[weight_name].flatten(end_dim=1).mean(dim=0)
    std = current_weights[weight_name].flatten(end_dim=1).std(dim=0)
    tab.layer_moment1.setText(format_2dtensor_str(mean))
    tab.layer_moment2.setText(format_2dtensor_str(std))
    tab.slider_out.setMaximum(current_weights[weight_name].shape[0] - 1)
    tab.slider_in.setMaximum(current_weights[weight_name].shape[1] - 1)
    tab.highlight_value.toggled.connect(Slot()(lambda _: tab.highlight_mode.setText("value")))
    tab.highlight_sign.toggled.connect(Slot()(lambda _: tab.highlight_mode.setText("sign")))
    tab.highlight_mode.textChanged.connect(Slot()(lambda _: update_tab(tab, weight_name)))
    tab.slider_in.valueChanged.connect(Slot()(lambda _: update_tab(tab, weight_name)))
    tab.slider_out.valueChanged.connect(Slot()(lambda _: update_tab(tab, weight_name)))
    update_tab(tab, weight_name)
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
