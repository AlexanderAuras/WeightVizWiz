from PySide6.QtGui import QColor, QImage, QPainter, QPaintEvent, QTransform
from PySide6.QtWidgets import QLabel, QWidget
from typing_extensions import override


class KernelView(QLabel):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.__image = None

    @property
    def image(self) -> QImage | None:
        return self.__image

    @image.setter
    def image(self, image: QImage) -> None:
        self.__image = image
        self.update()

    @override
    def paintEvent(self, arg__1: QPaintEvent) -> None:
        if self.__image is None:
            return

        scale = min(self.width() / self.__image.width(), self.height() / self.__image.height())
        width = self.__image.width() * scale
        height = self.__image.height() * scale

        painter = QPainter(self)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        painter.fillRect(self.rect(), QColor.fromString("#FFFFFFFF"))
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.setTransform(
            QTransform.fromTranslate(
                round(self.width() / 2.0 - width / 2.0),
                round(self.height() / 2.0 - height / 2.0),
            ).scale(
                1.0 / self.width() * width,
                1.0 / self.height() * height,
            )
        )
        painter.drawImage(self.rect(), self.__image)
        painter.end()
