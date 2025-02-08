import sys
import numpy as np
import pyqtgraph as pg

from PyQt5.QtCore import QRect, Qt, QRectF
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QGridLayout, QApplication, QLabel, QVBoxLayout
)


class ImageViewer2DWidget(QWidget):
    def __init__(self, width=300, height=300):
        super().__init__()
        self.image = None
        self.lut = None
        self.init(width, height)

    def init(self, width, height):
        # self.rect = QRect(0, 0, width, height)  # <-- REMOVED
        # Disable auto leveling
        self.imageItem = pg.ImageItem(autoLevels=False)
        self.imageItem.setAutoDownsample(True)

        self.graphicsScene = pg.GraphicsScene()
        self.graphicsScene.addItem(self.imageItem)

        self.graphicsView = pg.GraphicsView()
        self.graphicsView.setRenderHint(QPainter.Antialiasing)
        self.graphicsView.setScene(self.graphicsScene)

        # Optionally fix the widget's size to something approximate:
        self.setFixedSize(width, height)

        layout = QHBoxLayout()
        layout.addWidget(self.graphicsView)
        self.setLayout(layout)

    def setImage(self, image):
        """
        image: 2D or 3D (with last dimension =4 for RGBA).
        """
        assert len(image.shape) in (2, 3)
        self.image = image

        # CHANGED: Set the data without autoLevels
        self.imageItem.setImage(self.image, autoLevels=False)

        # CHANGED: Use the image shape to define the bounding rectangle
        # For a 2D array, shape == (height, width).
        # For e.g. RGBA, shape == (height, width, 4), so we only take the first 2 dims.
        h, w = image.shape[:2]
        self.imageItem.setRect(QRectF(0, 0, w, h))

        # CHANGED: Lock aspect ratio (so squares remain square)
        self.graphicsView.setAspectLocked(True)

        # CHANGED: Make sure we see the entire image item
        self.graphicsView.setRange(self.imageItem.boundingRect(), padding=0.0)

        if self.lut is not None:
            self.imageItem.setLookupTable(self.lut)

    def setLevels(self, levels, update=True):
        self.imageItem.setLevels(levels, update)

    def setLUT(self, lut):
        self.lut = lut
        self.imageItem.setLookupTable(self.lut)


class ImageSlicingWidget(QWidget):
    def __init__(self, width=300, height=300):
        super().__init__()
        self.slice = 0
        self.image3D = None
        self.lut = None

        # Set up child widgets
        self.viewer = ImageViewer2DWidget(width, height)
        self.label_slice = QLabel("Slice 0/0")

        vlayout = QVBoxLayout()
        vlayout.addWidget(self.viewer)
        vlayout.addWidget(self.label_slice, alignment=Qt.AlignCenter)
        self.setLayout(vlayout)

    def setImage(self, image):
        assert len(image.shape) == 3 or len(image.shape) == 4
        self.image3D = image
        self.setSlice(self.image3D.shape[0] // 2)  # default: middle slice

    def _updateImageSlice(self):
        if self.image3D is None:
            return
        slice_index = np.clip(self.slice, 0, self.image3D.shape[0] - 1)
        self.viewer.setImage(self.image3D[slice_index])
        if self.lut is not None:
            self.viewer.setLUT(self.lut)
        total_slices = self.image3D.shape[0]
        self.label_slice.setText(f"Slice {slice_index + 1}/{total_slices}")

    def setSlice(self, slice_index):
        if self.image3D is None:
            return
        slice_index = np.clip(slice_index, 0, self.image3D.shape[0] - 1)
        self.slice = slice_index
        self._updateImageSlice()

    def getSlice(self):
        return self.slice

    def setLevels(self, levels, update=True):
        self.viewer.setLevels(levels, update)

    def setLUT(self, lut):
        self.lut = lut
        self.viewer.setLUT(lut)


class BatchViewer(QWidget):
    def __init__(self, parent=None, width=300, height=300):
        super().__init__(parent)
        self.batch = None
        self.width = width
        self.height = height
        self.slicingWidgets = {}
        self._init_gui()

    def setBatch(self, batch, lut={}):
        assert len(batch.shape) == 4 or len(batch.shape) == 5
        for v in self.slicingWidgets.values():
            self._my_layout.removeWidget(v)
            v.deleteLater()
        self.slicingWidgets = {}
        self.batch = batch

        # If lut is a single LUT array, or None, handle that
        if not isinstance(lut, dict):
            single_lut = lut
            lut = {i: single_lut for i in range(batch.shape[0])}

        num_volumes = self.batch.shape[0]
        num_col = int(np.ceil(np.sqrt(num_volumes)))
        col = 0
        row = 0

        for i in range(num_volumes):
            w = ImageSlicingWidget(self.width, self.height)
            if i in lut.keys() and lut[i] is not None:
                w.setLUT(lut[i])
            vol = self.batch[i]
            w.setImage(vol)

            # Critical: set levels from entire volume
            mn, mx = vol.min(), vol.max()
            w.setLevels([mn, mx], update=True)

            self._my_layout.addWidget(w, row, col)
            col += 1
            if col >= num_col:
                col = 0
                row += 1
            self.slicingWidgets[i] = w

    def _init_gui(self):
        self._my_layout = QGridLayout()
        self.slicingWidgets = {}
        self.setLayout(self._my_layout)
        self.setWindowTitle("Batch Viewer")

    def wheelEvent(self, event):
        delta = np.sign(event.angleDelta().y())
        modifiers = QApplication.keyboardModifiers()
        if modifiers & Qt.ControlModifier:
            step = 5
        else:
            step = 1
        for v in self.slicingWidgets.values():
            current_slice = v.getSlice()
            v.setSlice(current_slice + step * delta)


def view_batch(*args, width=500, height=500, lut={}):
    use_these = args
    if not isinstance(use_these, (np.ndarray, np.memmap)):
        use_these = list(use_these)
        for i in range(len(use_these)):
            item = use_these[i]
            try:
                import torch
                if isinstance(item, torch.Tensor):
                    item = item.detach().cpu().numpy()
            except ImportError:
                pass
            while len(item.shape) < 4:
                item = item[None]
            if item.dtype == bool:
                item = item.astype(np.uint8, copy=False)
            use_these[i] = item
        use_these = np.vstack(use_these)
    else:
        while len(use_these.shape) < 4:
            use_these = use_these[None]

    global app
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    sv = BatchViewer(width=width, height=height)
    sv.setBatch(use_these, lut)
    sv.show()

    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None and ip.has_trait('kernel'):
            ip.magic('gui qt')
        else:
            app.exec_()
    except ImportError:
        app.exec_()


if __name__ == '__main__':
    # Example usage
    # app = QApplication.instance()
    # if app is None:
    #     app = QApplication(sys.argv)
    #
    # # random data: 6 volumes, each 100 slices, 100 x 100 in-plane
    # batch = np.random.uniform(0, 3, (6, 100, 100, 100)).astype(float)
    # batch[:, 50:] *= 0.5
    #
    # # example LUT dictionary for volumes #1 and #2
    # # each LUT is Nx4 (RGBA). Values should be in [0,255] for pyqtgraph
    # lut_dict = {
    #     1: np.array([[0, 0, 0, 255],      # black
    #                  [255, 0, 0, 255],    # red
    #                  [0, 255, 0, 255],    # green
    #                  [0, 0, 255, 255]],   # blue
    #                 dtype=np.uint8),
    #     2: np.array([[255, 255, 0, 255],
    #                  [0, 255, 255, 255],
    #                  [255, 0, 255, 255],
    #                  [128, 128, 128, 255]],
    #                 dtype=np.uint8),
    # }
    #
    # viewer = BatchViewer(width=400, height=400)
    # viewer.setBatch(batch, lut_dict)
    # viewer.show()
    #
    # sys.exit(app.exec_())

    a = np.zeros((64, 64, 64))
    # a[32:42, 32:42, 32:42] = 1
    # a[37:47, 37:47, 37:47] = 2
    a[:, 32, 32] = 2
    a[:, -1, :] = 1
    a[:, :, -1] = 1
    a[:, 0, :] = 1
    a[:, :, 0] = 1
    view_batch(a, width=500, height=500)
