# Copyright 2017 Fabian Isensee
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from PyQt5.QtCore import QRect
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QGridLayout, QApplication
import pyqtgraph as pg
import sys
import numpy as np


class ImageViewer2DWidget(QWidget):
    def __init__(self, width=300, height=300):
        super().__init__()
        self.image = None
        self.lut = None
        self.init(width, height)

    def init(self, width, height):
        self.rect = QRect(0, 0, width, height)
        self.imageItem = pg.ImageItem()
        self.imageItem.setImage(None)

        self.graphicsScene = pg.GraphicsScene()
        self.graphicsScene.addItem(self.imageItem)

        self.graphicsView = pg.GraphicsView()
        self.graphicsView.setRenderHint(QPainter.Antialiasing)
        self.graphicsView.setScene(self.graphicsScene)

        layout = QHBoxLayout()
        layout.addWidget(self.graphicsView)
        self.setLayout(layout)
        self.setMaximumSize(width, height)
        self.setMinimumSize(width - 10, height - 10)

    def setImage(self, image):
        assert len(image.shape) == 2 or len(image.shape) == 3
        if len(image.shape) == 4:
            assert image.shape[-1] == 4
        self.image = image
        self.imageItem.setImage(self.image)
        self.imageItem.setRect(self.rect)
        if self.lut is not None:
            self.imageItem.setLookupTable(self.lut)

    def setLevels(self, levels, update=True):
        self.imageItem.setLevels(levels, update)

    def setLUT(self, lut):
        self.lut = lut


class ImageSlicingWidget(ImageViewer2DWidget):
    def __init__(self, width=300, height=300):
        self.slice = 0
        super().__init__(width, height)

    def setImage(self, image):
        assert len(image.shape) == 3 or len(image.shape) == 4
        if len(image.shape) == 4:
            assert image.shape[-1] == 4
        self.image3D = np.array(image)
        self._updateImageSlice()

    def _updateImageSlice(self):
        self.imageItem.setImage(self.image3D[self.slice])
        self.imageItem.setRect(self.rect)
        if self.lut is not None:
            self.imageItem.setLookupTable(self.lut)

    def setSlice(self, slice):
        slice = np.max((0, slice))
        slice = np.min((slice, self.image3D.shape[0] - 1))
        self.slice = slice
        self._updateImageSlice()

    def getSlice(self):
        return self.slice


class BatchViewer(QWidget):
    def __init__(self, parent=None, width=300, height=300):
        super().__init__(parent)
        self.batch = None
        self.width = width
        self.height = height
        self.slicingWidgets = {}
        self._init_gui()

    def setBatch(self, batch, lut={}):
        assert len(batch.shape) == 4
        batch = np.copy(batch)
        for v in self.slicingWidgets.values():
            self._my_layout.removeWidget(v)
        self.slicingWidgets = {}

        if not isinstance(lut, dict):
            lut = {i: lut for i in range(self.batch.shape[0])}

        for b in range(batch.shape[0]):
            mn = batch[b].min()
            mx = batch[b].max()
            if batch.dtype in (int, np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64):
                vals = np.linspace(mn, mx, num=mx - mn + 1, dtype=batch.dtype)
            else:
                vals = np.linspace(mn, mx, num=255, dtype=batch.dtype)
            max_allowed = np.prod(batch[b].shape, dtype=np.int64) // 10
            vals = vals[:min(max_allowed, len(vals))]
            for slice in range(batch[b].shape[0]):
                batch[b, slice].ravel()[:len(vals)] = vals

        self.batch = batch
        num_col = int(np.ceil(np.sqrt(self.batch.shape[0])))
        col = 0
        row = 0
        for i in range(self.batch.shape[0]):
            w = ImageSlicingWidget(self.width, self.height)
            if lut is not None and i in lut.keys():
                w.setLUT(lut[i])
            w.setImage(self.batch[i])
            # w.setLevels([self.batch[i].min(), self.batch[i].max()])
            w.setSlice(self.batch.shape[1] // 2)
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

    def wheelEvent(self, QWheelEvent):
        for v in self.slicingWidgets.values():
            offset = np.sign(QWheelEvent.angleDelta().y())
            v.setSlice(v.getSlice() + np.sign(offset))


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
            use_these[i] = item
        use_these = np.concatenate(use_these, 0)
    else:
        while len(use_these.shape) < 4:
            use_these = use_these[None]

    global app
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    sv = BatchViewer(width=width, height=height)
    sv.setBatch(use_these, lut)
    sv.show()
    app.exit(app.exec_())


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    global app
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    sv = BatchViewer()
    batch = np.random.uniform(0, 3, (6, 100, 100, 100)).astype(int)
    lut = {2: np.array([[0, 0.5, 0, 1], [0, 0, 0.5, 1], [0.5, 0, 0, 1], [0.5, 0.5, 0, 1]]) * 255,
           1: np.array([[1, 0.5, 0, 1], [0, 0, 0.5, 1], [0.5, 0, 0, 1], [0.5, 0.5, 0, 1]]) * 255}
    sv.setBatch(batch, lut)
    sv.show()
    app.exec_()
    app.deleteLater()
    sys.exit()
    # IPython.embed()

'''class SliceViewer(QtGui.QWidget):
    def __init__(self):
        super(SliceViewer, self).__init__()
        self.initUI()

    def wheelEvent(self, event):
        for v in [self.imageViewer, self.imageViewer2]:
            v.setSlice(v.getSlice() + event.delta()/120)

    def initUI(self):
        image1 = np.random.uniform(-0.5, 255., (100, 100, 100)).astype(np.float32)
        image2 = np.random.uniform(-100., 255., (100, 100, 100)).astype(np.float32)

        self.imageViewer = ImageSlicingWidget()
        self.imageViewer2 = ImageSlicingWidget()

        self.imageViewer.setImage(image1)
        self.imageViewer2.setImage(image2)

        hLayout = QtGui.QHBoxLayout()
        hLayout.addWidget(self.imageViewer)
        hLayout.addWidget(self.imageViewer2)
        hLayout.addStretch()

        self.setLayout(hLayout)

        #self.setGeometry(0, 0, 1200, 1200)
        self.setWindowTitle('QtGui.QCheckBox')

        self.show()'''
