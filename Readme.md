# BatchViewer
Simple, but effective tool to visualize 3D data with color channels. I use it mainly to tune data augmentation for deep 
learning and to quickly visualize data.
It does not have much functionality. 

## Installation
* clone the repository
* go into the repository and install with ```pip install --upgrade .```

## How to use
Only works with 4D numpy arrays. Axes must be (c, x, y, z) where c is the color channel.

Run:

```from batchviewer import view_batch```

```view_batch(data, width=300, height=300)```

Note that width and height controls the size (in pixels) of each window. If the aspect ratio does not match the 
datas aspect ratio then the data is cropped. You can scroll through the 3D volume by using the mouse wheel. 
All color channels are synced. Scrolling with the mouse on one of the images will scroll 2 slices, otherwise 
just one. Panning and zooming not suported.