#!/usr/bin/env python3
import sys
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5.QtGui import QVector3D
from pyqtgraph.Qt import QtCore, QtGui

def index2position(image, metadata):
    indice = np.array(np.where(image > 0))
    ratio = np.array([[metadata['pixel_size_x'],
                      metadata['pixel_size_y'],
                      metadata['pixel_size_z']]]).T
    positions = indice * ratio[:len(indice)]
    return positions.T

def generate_random_colors(number, alpha=0.005):
    color_list = []
    for i in range(number): 
        r, g, b = np.random.random(), np.random.random(), np.random.random()
        r /= sum([r, g, b])
        g /= sum([r, g, b])
        b /= sum([r, g, b])
        color_list.append((r, g, b, alpha))
    return color_list

def generate_color_palette(number, alpha=0.01):
    red = (204./255, 0./255, 0./255, alpha)
    yellow = (204./255, 102./255, 0./255, alpha)
    green = (0./255, 204./255, 0./255, alpha)
    blue = (0./255, 0./255, 204./255, alpha)
    purple = (204./255, 0./255, 204./255, alpha)
    return [red, blue, green, yellow, purple][number % 5]

def get_label_color(labels):
    for label in labels.flatten():
        if label > 0:
            yield generate_color_palette(int(label))

def render_image(image, metadata, feature=None):
    pg.mkQApp()
    view = gl.GLViewWidget()
    view.show()
    image_positions = index2position(image, metadata)
    view.opts['center'] = QVector3D(image_positions.T[0].flatten().max() / 2, 
                                    image_positions.T[1].flatten().max() / 2, 
                                    image_positions.T[2].flatten().max() / 2)  # rotation centre of the camera
    view.opts['distance'] = image_positions.flatten().max() * 2  # distance of the camera respect to the center
    image_color = np.zeros([len(image_positions), 4]) + np.array([0.1, 0.1, 1, 0.01]) 
    point_image = gl.GLScatterPlotItem(pos=image_positions, color=image_color, pxMode=False)
    view.addItem(point_image)
    if type(feature) != type(None):
        feature = feature.T
        feature = np.array([feature[0] * metadata['pixel_size_x'],
                            feature[1] * metadata['pixel_size_y'],
                            feature[2] * metadata['pixel_size_z']])
        feature_size = np.ones(feature.shape[1]) * 4
        feature_color = np.zeros([feature.shape[1], 1]) + np.array([1, 0, 0, 1])
        point_feature = gl.GLScatterPlotItem(pos=feature.T, color=feature_color, size=feature_size, pxMode=False)
        view.addItem(point_feature)
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

def render_labels(labels, metadata):
    """
    labels.shape: (x_size, y_size, z_size)
    """
    pg.mkQApp()
    view = gl.GLViewWidget()
    view.show()
    label_positions = index2position(labels, metadata)
    view.opts['center'] = QVector3D(label_positions.T[0].flatten().max() / 2, 
                                    label_positions.T[1].flatten().max() / 2, 
                                    label_positions.T[2].flatten().max() / 2)  # rotation centre of the camera
    view.opts['distance'] = label_positions.flatten().max() * 2  # distance of the camera respect to the center
    label_color = np.array(list(get_label_color(labels)))
    point_label = gl.GLScatterPlotItem(pos=label_positions, color=label_color, pxMode=False)
    view.addItem(point_label)
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

def refresh_scatter(plot, feature, upper, lower, **kargs):
    plot.clear()
    scatter = [(p[0], p[1]) for p in feature.T if (p[2] < upper and p[2] > lower)]
    plot.setData(pos=scatter, **kargs)

def refresh_image(canvas, new_image, z):
    canvas.clear()
    canvas.setImage(new_image[z])

def annotate_maxima(image, feature):
    image = np.moveaxis(image, -1, 0)  # x,y,z ---> z,x,y
    pg.mkQApp()
    window = pg.GraphicsLayoutWidget()
    p1 = window.addPlot(row=1, col=0, rowspan=3)
    p1.setPreferredHeight(1)
    p2 = window.addPlot(row=4, col=0, rowspane=1)
    p2.setXRange(0, len(image))
    vline = pg.LineSegmentROI([[1, 0], [1, 10]], pen='r')  # vertical line
    p2.addItem(vline)
    axis = pg.ScatterPlotItem()
    canvas = pg.ImageItem()
    region = pg.LinearRegionItem()
    region.setRegion([0, 2])
    p2.addItem(region)
    feature = feature.T

    def vline_update():
        z = int(vline.pos()[0])
        z = (z > 0) * z
        z = (z < len(image)) * z + (z >= len(image) * len(image) - 1)
        refresh_image(canvas, image, z)
        lower, upper = region.getRegion()
        rw = upper - lower  # region_width
        region.setRegion([z-rw/2, z+rw/2])

    def region_update():
        lower, upper = region.getRegion()
        lower = int(lower)
        upper = int(upper)
        refresh_scatter(axis, feature, upper, lower, 
                        size=10, brush=pg.mkBrush(color=(255, 0, 0, 255)))

    refresh_scatter(axis, feature, 0, 1, size=10, brush=pg.mkBrush(color=(255, 0, 0, 255)))
    refresh_image(canvas, image, 1)

    vline.sigRegionChanged.connect(vline_update)
    region.sigRegionChanged.connect(region_update)
    p1.addItem(axis)
    p1.addItem(canvas)
    canvas.setZValue(-100)
    window.resize(800, 800)
    window.show()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

def refresh_labels(plot, labels, z, canvas, image):
    plot.clear()
    plot.addItem(canvas)
    projection = labels[z]
    pens = []
    brushs = []
    xys = [[], []]  # todo: this is stupid
    coms = []
    values = []
    for i in set(projection[projection>0].ravel()):
        xy = np.array(np.where(projection == i))
        length = xy.shape[1]
        if xy is not []:
            coms.append(np.average(xy, axis=1))
            values.append(str(i))
            xys[0] += list(xy[0])
            xys[1] += list(xy[1])
            color = generate_color_palette(int(i), alpha=0.5)
            color = np.array(color) * 255
            brush=pg.mkBrush(color=color)
            pen=pg.mkPen(color=color)
            pens += [pen] * length  # for i in range(length)]
            brushs += [brush] * length  # for i in range(length)]
    xys = np.array(xys)
    coms = np.array(coms)
    scatters = pg.ScatterPlotItem()
    scatters.setData(pos=xys.T, brush=brushs, pen=pens)
    plot.addItem(scatters)
    for value, position in zip(values, coms):
        x, y = position
        html = '<font size="12" color="white">%s</font>' % value
        text = pg.TextItem(html=html)
        text.setPos(x, y)
        plot.addItem(text)

def annotate_labels(image, labels):
    image = np.moveaxis(image, -1, 0)  # x,y,z ---> z,x,y
    labels = np.moveaxis(labels, -1, 0)  # x,y,z ---> z,x,y
    pg.mkQApp()
    window = pg.GraphicsLayoutWidget()
    p1 = window.addPlot(row=1, col=0, rowspan=3)
    p1.setPreferredHeight(1)
    p2 = window.addPlot(row=4, col=0, rowspane=1)
    p2.setXRange(0, len(image))
    vline = pg.LineSegmentROI([[1, 0], [1, 10]], pen='r')  # vertical line
    p2.addItem(vline)
    canvas = pg.ImageItem()

    def vline_update():
        z = int(vline.pos()[0])
        z = (z > 0) * z
        z = (z < len(labels)) * z + (z >= len(labels) * len(labels) - 1)
        refresh_image(canvas, image, z)
        refresh_labels(p1, labels, z, canvas, image)

    refresh_image(canvas, image, 1)
    refresh_labels(p1, labels, 1, canvas, image)
    vline.sigRegionChanged.connect(vline_update)
    p1.addItem(canvas)
    canvas.setZValue(-100)
    window.resize(800, 800)
    window.show()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

def label_scatter(positions, labels):
    pg.mkQApp()
    view = gl.GLViewWidget()
    view.show()
    view.opts['distance'] = positions.flatten().max() * 2  # distance of the camera respect to the center
    label_color = np.array(list(get_label_color(labels)))
    point_label = gl.GLScatterPlotItem(pos=positions, color=label_color, pxMode=False)
    view.addItem(point_label)
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
    
def render_convex_hull(convex_hull, metadata=None, index=0):
    """
    labels.shape: (x_size, y_size, z_size)
    """
    pg.mkQApp()
    view = gl.GLViewWidget()
    view.show()
    cvh_positions = convex_hull.points
    cvh_facets_indice = convex_hull.simplices
    view.opts['center'] = QVector3D(cvh_positions.T[0].flatten().max() / 2, 
                                    cvh_positions.T[1].flatten().max() / 2, 
                                    cvh_positions.T[2].flatten().max() / 2)  # rotation centre of the camera
    view.opts['distance'] = cvh_positions.flatten().max() * 2  # distance of the camera respect to the center
    color = generate_color_palette(index)
    cvh_mesh = gl.GLMeshItem(vertexes=cvh_positions, faces=cvh_facets_indice,
                             color=color,
                             drawEdges=True,
                             edgeColor=(1, 1, 1, 1),
                             shader='balloon')
    view.addItem(cvh_mesh)
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
