"""
Terra15 Treble: This example demonstrates how to plot the temporal waveform
of at a certain range of distances, real-time.
"""
import pyqtgraph as pg
import numpy as np

# from treble import acq_client

treble_address = "tcp://10.0.0.215:48000"

app = pg.mkQApp()
client = 1  # acq_client.acq_Client()
client.connect_to_server(treble_address)

plot1 = pg.PlotWidget()
plot1.setWindowTitle("Treble: " + treble_address)
plot1.show()
curve = plot1.plot()

n_frames = 20
frame_list = list(
    range(-n_frames + 1, 1, 1)
)  # negative frame numbers: longer back in time.

_, metadata = client.fetch_data_product([0], with_client_fn="return_meta_only")
dx = metadata["dx"]  # size of each sample along distance axis
nx = metadata["nx"]  # number of samples along distance axis
gauge_length = 20  # specified in [m]
dt = metadata["dT"]  # temporal resolution in [s]
nt = metadata["nT"]  # number of samples in each frame
t0 = metadata["acq_times"][0]  # time stamp of start of most recent frame (frame [0])
xaxis = np.linspace(0, n_frames * nt * dt, num=nt * n_frames)

# specify the region of interest here (start_index2 and end_index2, as positions along the fibre):
kwargs = {
    "start_index2": 160,
    "end_index2": 180,
    "gauge_length": gauge_length,
    "dx_dec": dx,
}
data, metadata = client.fetch_data_product(
    frame_list,
    timeout=20000,
    with_client_fn="convert_velocity_do_range_mean",
    client_fn_args=kwargs,
)
data = data.reshape((data.shape[0] * data.shape[1], -1)).squeeze()

# Setting up the first plot, so we can update it faster later:
xaxis = np.linspace(0, n_frames * nt * dt, num=nt * n_frames)
curve = plot1.plot(xaxis, np.flip(data))

ptr = 0


def update():
    """Update"""
    global curve, data, ptr, plot1, kwargs
    data, metadata = client.fetch_data_product(
        frame_list,
        timeout=20000,
        with_client_fn="convert_velocity_do_range_mean",
        client_fn_args=kwargs,
    )
    data = data.reshape((data.shape[0] * data.shape[1], -1)).squeeze()
    xaxis = np.linspace(0, n_frames * nt * dt, num=nt * n_frames)
    curve.setData(xaxis, np.flip(data))
    if ptr == 0:
        plot1.enableAutoRange(
            "xy", False
        )  # stop auto-scaling after the first data set is plotted
    ptr += 1


timer = pg.Qt.QtCore.QTimer()
timer.timeout.connect(update)
timer.start(160)  # 160ms, as that's the smallest frame size

if __name__ == "__main__":
    try:
        from pydev_ipython.inputhook import (
            enable_gui,
        )  # this is helpful if running with PyCharm, in "run with console" mode

        enable_gui("qt5")
    except Exception:
        pg.exec()
