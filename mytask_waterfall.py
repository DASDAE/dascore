import dascore as dc

patch = dc.get_example_patch("dispersion_event")
patch.viz.waterfall(show=True, cmap="bone", scale=0.1)
