"""A script to make the vizualization figure of the DASCore paper."""
import matplotlib.pyplot as plt

import dascore as dc

# setup matplotlib figure/axis
_, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# load example patch
patch = dc.get_example_patch("example_event_2")

# sub-select only center channels
sub_patch = patch.select(distance=(650, 750))

# plot waterfall
patch.viz.waterfall(ax=ax1, scale=0.5)
# plot wiggle
sub_patch.viz.wiggle(ax=ax2, scale=0.5)

# Add subplot labels
ax1.text(0.01, 0.99, "A", ha="left", va="top", transform=ax1.transAxes, size=24)
ax2.text(0.01, 0.99, "B", ha="left", va="top", transform=ax2.transAxes, size=24)

plt.tight_layout()
plt.savefig("waterfalls_and_wiggles.pdf")
plt.show()
