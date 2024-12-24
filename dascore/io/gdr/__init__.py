"""
Support for the Geothermal Data Repository (gdr) h5 format.

The gdr format is a combination of prodml and the Earthscope DMC's meta
data spec. It houses many data sets, not just DFOS.

Find more information here: https://gdr.openei.org/. Information regarding
the DAS format can be found here: https://gdr.openei.org/das_data_standard
"""

from .core import GDR_V1
