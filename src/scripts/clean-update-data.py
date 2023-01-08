import gala.dynamics as gd
from pyia import GaiaData
from schnecke.config import galcen_frame

g = GaiaData(tbl)

# TODO: parallax zero point
# TODO: correct radial velocities.
c = g.get_skycoord()
galcen = c.transform_to(galcen_frame)
w = gd.PhaseSpacePosition(galcen.data)
Lz = w.angular_momentum()[2]

def get_r_guide():
    eilers_tbl = at.Table.read('../src/data/Eilers2019-circ-velocity.txt', format='ascii.basic')
    eilers_vcirc = InterpolatedUnivariateSpline(eilers_tbl['R'], eilers_tbl['v_c'], k=3)

# Rg =