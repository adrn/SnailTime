import astropy.coordinates as coord
import astropy.units as u

# Same values from Hunt+2022
galcen_frame = coord.Galactocentric(
    galcen_distance=8.275*u.kpc,
    galcen_v_sun=[8.4, 251.8, 8.4] * u.km/u.s
)
