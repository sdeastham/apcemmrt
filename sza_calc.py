import numpy as np
import datetime

def solar_declination(ordinal_date):
    n = np.floor(ordinal_date - 1)
    dec_in_rad = -np.arcsin(0.39779 * np.cos((np.pi/180.0) * (0.9855 * (n + 10) + 1.914 *
                                                              np.sin(0.98565 * (n-2) * np.pi/180.0))))
    return dec_in_rad

def cos_sza_utc(sec_utc,ordinal_date,lat_in_rad,lon_in_rad):
    # A function which operates only in numpy world
    utc_frac = sec_utc/(3600*24)
    while utc_frac > 1:
        ordinal_date += 1
        utc_frac -= 1
    hour_angle = 2.0*np.pi*(utc_frac - 0.5) + lon_in_rad
    # Need to get hour_angle into the range -pi : pi
    hour_angle = np.mod(hour_angle + np.pi,2*np.pi) - np.pi
    # NB: Ordinal date is an integer, and can be retrieved from date_utc.toordinal()
    sol_dec = solar_declination(ordinal_date)
    sin_term = np.sin(lat_in_rad) * np.sin(sol_dec)
    cos_term = np.cos(lat_in_rad) * np.cos(sol_dec) * np.cos(hour_angle)
    return sin_term + cos_term

def cos_sza(time_utc,ordinal_date,lat_in_rad,lon_in_rad):
    #hour_angle = (lon_in_rad)*12.0/np.pi # Equivalent to lon_in_deg/15.0
    # Fraction of UTC day completed
    sec_utc = (time_utc - datetime.datetime(time_utc.year,time_utc.month,time_utc.day,0,0,0)).total_seconds()
    return cos_sza_utc(sec_utc,ordinal_date,lat_in_rad,lon_in_rad)

def global_csza(t_utc,lon_grid,lat_grid):
    # Expect lon and lat to be in degrees, t_utc to be a datetime object
    ordinal_date = t_utc.toordinal() - datetime.datetime(t_utc.year,1,1,0,0,0).toordinal() # integer
    lon_in_rad = lon_grid * np.pi / 180.0
    lat_in_rad = lat_grid * np.pi / 180.0
    csza = cos_sza(t_utc,ordinal_date,lat_in_rad,lon_in_rad)
    return csza
