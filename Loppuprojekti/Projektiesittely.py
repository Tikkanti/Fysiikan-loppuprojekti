import streamlit as st

import pandas as pd
import folium
import numpy as np
import matplotlib.pyplot as plt
from streamlit_folium import st_folium

df_askeldata = pd.read_csv('./data/Askeldata.csv')
df_gps = pd.read_csv('./data/Gps.csv')

#Askelmäärä laskettuna suodatetusta kiihtyysdatasta
from scipy.signal import butter,filtfilt
def butter_lowpass_filter(data, cutoff, nyq, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def butter_highpass_filter(data, cutoff,  nyq, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y

data = df_askeldata['Linear Acceleration y (m/s^2)']
T_tot = df_askeldata['Time (s)'].max() #Koko datan pituus
n = len(df_askeldata['Time (s)']) #Datapisteiden lukumäärä
fs = n/T_tot #Näytteenottotaajus, OLETETAAN VAKIOKSI
nyq = fs/2 #Nyqvistin taajuus, suurin taajuus, joka datasta voidaan havaita
order = 3
cutoff = 1/0.4 #Cut-off taajuus, tätä suuremmat taajuuden alipäästösuodatin poista datasta
#Cut-off -taajuuden tulee olla riittävän pieni, jotta data yleensäkin suodattuu
#Cut-off -taajuuden ei tule olla niin pieni, että se suodattaisi pois askelia
data_filt = butter_lowpass_filter(data, cutoff, nyq, order)

jaksot_suodatettu = 0
for i in range(n-1):
    if data_filt[i]/data_filt[i+1] < 0: #True jos arvoilla data_filt[i] ja data_filt[i+1] on eri etumerkki --> nollan ylitys
        jaksot_suodatettu = jaksot_suodatettu + 1/2


#Askelmäärä laskettuna Fourier-analyysin avulla
signal = df_askeldata['Linear Acceleration y (m/s^2)']
t = df_askeldata['Time (s)'] # Aika alkaa nollasta, sekunteina
N = len(signal) # Havaintojen määrä
dt = np.max(t)/N # Näytteenottoväli (oletetaan vakioksi)
fourier = np.fft.fft(signal,N) # Fourier-muunnos
psd = fourier*np.conj(fourier)/N # Tehospektri
freq = np.fft.fftfreq(N,dt) # Taajuudet
L = np.arange(1,int(N/2)) # Negatiivisten ja nollataajuuksien rajaus
f_max = freq[L][psd[L] == np.max(psd[L])][0] #Kävelymittauksen kiihtyvyyden y-komponentin tehospektrin suurinta tehoa vastaava taajuus.
T = 1/f_max #Askeleeseen kuluva aika, eli jaksonaika (oletetaan, että dominoiva taajuus on askeltaajuus)
jaksot_fourier = f_max*np.max(t) # Askelmäärä. Voi laskea myös np.max(t)/T

#Suodatetun kiihtyvyysdatan piirtäminen
filtAccFig, ax = plt.subplots(figsize=(70,20))
plt.plot(df_askeldata['Time (s)'],data_filt,label = 'suodatettu data')
plt.grid()
plt.legend()
filtAccFig.suptitle("Suodatettu kiihtyvyysdata", fontsize=30)
plt.show()

#Tehospektrin piirtäminen
pwSpFig, ax = plt.subplots(figsize=(15,6))
plt.plot(freq[L],psd[L].real)
plt.xlabel('Taajuus [Hz] = [1/s]')
plt.ylabel('Teho')
plt.axis([0,10,0,8000])
filtAccFig.suptitle("Tehospektri", fontsize=30)
plt.show()


#Karttakuvan piirtäminen
lat1 = df_gps['Latitude (°)'].mean() #Latitudin keskiarvo
long1 = df_gps['Longitude (°)'].mean() #Longitudin keskiarvo
my_map = folium.Map(location = [lat1, long1], zoom_start = 15) # Luodaan kartta
folium.PolyLine(df_gps[['Latitude (°)', 'Longitude (°)']], color = 'red', weight = 3).add_to(my_map)



#Lasketaan kuljettu matka
from math import radians, cos, sin, asin, sqrt
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r

df_gps['Distance_calc'] = np.zeros(len(df_gps))

for i in range(len(df_gps)-1):
    lon1 = df_gps['Longitude (°)'][i]
    lon2 = df_gps['Longitude (°)'][i+1]
    lat1 = df_gps['Latitude (°)'][i]
    lat2 = df_gps['Latitude (°)'][i+1]
    df_gps.loc[i+1,'Distance_calc'] = haversine(lon1, lat1, lon2, lat2)
    total_distance_km = df_gps['Distance_calc'].sum()
    total_distance_m = total_distance_km * 1000



#Lasketaan keskinopeus
total_time_s = df_gps['Time (s)'].iloc[-1] - df_gps['Time (s)'].iloc[0]
avg_speed_ms = round((total_distance_km * 1000) / total_time_s, 2)


#Lasketaan askelpituus
step_length = total_distance_m / jaksot_suodatettu * 100






st.title('Lyhyen lenkin havainnot')
st.write('Askelmäärä laskettuna suodatuksen avulla: ',int(jaksot_suodatettu), ' askelta')
st.write('Askelmäärä laskettuna Fourier-muunnoksen avulla: ',int(jaksot_fourier), ' askelta')
st.write('Keskinopeus: ', avg_speed_ms, ' m/s')
st.write('Kokonaismatka: ',int(total_distance_m),' metriä')
st.write('Askelpituus: ', int(step_length),' cm')
st.subheader('Suodatettu kiihtyvyysdata')
st.pyplot(filtAccFig)
st.subheader('Askelmittauksen Tehospektri')
st.pyplot(pwSpFig)
st.subheader('Karttakuva matkasta')
st_folium(my_map, width=700, height=500)

