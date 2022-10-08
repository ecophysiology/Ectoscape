#notes - add year to results dataframe

#------LIBRARIES------#
import numpy as np
from math import *
from scipy.interpolate import *
from matplotlib import pylab
from pylab import *
import matplotlib.pyplot as plt
from pandas import *

#-----CONSTANTS------#
SOLAR_CONSTANT = 1360. #W*m^-2
TAU = 0.7 #clear day
STEFAN_BOLTZMANN = 5.670373*10**(-8) #W*m^-2*K^-4
#ALBEDO = 0.4 #ground reflectance (albedo) of dry sandy soil ranges 0.25-0.45
#E_G = 0.9 #surface emissivity of sandy soil w/<2% organic matter approx 0.88
OMEGA = pi/12.

#-----CLASSESS------#
class Individual():
    def __init__(self,shade,latitude,longitude,elevation):
        self.MASS = 0.01 #g
        self.latitude = latitude #for Oulu, Finland 64.68, 47 Montana
        self.longitude = longitude #for Oulu, Finland 25.09, -113 Montana
        self.altitude = elevation #for Oulu, Finland 48, 1213 Montana
        self.D = 0.001 #characteristic dimension
        self.H = 0.005 #length of cylinder, meters
        self.S = shade #proportion of lizard exposed to direct solar radiation (non-shaded)
        self.A_S = 0.926 #absorptance of organism to shortwave radiation; Henwood 1975
        self.A_L = 0.965 #absorptance of organism to longwave radiation
        self.E_S = 0.97 #emissivity of organism
        self.albedo = 0.16 #ground reflectance (albedo)
        self.E_G = 0.96 #surface emissivity
        self.boundary_conductance = 1.4*0.135*sqrt(.1/self.D)
        self.maxTemps = array([13.9,14.0,17.0,22.0,26.0,31.0,36.0,36.0,34.0,32.0,26.0,19.0,14.0,13.9])
        self.minTemps = array([-5.9,-6.0,-3.0,0.0,4.0,10.0,15.0,18.0,17.0,13.0,5.0,2.0,-6.0,-5.9])
        self.maxTemps_soil = array([26.0,26.12,32.23,37.78,45.70,53.16,60.95,59.68,58.56,53.98,46.53,33.48,25.93,25.0])#dark
        self.minTemps_soil = array([-6.95,-6.99,-4.55,-2.088,1.52,6.62,12.23,17.15,16.02,11.705,3.646,-2.68,-6.25,-6.24])#averaged
        self.days = array([-16, 15, 46, 75, 106, 136, 167, 197, 228, 259, 289, 320, 350, 381])
        self.maxT = UnivariateSpline(self.days, self.maxTemps, k=3) # <- This is a fxn that takes the Julian day and predicts tha high temperature for that day using a cubic spline
        self.minT = UnivariateSpline(self.days, self.minTemps, k=3)
        self.maxT_soil = UnivariateSpline(self.days, self.maxTemps_soil, k=3) # <- This is a fxn that takes the Julian day and predicts tha high temperature for that day using a cubic spline
        self.minT_soil = UnivariateSpline(self.days, self.minTemps_soil, k=3)
        self.maxes = self.maxT(arange(365))  
        self.mins = self.minT(arange(365))
        self.maxes_soil = self.maxT_soil(arange(365))  
        self.mins_soil = self.minT_soil(arange(365))
        self.T_b = self.ground_temp(0,17.)
        self.CTmax = 40.0
        self.CTmax_water_loss = 38.0 #also the pejus temperature
        self.stTmax = 32.0
    
    def orbit_correction(self,day):
        return 1 + 2 * 0.01675 * cos((((2*pi)/365))*day)
    
    def direct_solar_radiation(self,day):
        return self.orbit_correction(day)*SOLAR_CONSTANT
        
    def f(self,day):
        return 279.575 + (0.9856 * day)
    
    def ET(self,day):
        _f = radians(self.f(day))
        return (-104.7*sin(_f)+596.2*sin(2*_f)+4.3*sin(3*_f)-12.7*sin(4*_f)-429.3*cos(_f)-2.0*cos(2*_f)+19.3*cos(3*_f))/3600.
    
    def LC(self,lon):
        return ((lon%15)*4.0)/60

    def t0(self,lc,et):
        t = 12 + lc - et
        return t

    def hour(self,t,t_zero):
        h = 15*(t-t_zero)
        return h

    def declin(self,day):
        return degrees(asin(0.39785* sin(radians(278.97 + 0.9856 * day + 1.9165 * sin(radians(356.6 + 0.9856 * day))))))
        
    def zenith(self,day,t):
        if acos(sin(radians(self.latitude))*sin(radians(self.declin(day))) + cos(radians(self.latitude))*cos(radians(self.declin(day)))*cos(radians(self.hour(t,(self.t0(self.LC(self.longitude),self.ET(day))))))) >= 0.:
            return acos(sin(radians(self.latitude))*sin(radians(self.declin(day))) + cos(radians(self.latitude))*cos(radians(self.declin(day)))*cos(radians(self.hour(t,(self.t0(self.LC(self.longitude),self.ET(day)))))))
        else:
            return 0.
            
    def m(self,day,hrs):
        p_a = 101.3*exp(-self.altitude/8200)
        if cos(self.zenith(day,hrs))>=0.:
            return p_a/(101.3*(cos(self.zenith(day,hrs))))
        else:
            return 0.
            
    def hS0(self,day,hrs):
        z = self.zenith(day,hrs)
        if cos(z)>= 0.:
            return self.direct_solar_radiation(day)*(cos(z))
        else:
            return 0.
            
    def hS(self,day, hrs, tau):
         return self.hS0(day,hrs)*tau**self.m(day,hrs)

    def diffuse_solar(self,day,hrs,tau):
        return self.hS0(day,hrs)*0.3*(1.-(TAU**self.m(day,hrs)))

    def reflected_radiation(self,day,t,tau):
        return self.albedo*self.hS(day,t,tau)
               
    def view_factor(self,zenith):
        return (1.+((4.*(self.H)*sin(radians(90.-degrees(zenith))))/(pi*(self.D))))/(4.+(4.*(self.H)/(self.D)))

    def dimensionless_temperature(self,hour):
        return 0.44-(0.46*sin(((pi/12.)*hour)+0.9))+0.11*sin(2.*(pi/12.)*hour+0.9)

    def air_temp(self,day,hour):#function for daily fluctuation
        if day < 0:
            day = 364 + day
        if hour > -1.0 and hour <= 5.:
            return self.maxes[day-1]*self.dimensionless_temperature(hour)+self.mins[day]*(1-self.dimensionless_temperature(hour))
        if hour > 5. and hour <= 14.:
            return self.maxes[day]*self.dimensionless_temperature(hour)+self.mins[day]*(1-self.dimensionless_temperature(hour))
        if hour >14 and hour <= 25.:
            if day == 364:
                return self.maxes[day]*self.dimensionless_temperature(hour)+self.mins[0]*(1-self.dimensionless_temperature(hour))
            else:
                return self.maxes[day]*self.dimensionless_temperature(hour)+self.mins[day+1]*(1-self.dimensionless_temperature(hour))
                
    def ground_temp_interpolate(self,day,hour):#function for daily fluctuation
        if day < 0:
            day = 364 + day
        if hour > -1.0 and hour <= 5.:
            return self.maxes_soil[day-1]*self.dimensionless_temperature(hour)+self.mins_soil[day]*(1-self.dimensionless_temperature(hour))
        if hour > 5. and hour <= 14.:
            return self.maxes_soil[day]*self.dimensionless_temperature(hour)+self.mins_soil[day]*(1-self.dimensionless_temperature(hour))
        if hour >14 and hour <= 25.:
            if day == 364:
                return self.maxes_soil[day]*self.dimensionless_temperature(hour)+self.mins_soil[0]*(1-self.dimensionless_temperature(hour))
            else:
                return self.maxes_soil[day]*self.dimensionless_temperature(hour)+self.mins_soil[day+1]*(1-self.dimensionless_temperature(hour))
        
    def ground_temp(self,day,hour):#function that estimates ground temperatures
        return self.temp_average(day) + 10 * exp(-0./0.08)*sin(((pi/12)*(hour-8))-0./0.08)
        
    def ground_temp_2(self,hour,minT,maxT):#function that estimates ground temperatures
        average_T = ((minT + maxT)/2.0)
        A_0 = ((maxT - minT)/2.0)
        return average_T + A_0 * exp(-0./0.08)*sin(((pi/12)*(hour-8))-0./0.08)

    def temp_average(self,day):
        T_ave = []
        for i in range(14):
            for j in range(24):
                T_ave.append(self.air_temp(day-i,j))
                #previous script: T_ave.append(self.air_temp(day-i,j)), the i was changing day to 0 to -13
        return mean(T_ave)

    def radiative_conductance(self,day,hour):
        return (4.*STEFAN_BOLTZMANN*(self.ground_temp_interpolate(day,hour)+273.15)**3.)/29.3

    def longwave_sky(self,temperature):
        return 1.0*(5.670373*10**-8 * (temperature + 273.15)**4)

    def longwave_ground(self,temperature):
        b = 5.670373*10**-8
        return self.E_G*b*(temperature+273.15)**4.

    def radiation_abs(self,day,hrs,tau,air_temperature,ground_temperature):
        return ((self.S*self.A_S)*((self.view_factor(self.zenith(day,hrs))*self.hS0(day,hrs))+(0.5*(self.diffuse_solar(day,hrs,tau)))+(0.5*(self.reflected_radiation(day,hrs,tau)))))+(0.5*(self.A_L*(self.longwave_sky(air_temperature)+self.longwave_ground(ground_temperature))))

    def operative_temperature(self,day,hour,tau,air_temperature,ground_temperature):
        return self.ground_temp_interpolate(day,hour) + (((self.radiation_abs(day,hour,tau,air_temperature,ground_temperature))-(self.E_S*STEFAN_BOLTZMANN*((273.5 + self.ground_temp_interpolate(day,hour))**4)))/(29.3*(self.boundary_conductance+self.radiative_conductance(day,hour))))

    def operative_temperature_weather_station(self,day,hour,tau,air_temperature,ground_temperature):
        radiative_conductance = (4. * STEFAN_BOLTZMANN * (air_temperature + 273.15) ** 3.) / 29.3
        return air_temperature + (((self.radiation_abs(day,hour,tau,air_temperature,ground_temperature))-(self.E_S*STEFAN_BOLTZMANN*((273.5 + air_temperature)**4)))/(29.3*(self.boundary_conductance+radiative_conductance)))

    def humid_operative_temperature(self,day,hour,Tair,elev,rh,tau,gvs_obs,wind):
        'this function calculates the humid operative temperature from Campbell and Norman 2010'
        es = 0.611*exp(((2.5*10**6)/461.5)*((1./273.15)-(1./(Tair + 273.15))))
        vpd = es - (es*(rh/100))
        windspeed = wind #m/s
        gamma = 0.000666
        characteristic_dimension = 0.7 * 0.0381 #characteristic dimension of a birch leaf
        radiative_conductance = (4. * STEFAN_BOLTZMANN * (Tair + 273.15) ** 3.) / 29.3
        gvs = gvs_obs #silver birch (Betulua verrucosa) open = 0.360, closed = 0.0059 mol m^-2 s^-1
        gva = 1.4 * 0.147*sqrt(windspeed/characteristic_dimension)
        gHa = 1.4 * 0.135*sqrt(windspeed/characteristic_dimension)
        gHr = radiative_conductance + gHa
        gv = (0.5 * gvs * gva)/(gvs+gva)
        gamma_naut = gamma * (gHr/gv)
        s = ((((17.502*240.97))*0.611*exp((17.502*Tair)/(Tair+240.97)))/(240.97+Tair)**2)/((101.3*exp(-elev/8200)))
        view_factor = 0.5*(self.zenith(day,hour)) #Cambell and Norman, page 181
        shortwave_absorptance = 0.5 # Cambell and Norman, Table 11.4
        Rabs = ((self.S*shortwave_absorptance)*((view_factor*self.hS0(day,hour))+(0.5*(self.diffuse_solar(day,hour,tau)))+(0.5*(self.reflected_radiation(day,hour,tau)))))+(0.5*(self.A_L*(self.longwave_sky(Tair)+self.longwave_ground(Tair))))
        leaf_emissivity = 0.95 #Campbell and Norman, page 273
        T_eh = Tair+(gamma_naut/(gamma_naut+s))*(((Rabs - (leaf_emissivity*(5.670373*10**-8)*((Tair+273.15)**4)))/(29.3*(radiative_conductance+gHa)))-(vpd/(gamma_naut*(101.3*exp(-elev/8200)))))
        return T_eh

    def body_temperature(self,DAY,hour,TAU,air_temperature,ground_temperature,initial_Tb):
        T_e = self.operative_temperature(DAY,hour,TAU,air_temperature,ground_temperature)
        if T_e >= initial_Tb:
            tau = exp(0.72+0.36*log(self.MASS))
            self.T_b = (exp(-1./tau) * (initial_Tb - T_e) + T_e)
        if T_e < initial_Tb:
            tau = exp(0.42+0.44*log(self.MASS))
            self.T_b = (exp(-1./tau) * (initial_Tb - T_e) + T_e)
    
    def daily_activity(self,daily_Te_list):#returns total number of hours of activity in a day
        activity_total = 0.0
        for each_hour in range(len(daily_Te_list)):
            if daily_Te_list[each_hour] >= 35.0 and daily_Te_list[each_hour] <= 40.0:
                activity_total += 1.0
        return activity_total
        
    def hourly_activity(self,Te,Te_previous,hour):#returns total number of hours of activity in a day
        if Te >= 35.0 and Te <= 40.0:
            if Te_previous < 35.0:
                slope = Te - Te_previous
                intercept = Te - (slope * hour)
                threshold_time = (35.0 - intercept)/slope
                return hour - threshold_time
            elif Te_previous >= 35.0 and Te_previous <= 40.0:
                return 1.0
            elif Te_previous > 40.0:
                slope = Te - Te_previous
                intercept = Te - (slope * hour)
                threshold_time = (40.0 - intercept)/slope
                return hour - threshold_time
        elif Te > 40.0:
            if Te_previous > 40.0:
                return 0.0
            #what if the previous hour was less than 35?
            elif Te_previous <= 40.0:
                slope = Te - Te_previous
                intercept = Te - (slope * hour)
                threshold_time = (40.0 - intercept)/slope
                return hour - threshold_time
        elif Te < 35.0:
            if Te_previous < 35.0:
                return 0.0
            elif Te_previous >= 35.0:
                slope = Te - Te_previous
                intercept = Te - (slope * hour)
                threshold_time = (35.0 - intercept)/slope
                return hour - threshold_time
    
    def calculate_previous_Te(self,day,hour,tau,minT,maxT):
        if day == 0.0 and hour == 0.0:
            return self.operative_temperature(364,23,tau,self.air_temp(364,23),self.ground_temp_interpolate(day,hour))
        elif day > 0.0 and hour == 0.0:
            return self.operative_temperature(day-1,23,tau,self.air_temp(day-1,23),self.ground_temp_interpolate(day,hour))
        else:
            return self.operative_temperature(day,hour-1,tau,self.air_temp(day,hour-1),self.ground_temp_interpolate(day,hour))

    def thermal_safety(self,Te,threshold):
        margin = threshold - Te
        activity = 0
        if Te < threshold:
            activity = 0
        else:
            activity = 1
        return [margin,activity]

    def find_sun(self,doy,hour,tau,sun_state):
        'a function that determines whether the sun has risen or set'
        times = np.arange(hour-1,hour+0.01,0.01)
        for i in range(len(times)):
            sun = self.hS(doy,hour,tau)
            if sun_state == 'sunrise':
                if sun > 0:
                    return times[i]
            elif sun_state == 'sunset':
                if sun == 0:
                    return times[i]

    def RCP_60(self,temp_sd,doy):
        if doy >= 0 or doy <= 31:
            day = np.random.normal(2.1,(temp_sd + temp_sd*np.random.uniform(-0.25,0.25)))
            night = np.random.normal(6.0, (temp_sd + temp_sd * np.random.uniform(-0.25,0.25)))
        elif doy >= 32 or doy <= 59:
            day = np.random.normal(1.4,(temp_sd + temp_sd*np.random.uniform(-0.25,0.25)))
            night = np.random.normal(5.1, (temp_sd + temp_sd * np.random.uniform(-0.25,0.25)))
        elif doy >= 60 or doy <= 91:
            day = np.random.normal(0,(temp_sd + temp_sd*np.random.uniform(-0.25,0.25)))
            night = np.random.normal(3.2, (temp_sd + temp_sd * np.random.uniform(-0.25,0.25)))
        elif doy >= 92 or doy <= 121:
            day = np.random.normal(1.3, (temp_sd + temp_sd * np.random.uniform(-0.25, 0.25)))
            night = np.random.normal(3.8, (temp_sd + temp_sd * np.random.uniform(-0.25,0.25)))
        elif doy >= 122 or doy <= 152:
            day = np.random.normal(1.8, (temp_sd + temp_sd * np.random.uniform(-0.25, 0.25)))
            night = np.random.normal(2.3, (temp_sd + temp_sd * np.random.uniform(-0.25,0.25)))
        elif doy >= 153 or doy <= 182:
            day = np.random.normal(3.3, (temp_sd + temp_sd * np.random.uniform(-0.25, 0.25)))
            night = np.random.normal(4.1, (temp_sd + temp_sd * np.random.uniform(-0.25,0.25)))
        elif doy >= 183 or doy <= 213:
            day = np.random.normal(5.5, (temp_sd + temp_sd * np.random.uniform(-0.25, 0.25)))
            night = np.random.normal(1.7, (temp_sd + temp_sd * np.random.uniform(-0.25, 0.25)))
        elif doy >= 214 or doy <= 244:
            day = np.random.normal(5.4, (temp_sd + temp_sd * np.random.uniform(-0.25, 0.25)))
            night = np.random.normal(2.1, (temp_sd + temp_sd * np.random.uniform(-0.25,0.25)))
        elif doy >= 245 or doy <= 274:
            day = np.random.normal(3.9, (temp_sd + temp_sd * np.random.uniform(-0.25, 0.25)))
            night = np.random.normal(1.9, (temp_sd + temp_sd * np.random.uniform(-0.25,0.25)))
        elif doy >= 275 or doy <= 305:
            day = np.random.normal(2.4, (temp_sd + temp_sd * np.random.uniform(-0.25, 0.25)))
            night = np.random.normal(2.3, (temp_sd + temp_sd * np.random.uniform(-0.25,0.25)))
        elif doy >= 306 or doy <= 335:
            day = np.random.normal(2.9, (temp_sd + temp_sd * np.random.uniform(-0.25, 0.25)))
            night = np.random.normal(4.0, (temp_sd + temp_sd * np.random.uniform(-0.25,0.25)))
        else:
            day = np.random.normal(3.4, (temp_sd + temp_sd * np.random.uniform(-0.25, 0.25)))
            night = np.random.normal(2.9, (temp_sd + temp_sd * np.random.uniform(-0.25,0.25)))
        return [day,night]

    def day_or_night(self,day,hour,tau,temperatures):
        sun = self.hS(day,hour,tau)
        if sun > 0:
            return temperatures[0]
        else:
            return temperatures[1]


###############################
#   LEAF VALIDATION - WOODS   #
###############################
#"""
#ENVIRONMENT
weather_station = pandas.read_csv('Manuscripts/click_beetle/environmental_data/validation_woods.csv')

#RESULTS DATAFRAME
results = pandas.DataFrame(columns = ['shade','RH','conductance','doy','hour','Te','Tleaf','air_temp'])

#SITE
latitude = 47.0
longitude = -113.0
elevation = 1230.0

for i in range(600):
    for time_step in range(len(weather_station)):
        if weather_station['hour'][time_step] > 9:
            shade = np.random.uniform(0, 1)
        else:
            shade = np.random.uniform(0, 0.25)
        rh = np.random.normal(50,20)
        winds = np.random.uniform(0.1,0.25)
        conductance = np.random.uniform(0.4,0.005)
        ectotherm = Individual(shade,latitude,longitude,elevation)
        Tair = np.random.normal(weather_station['Tair'][time_step],weather_station['sd'][time_step])
        Tleaf = ectotherm.humid_operative_temperature(weather_station['doy'][time_step],weather_station['hour'][time_step],Tair,ectotherm.altitude,rh,TAU,conductance,winds)
        Te = ectotherm.operative_temperature_weather_station(weather_station['doy'][time_step],weather_station['hour'][time_step],TAU,Tair,Tleaf)
        delta_leaf = Tleaf-Tair
        dataframe = pandas.DataFrame([[ectotherm.S,rh,conductance,weather_station['doy'][time_step],weather_station['hour'][time_step],Te,Tleaf,Tair]],columns = ['shade','RH','conductance','doy','hour','Te','Tleaf','air_temp'])
        results = results.append(dataframe)
results.to_csv('Manuscripts/click_beetle/Ectoscape/results/leaf_validation_woods.csv',index = False)
#"""

#################################################
#   LEAF VALIDATION - WOODS - NO MORNING SHADE  #
#################################################
#"""
#ENVIRONMENT
weather_station = pandas.read_csv('Manuscripts/click_beetle/environmental_data/validation_woods.csv')

#RESULTS DATAFRAME
results = pandas.DataFrame(columns = ['shade','RH','conductance','doy','hour','Te','Tleaf','air_temp'])

#SITE
latitude = 47.0
longitude = -113.0
elevation = 1230.0

for i in range(600):
    for time_step in range(len(weather_station)):
        shade = np.random.uniform(0, 1)
        rh = np.random.normal(50,20)
        winds = np.random.uniform(0.1,0.25)
        conductance = np.random.uniform(0.4,0.005)
        ectotherm = Individual(shade,latitude,longitude,elevation)
        Tair = np.random.normal(weather_station['Tair'][time_step],weather_station['sd'][time_step])
        Tleaf = ectotherm.humid_operative_temperature(weather_station['doy'][time_step],weather_station['hour'][time_step],Tair,ectotherm.altitude,rh,TAU,conductance,winds)
        dataframe = pandas.DataFrame([[ectotherm.S,rh,conductance,weather_station['doy'][time_step],weather_station['hour'][time_step],Te,Tleaf,Tair]],columns = ['shade','RH','conductance','doy','hour','Te','Tleaf','air_temp'])
        results = results.append(dataframe)
results.to_csv('Manuscripts/click_beetle/Ectoscape/results/leaf_validation_woods_nomorningshade.csv',index = False)
#"""
##################################
#   CLICK BEETLE Te VALIDATION   #
##################################
#"""

#ENVIRONMENT
weather_station = pandas.read_csv('Manuscripts/click_beetle/environmental_data/validation.csv')

#RESULTS DATAFRAME
results = pandas.DataFrame(columns = ['shade','RH','conductance','doy','hour','Te','Tleaf','air_temp','TSM_CTmax','Active_CTmax','TSM_CTmax_water_loss','Active_CTmax_water_loss','TSM_stTmax','Active_StTmax'])

#SITE
latitude = 64.68
longitude = 25.09
elevation = 48.0

shade = [1.0,0.0]
RHs = [70.0]
conductances = [0.360]
for i in range(1):
    for shade_cover in range(len(shade)):
        for rh in range(len(RHs)):
            for conductance in range(len(conductances)):
                for time_step in range(len(weather_station)):
                    ectotherm = Individual(shade[shade_cover],latitude,longitude,elevation)
                    winds = 0.1 #np.random.uniform(0.1, 0.25)
                    Tair = weather_station['TA_PT1H_AVG'][time_step]
                    Tleaf = ectotherm.humid_operative_temperature(weather_station['doy'][time_step],weather_station['hour'][time_step],Tair,ectotherm.altitude,RHs[rh],TAU,conductances[conductance],winds)
                    Te = ectotherm.operative_temperature_weather_station(weather_station['doy'][time_step],weather_station['hour'][time_step],TAU,Tair,Tleaf)
                    CTmax = ectotherm.thermal_safety(Te,ectotherm.CTmax)
                    CTmax_water_loss = ectotherm.thermal_safety(Te,ectotherm.CTmax_water_loss)
                    stTmax = ectotherm.thermal_safety(Te,ectotherm.stTmax)
                    dataframe = pandas.DataFrame([[ectotherm.S,RHs[rh],conductances[conductance],weather_station['doy'][time_step],weather_station['hour'][time_step],Te,Tleaf,Tair,CTmax[0],CTmax[1],CTmax_water_loss[0],CTmax_water_loss[1],stTmax[0],stTmax[1]]],columns = ['shade','RH','conductance','doy','hour','Te','Tleaf','air_temp','TSM_CTmax','Active_CTmax','TSM_CTmax_water_loss','Active_CTmax_water_loss','TSM_stTmax','Active_StTmax'])
                    results = results.append(dataframe)
results.to_csv('Manuscripts/click_beetle/Ectoscape/results/te_validation.csv',index = False)
#"""
#######################
#   CURRENT WEATHER   #
#######################
#"""
#Environment
weather_station = pandas.read_csv('Manuscripts/click_beetle/weather_station_data/weather_station_data.csv') #for current weather

#Dataframe
results = pandas.DataFrame(columns = ['shade','RH','conductance','doy','hour','Te','Tleaf','air_temp','TSM_CTmax','Active_CTmax','TSM_CTmax_water_loss','Active_CTmax_water_loss','TSM_stTmax','Active_StTmax'])

#SITE
latitude = 64.68
longitude = 25.09
elevation = 48.0

shade = [0.5]
for shade_cover in range(len(shade)):
    for time_step in range(len(weather_station)):
        ectotherm = Individual(shade[shade_cover],latitude,longitude,elevation)
        winds = np.random.uniform(0.1, 0.25)
        conductance = np.random.uniform(0.4, 0.005)
        Tair = weather_station['TA_PT1H_AVG'][time_step]
        RH = weather_station['RH_PT1H_AVG'][time_step]
        Tleaf = ectotherm.humid_operative_temperature(weather_station['doy'][time_step],weather_station['hour'][time_step],weather_station['TA_PT1H_AVG'][time_step],ectotherm.altitude,RH,TAU,conductance,winds)
        Te = ectotherm.operative_temperature_weather_station(weather_station['doy'][time_step],weather_station['hour'][time_step],TAU,weather_station['TA_PT1H_AVG'][time_step],Tleaf)
        CTmax = ectotherm.thermal_safety(Te,ectotherm.CTmax)
        CTmax_water_loss = ectotherm.thermal_safety(Te,ectotherm.CTmax_water_loss)
        stTmax = ectotherm.thermal_safety(Te,ectotherm.stTmax)
        dataframe = pandas.DataFrame([[ectotherm.S,RH,conductance,weather_station['doy'][time_step],weather_station['hour'][time_step],Te,Tleaf,Tair,CTmax[0],CTmax[1],CTmax_water_loss[0],CTmax_water_loss[1],stTmax[0],stTmax[1]]],columns = ['shade','RH','conductance','doy','hour','Te','Tleaf','air_temp','TSM_CTmax','Active_CTmax','TSM_CTmax_water_loss','Active_CTmax_water_loss','TSM_stTmax','Active_StTmax'])
        results = results.append(dataframe)
results.to_csv('Manuscripts/click_beetle/Ectoscape/results/Oulu_Current_Climate.csv',index = False)
#"""
#######################
#    FUTURE WEATHER   #
#######################
#"""
#ENVIRONMENT
weather_station = pandas.read_csv('Manuscripts/click_beetle/weather_station_data/weather_station_data_DOY_HR_summary.csv') #future warming scenario RCP60
weather_station_rh = pandas.read_csv('Manuscripts/click_beetle/weather_station_data/rh_station_data_DOY_HR_summary.csv') #future warming scenario RCP60
doy_sd = pandas.read_csv('Manuscripts/click_beetle/weather_station_data/weather_station_data_DOY_summary.csv') #sd for each day

#RESULTS DATAFRAME
results = pandas.DataFrame(columns = ['shade','RH','conductance','doy','hour','Te','Tleaf','air_temp','TSM_CTmax','Active_CTmax','TSM_CTmax_water_loss','Active_CTmax_water_loss','TSM_stTmax','Active_StTmax'])

#SITE
latitude = 64.68
longitude = 25.09
elevation = 48.0

shade = [0.5] #1 = full sun
for i in range(25):
    for shade_cover in range(len(shade)):
        ectotherm = Individual(shade[shade_cover],latitude, longitude, elevation)
        for doy in range(len(doy_sd)):
            temperature_change = ectotherm.RCP_60(doy_sd['sd'][doy],doy+1)
            for hour in range(24):
                winds = np.random.uniform(0.1, 0.25)
                conductance = np.random.uniform(0.4, 0.005)
                RH = weather_station_rh['meanRH'][(doy)*24+hour]
                Tair = weather_station['meanTa'][(doy)*24+hour]
                Tchange_RCP60 = ectotherm.day_or_night(doy+1,hour,TAU,temperature_change)
                Tair_RCP60 = Tair + Tchange_RCP60
                Tleaf = ectotherm.humid_operative_temperature(doy,hour,Tair_RCP60,ectotherm.altitude,RH,TAU,conductance,winds)
                Te = ectotherm.operative_temperature_weather_station(doy,hour,TAU,Tair_RCP60,Tleaf)
                CTmax = ectotherm.thermal_safety(Te,ectotherm.CTmax)
                CTmax_water_loss = ectotherm.thermal_safety(Te,ectotherm.CTmax_water_loss)
                stTmax = ectotherm.thermal_safety(Te,ectotherm.stTmax)
                dataframe = pandas.DataFrame([[ectotherm.S,RH,conductance,doy,hour,Te,Tleaf,Tair_RCP60,CTmax[0],CTmax[1],CTmax_water_loss[0],CTmax_water_loss[1],stTmax[0],stTmax[1]]],columns = ['shade','RH','conductance','doy','hour','Te','Tleaf','air_temp','TSM_CTmax','Active_CTmax','TSM_CTmax_water_loss','Active_CTmax_water_loss','TSM_stTmax','Active_StTmax'])
                results = results.append(dataframe)
results.to_csv('Manuscripts/click_beetle/Ectoscape/results/RCP60_hourly.csv',index = False)
#"""
