from enum import Enum


class Zone(Enum):
    pass    

class Region(Enum):
    Taiwan_Datacenter = 1
    Hong_Kong_Datacenter = 2
    Mainland_China_Datacenter = 3
    Japan_Datacenter = 4
    Korea_Datacenter = 5
    India_Datacenter = 6
    Singapore_Datacenter = 7
    Indonesia_Datacenter = 8
    Australia_Datacenter = 9
    Great_Britain_Datacenter = 10
    Germany_Datacenter = 11
    Madrid_Spain_Datacenter = 12
    Finland_Datacenter = 13
    Poland_Datacenter = 14
    Belgium_Datacenter = 15
    Netherlends_Datacenter = 16
    France_Datacenter = 17
    Italy_Datacenter = 18
    Switzerland_Datacenter = 19
    Canada_Ontario_Datacenter = 20
    Canada_Montreal_Datacenter = 21
    Brazil_Datacenter = 22
    Chile_Dataceter = 23
    #below are all US datacenters


class Step_Type(Enum):
    Hour = 1
    Day = 2
    Half_Hour = 3
    Month = 4
    Year = 5
    Auto = 6

step_to_m = {Step_Type.Auto: 1, Step_Type.Day: 7, Step_Type.Month: 12, Step_Type.Year: 1, Step_Type.Hour: 24, Step_Type.Half_Hour: 48}


step_type_to_neural_prophet_freq = {
    Step_Type.Day: "D",
    Step_Type.Half_Hour: "30min",
    Step_Type.Month: "M",
    Step_Type.Year: "Y",
    Step_Type.Hour: "D",
    Step_Type.Auto: "auto"
}


zones_to_region = {


}

