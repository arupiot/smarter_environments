# This connects to the mfi mPower power adapter, periodically reads power consumption data from the ports and publishes it to influxdb

# Written by: Ashrant Aryal, aaryal@usc.edu
# REST API syntax from: https://community.ubnt.com/t5/mFi/mPower-mFi-Switch-and-mFi-In-Wall-Outlet-HTTP-API/td-p/1076449

import os
from datetime import datetime
import requests
from time import sleep
import json
from influxdb import InfluxDBClient

# the folder location where this code is located - for Raspberry pi or other linux based OS
os.chdir('/home/pi/mPower/')

# class definition for mPower
class mPower:

    #Change the parameters in the init fuction as required
    #AIROS_SESSIONID - 32 random digits
    #IP - ip address of mPower.
    def __init__(self):
        self.AIROS_SESSIONID = '32 digits random number'
        self.IP = 'IP address'
        self.username = 'ubnt'
        self.password = 'ubnt'
		
    def login(self):
        cookies = {
            'AIROS_SESSIONID': self.AIROS_SESSIONID,
        }

        data = [
            ('username', self.username),
            ('password', self.password),
        ]

        try:
            response = requests.post('http://'+self.IP+'/login.cgi', cookies=cookies, data=data)
            return response
        except Exception as e:
            print("Error Reading from mPower")
            print(e)
		
		
    def logout(self):
		cookies = {
            'AIROS_SESSIONID': self.AIROS_SESSIONID,
        }
		try:
            response = requests.post('http://'+self.IP+'/logout.cgi', cookies=cookies)
            return response
        except Exception as e:
            print("Error Reading from mPower")
            print(e)

    def query(self):
        cookies = {
            'AIROS_SESSIONID': self.AIROS_SESSIONID,
        }
        try:
            response = requests.get('http://'+self.IP+'/sensors', cookies=cookies)
            return response
        except Exception as e:
            print("Error Reading from mPower")
            print(e)


def parseAndWriteResult(result, devID, measurementName, influxClient):
    data = json.loads(result.text)
    power1 = data["sensors"][0]['power']
    power2 = data["sensors"][1]['power']
    power3 = data["sensors"][2]['power']
    try:
        data = {
            "measurement": str(devID + "_" + measurementName + '_port1'),
            "time": (datetime.utcnow().replace(microsecond=0).isoformat() + "Z"),
            "tags": {},
            "fields":{"Float_value": power1,}
        }
        influxClient.write_points([data])
        print ('Published to influxDB', data)
    except Exception as e:
        print("Error publishing to InfluxDB:")
        print(e)

    try:
        data = {
            "measurement": str(devID + "_" + measurementName + '_port2'),
            "time": (datetime.utcnow().replace(microsecond=0).isoformat() + "Z"),
            "tags": {},
            "fields":{"Float_v"
                      ""
                      "alue": power2,}
        }
        influxClient.write_points([data])
        print ('Published to influxDB', data)
    except Exception as e:
        print("Error publishing to InfluxDB:")
        print(e)

    try:
        data = {
            "measurement": str(devID + "_" + measurementName + '_port3'),
            "time": (datetime.utcnow().replace(microsecond=0).isoformat() + "Z"),
            "tags": {},
            "fields":{"Float_value": power3,}
        }
        influxClient.write_points([data])
        print ('Published to influxDB', data)
    except Exception as e:
        print("Error publishing to InfluxDB:")
        print(e)


if __name__ == "__main__":
    # Influxdb parameters
	host = "host IP"
    port = "8086"
    user = "username"
    passwd = "password"
    db = "iotdesks"
    ssl = False
    devID = "device_ID"
    measurementName = 'Power_Consumption_Watts'
    client = InfluxDBClient(host, port, user, passwd, db)
	delay = 4.9 # delay between measurements in seconds
	
    mPowerObj = mPower()
    mPowerObj.login()

    while True:

        try:
            queryResult = mPowerObj.query()
            if queryResult is not None:
                print queryResult.text
                parseAndWriteResult(queryResult, devID, measurementName,client)
                sleep(delay)
            else:
                mPowerObj.logout()
                sleep(delay)
                mPowerObj.login()
                print "New connection started with mPower"
        except Exception as e:
            print ("Error connecting to mPOwer")
            print e
            mPowerObj.logout()
            sleep(delay)
            mPowerObj.login()
