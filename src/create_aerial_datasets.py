import airsim
import time
import math
import os

import datetime
import matplotlib.pyplot as plt

import utils.airsim.multirotor_session as ms
import random


def largest_sample(path):
    if os.path.exists(path):
        return len(os.listdir(path))
    return 0


def take_random_images(client: ms.MultirotorSession, path, size = 1000, wait_time=0):
    for i in range(largest_sample(f"{path}/images"), size):
        random_x = int(50 * (random.random() - 0.5))
        random_y = int(50 * (random.random() - 0.5))
        random_z = int(-10)

        client.set_position(random_x, random_y, random_z)
        client.set_rotation(pitch=90, roll=270, yaw=0)

        if wait_time == 0:
            continue
        client.stay_in_air(wait_time)

        client.snap_image(f"{path}/images/{i}")
        client.save_metadata(f"{path}/metadata.jsonl", i)


def take_random_weather_images(client: ms.MultirotorSession, path, wait_time=0):
    for i in range(largest_sample(f"{path}/images"), 1000):
        random_x = int(50 * (random.random() - 0.5))
        random_y = int(50 * (random.random() - 0.5))
        random_z = int(-10)
        

        random_snow = random.random()
        random_maple_leaf = random.random()
        random_rain = random.random()
        random_dust = random.random()
        random_fog = random.random()

        if random.random() < 0.5:
            random_snow = 0
        if random.random() < 0.5:
            random_maple_leaf = 0
        if random.random() < 0.5:
            random_rain = 0
        if random.random() < 0.5:
            random_dust = 0
        if random.random() < 0.5:
            random_fog = 0

        client.set_weather_parameter(airsim.WeatherParameter.Snow, random_snow)
        client.set_weather_parameter(
            airsim.WeatherParameter.MapleLeaf, random_maple_leaf
        )
        client.set_weather_parameter(airsim.WeatherParameter.Rain, random_rain)
        client.set_weather_parameter(airsim.WeatherParameter.Dust, random_dust)
        client.set_weather_parameter(airsim.WeatherParameter.Fog, random_fog)

        client.set_position(random_x, random_y, random_z)
        client.set_rotation(pitch=90, roll=270, yaw=0)

        client.stay_in_air(wait_time)

        client.snap_image(f"{path}/images/{i}")
        client.save_metadata(f"{path}/metadata.jsonl", i)


def take_images(params, name, wait_time):
    time.sleep(3)
    client = ms.MultirotorSession()
    client.take_off()
    client.reset_weather_parameters()
    if len(params) != 0:
        for param, value in params:
            client.set_weather_parameter(param, value)
    time.sleep(3)
    take_random_images(
        client, f"data/cityenviron/aerial/{name}/train", 10000, wait_time
    )

    time.sleep(3)
    client = ms.MultirotorSession()
    client.take_off()
    client.reset_weather_parameters()
    if len(params) != 0:
        for param, value in params:
            client.set_weather_parameter(param, value)
    time.sleep(3)
    take_random_images(
        client, f"data/cityenviron/aerial/{name}/test", 100, wait_time
    )
    
    # time.sleep(3)
    # client = ms.MultirotorSession()
    # client.take_off()
    # client.reset_weather_parameters()
    # if len(params) != 0:
    #     for param, value in params:
    #         client.set_weather_parameter(param, value)
    # time.sleep(3)
    # take_random_images(
    #     client, f"data/cityenviron/aerial/{name}/val", 100, wait_time
    # )


# take_images([(airsim.WeatherParameter.Fog, 0.15)], "fog-0.15", 0.1)
# take_images([(airsim.WeatherParameter.Fog, 0.3)], "fog-0.3", 0.1)
take_images([(airsim.WeatherParameter.Fog, 0.5)], "fog-0.5", 0.1)
# take_images([(airsim.WeatherParameter.Dust, 0.15)], "dust-0.15", 0.1)
# take_images([(airsim.WeatherParameter.Dust, 0.3)], "dust-0.3", 0.1)
take_images([(airsim.WeatherParameter.Dust, 0.5)], "dust-0.5", 0.1)
# take_images([(airsim.WeatherParameter.MapleLeaf, 0.15), (airsim.WeatherParameter.RoadLeaf, 0.15)], "maple_leaf-0.15", 5)
# take_images([(airsim.WeatherParameter.MapleLeaf, 0.3), (airsim.WeatherParameter.RoadLeaf, 0.3)], "maple_leaf-0.3", 5)
take_images([(airsim.WeatherParameter.MapleLeaf, 0.5), (airsim.WeatherParameter.RoadLeaf, 0.5)], "maple_leaf-0.5", 5)
# take_images([(airsim.WeatherParameter.Rain, 0.15), (airsim.WeatherParameter.Roadwetness, 0.15)], "rain-0.15", 0.1)
# take_images([(airsim.WeatherParameter.Rain, 0.3), (airsim.WeatherParameter.Roadwetness, 0.3)], "rain-0.3", 0.1)
take_images([(airsim.WeatherParameter.Rain, 0.5), (airsim.WeatherParameter.Roadwetness, 0.5)], "rain-0.5", 3)
# take_images([(airsim.WeatherParameter.Snow, 0.15), (airsim.WeatherParameter.RoadSnow, 0.15)], "snow-0.15", 0.1)
# take_images([(airsim.WeatherParameter.Snow, 0.3), (airsim.WeatherParameter.RoadSnow, 0.3)], "snow-0.4", 0.1)
take_images([(airsim.WeatherParameter.Snow, 0.5), (airsim.WeatherParameter.RoadSnow, 0.5)], "snow-0.5", 3)
take_images([], "normal", 0.1)

time.sleep(3)
client = ms.MultirotorSession()
client.take_off()
client.reset_weather_parameters()
time.sleep(3)
take_random_weather_images(client, "data/cityenviron/test", 3)
