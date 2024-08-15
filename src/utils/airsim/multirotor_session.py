import os
import json
import math
import time

import airsim
import numpy as np
import cv2


class MultirotorSession:
  def __init__(self):
    self.camera_name = "low_res"

    self.client = airsim.MultirotorClient()
    self.client.confirmConnection()
    self.client.enableApiControl(True)
    self.client.armDisarm(True)
    self.client.simEnableWeather(True)
    self.weather_params = self.default_weather_params()

  def __del__(self):
    self.reset()
    self.reset_weather_parameters()
    # self.client.armDisarm(False)
    # self.client.enableApiControl(False)
    # self.client.simEnableWeather(False)

  def reset_weather_parameters(self):
    self.set_weather_parameter(airsim.WeatherParameter.Rain, 0)
    self.set_weather_parameter(airsim.WeatherParameter.Roadwetness, 0)
    self.set_weather_parameter(airsim.WeatherParameter.Snow, 0)
    self.set_weather_parameter(airsim.WeatherParameter.RoadSnow, 0)
    self.set_weather_parameter(airsim.WeatherParameter.MapleLeaf, 0)
    self.set_weather_parameter(airsim.WeatherParameter.RoadLeaf , 0)
    self.set_weather_parameter(airsim.WeatherParameter.Dust, 0)
    self.set_weather_parameter(airsim.WeatherParameter.Fog, 0)

  def snap_image(self, directory: str):
    """
    A convenience method to take a single image and save it to a directory
    """
    responses = self.get_images()
    self.save_images(responses, directory)

  def reset(self):
    self.client.reset()

  def wait_key(self, message):
    airsim.wait_key(message)

  # ==============================
  # Movement methods
  # ==============================
  def take_off(self):
    self.client.takeoffAsync().join()

  def stay_in_air(self, seconds):

    current_kinematics = self.client.simGetGroundTruthKinematics()

    start_time = time.time()
    while time.time() - start_time < seconds: 
      self.client.simSetKinematics(current_kinematics, ignore_collision=True)
    self.client.simSetKinematics(current_kinematics, ignore_collision=True)

  def move_to_position(self, x, y, z, velocity = 1):
    self.client.moveToPositionAsync(x, y, z, velocity).join()

  def set_position(self, x, y, z):
    kinematics_state = self.client.simGetGroundTruthKinematics()
    kinematics_state.position.x_val = x
    kinematics_state.position.y_val = y
    kinematics_state.position.z_val = z
    self.client.simSetKinematics(kinematics_state, ignore_collision=True)

  def set_rotation(self, roll = 0, pitch = 0, yaw = 0):
    '''
    roll - roll angle in degrees
    pitch - pitch angle in degrees
    yaw - yaw angle in degrees
    '''

    roll_radians = math.radians(roll)
    pitch_radians = math.radians(pitch)
    yaw_radians = math.radians(yaw)

    kinematics_state = self.client.simGetGroundTruthKinematics()
    kinematics_state.orientation = airsim.to_quaternion(roll_radians, pitch_radians, yaw_radians)
    self.client.simSetKinematics(kinematics_state, ignore_collision=False)

  def hover(self):
    self.client.hoverAsync().join()

  # ==============================
  # Image methods
  # ==============================
  def get_images(self):
    """
    Get images from the drone's cameras
    """
    responses = self.client.simGetImages([
      airsim.ImageRequest(self.camera_name, airsim.ImageType.Scene), # scene vision image in png format
      # airsim.ImageRequest(self.camera_name, airsim.ImageType.DepthPlanar),
      # airsim.ImageRequest(self.camera_name, airsim.ImageType.DepthPerspective),
      # airsim.ImageRequest(self.camera_name, airsim.ImageType.DepthVis),
      # airsim.ImageRequest(self.camera_name, airsim.ImageType.DisparityNormalized),
      airsim.ImageRequest(self.camera_name, airsim.ImageType.Segmentation),
      # airsim.ImageRequest(self.camera_name, airsim.ImageType.SurfaceNormals),
      # airsim.ImageRequest(self.camera_name, airsim.ImageType.Infrared),
      # airsim.ImageRequest(self.camera_name, airsim.ImageType.OpticalFlow),
      # airsim.ImageRequest(self.camera_name, airsim.ImageType.OpticalFlowVis),
    ])
    return responses
  
  def save_images(self, responses, directory):
    """
    Save images to a directory.
    """

    # Create path if it doesn't exist
    if not os.path.exists(directory):
      os.makedirs(directory)

    for index, response in enumerate(responses):
      filename = os.path.join(directory, f"{self.imagetype_to_string(response.image_type)}")
      self.save_response_to_file(response, filename)

  # ==============================
  # Weather methods
  # ==============================
  def set_weather_parameter(self, weather, amount):
    self.weather_params[self.airsim_weather_to_str(weather)] = amount
    self.client.simSetWeatherParameter(weather, amount)
  
  def set_time_of_day(self, time_of_day):
    self.client.simSetTimeOfDay(time_of_day)


  # ==============================
  # Metadata methods
  # ==============================
  def airsim_weather_to_str(self, weatherType: airsim.WeatherParameter):
    if weatherType == airsim.WeatherParameter.Rain: return "rain"
    if weatherType == airsim.WeatherParameter.Snow: return "snow"
    if weatherType == airsim.WeatherParameter.MapleLeaf: return "maple_leaf"
    if weatherType == airsim.WeatherParameter.RoadLeaf: return "road_leaf"
    if weatherType == airsim.WeatherParameter.Dust: return "dust"
    if weatherType == airsim.WeatherParameter.Fog: return "fog"
    else: return "unknown"

  def default_weather_params(self):
    return {
      self.airsim_weather_to_str(airsim.WeatherParameter.Rain): 0,
      self.airsim_weather_to_str(airsim.WeatherParameter.Snow): 0,
      self.airsim_weather_to_str(airsim.WeatherParameter.MapleLeaf): 0,
      self.airsim_weather_to_str(airsim.WeatherParameter.RoadLeaf): 0,
      self.airsim_weather_to_str(airsim.WeatherParameter.Dust): 0,
      self.airsim_weather_to_str(airsim.WeatherParameter.Fog): 0
    }

  def save_metadata(self, filename, sample_index):
    file = open(filename, 'a')

    camera_info = self.client.simGetCameraInfo("high_res")
    environment_state = self.client.simGetGroundTruthEnvironment()

    obj = {
      "sample_index": sample_index,
      "weather_params": self.weather_params,
      "camera_info": self.camera_info_to_dict(camera_info),
      "environment_state": self.environment_state_to_dict(environment_state),
    }
    file.write(f"{json.dumps(obj)}\n")
    file.close()

    
  # ==============================
  # Helper methods
  # ==============================
  def imagetype_to_string(self, image_type):
    if image_type == airsim.ImageType.Scene: return "Scene"
    if image_type == airsim.ImageType.DepthPlanar: return "DepthPlanar"
    if image_type == airsim.ImageType.DepthPerspective: return "DepthPerspective"
    if image_type == airsim.ImageType.DepthVis: return "DepthVis"
    if image_type == airsim.ImageType.DisparityNormalized: return "DisparityNormalized"
    if image_type == airsim.ImageType.Segmentation: return "Segmentation"
    if image_type == airsim.ImageType.SurfaceNormals: return "SurfaceNormals"
    if image_type == airsim.ImageType.Infrared: return "Infrared"
    if image_type == airsim.ImageType.OpticalFlow: return "OpticalFlow"
    if image_type == airsim.ImageType.OpticalFlowVis: return "OpticalFlowVis"
    else: return "Unknown"

  def save_response_to_file(self, response, filename):
    if response.pixels_as_float:
      airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
    elif response.compress: #png format
      airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
    else: #uncompressed array
      img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) # get numpy array
      img_rgb = img1d.reshape(response.height, response.width, 3) # reshape array to 4 channel image array H X W X 3
      cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb) # write to png

  def camera_info_to_dict(self, camera_info: airsim.CameraInfo):
    return {
      "pose": {
        "position": {
          "x": camera_info.pose.position.x_val,
          "y": camera_info.pose.position.y_val,
          "z": camera_info.pose.position.z_val
        },
        "orientation": {
          "w_val": camera_info.pose.orientation.w_val,
          "x_val": camera_info.pose.orientation.x_val,
          "y_val": camera_info.pose.orientation.y_val,
          "z_val": camera_info.pose.orientation.z_val
        }
      },
      "fov": camera_info.fov,
    }

  def environment_state_to_dict(self, environment_state: airsim.EnvironmentState): 
    return {
      "position": {
        "x": environment_state.position.x_val,
        "y": environment_state.position.y_val,
        "z": environment_state.position.z_val
      },
      "air_density": environment_state.air_density,
      "gravity": {
        "x": environment_state.gravity.x_val,
        "y": environment_state.gravity.y_val,
        "z": environment_state.gravity.z_val
      },
      "temperature": environment_state.temperature,
      "air_pressure": environment_state.air_pressure,
      "geo_point": {
        "altitude": environment_state.geo_point.altitude,
        "latitude": environment_state.geo_point.latitude,
        "longitude": environment_state.geo_point.longitude
      },
    }