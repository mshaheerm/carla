#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
from __future__ import print_function

import argparse
import logging
import random
import time
import math
import grpc
import numpy as np

from carla.client import make_carla_client
# from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
# from carla.util import print_over_same_line


import test_grpc_pb2_grpc
import osi_groundtruth_pb2


def run_carla_client(args):
    # Here we will run episodes with frames/episode.
    number_of_episodes = 1
    frames_per_episode = 300
    settings = CarlaSettings()
    # We assume the CARLA server is already waiting for a client to connect at
    # host:port. To create a connection we can use the `make_carla_client`
    # context manager, it creates a CARLA client object and starts the
    # connection. It will throw an exception if something goes wrong. The
    # context manager makes sure the connection is always cleaned up on exit.
    with make_carla_client(args.host, args.port) as client:
        print('CarlaClient connected')
        for episode in range(0, number_of_episodes):
            # Start a new episode.

            if args.settings_filepath is None:

                # Create a CarlaSettings object. This object is a wrapper around
                # the CarlaSettings.ini file. Here we set the configuration we
                # want for the new episode.

                settings.set(
                    SynchronousMode=True,
                    SendNonPlayerAgentsInfo=True,
                    NumberOfVehicles=5,
                    NumberOfPedestrians=0,
                    WeatherId='Default',
                    QualityLevel=args.quality_level)
                settings.randomize_seeds()

            else:

                # Alternatively, we can load these settings from a file.
                with open(args.settings_filepath, 'r') as fp:
                    settings = fp.read()

            # Now we load these settings into the server. The server replies
            # with a scene description containing the available start spots for
            # the player. Here we can provide a CarlaSettings object or a
            # CarlaSettings.ini file as string.
            scene = client.load_settings(settings)

            # Choose one player start at random.
            number_of_player_starts = len(scene.player_start_spots)
            player_start = random.randint(0, max(0, number_of_player_starts - 1))

            # Notify the server that we want to start the episode at the
            # player_start index. This function blocks until the server is ready
            # to start the episode.
            print('Starting new episode at %r...' % scene.map_name)
            client.start_episode(player_start)

            # Iterate every frame in the episode.
            for frame in range(0, frames_per_episode):

                # Read the data produced by the server this frame.
                measurements, sensor_data = client.read_data()

                # Start the gRPC communication between server and client
                grpc_communication_module(measurements)

                # Now we have to send the instructions to control the vehicle.
                # If we are in synchronous mode the server will pause the
                # simulation until we send this control.

                client.send_control(
                    steer=random.uniform(-1.0, 1.0),
                    throttle=0.5,
                    brake=0.0,
                    hand_brake=False,
                    reverse=False)


def grpc_communication_module(measurements):
    """
    Assigns the measurements with OSI Standard
    :param measurements: measurements of the agents in the CARLA scene
    :return: sends A gRPC response if enabled
    """
    channel = grpc.insecure_channel('localhost:63558')
    stub = test_grpc_pb2_grpc.GroundtruthdataStub(channel)
    ground_truth = osi_groundtruth_pb2.GroundTruth()
    ego_vehicle = ground_truth.moving_object.add()

    # EGO Vehicle ID
    ground_truth.host_vehicle_id.value = ((2 ** 64) - 1)
    ego_vehicle.id.value = ((2 ** 64) - 1)

    # EGO Vehicle Location Velocity and dimension
    ego_vehicle.base.position.x = measurements.player_measurements.transform.location.x + \
        measurements.player_measurements.bounding_box.transform.location.x
    ego_vehicle.base.position.y = measurements.player_measurements.transform.location.y + \
        measurements.player_measurements.bounding_box.transform.location.y
    ego_vehicle.base.position.z = measurements.player_measurements.transform.location.z + \
        measurements.player_measurements.bounding_box.transform.location.z

    # Converting the Floating Point velocity from CARLA into a 3D OSI Standard Velocity
    yaw_host = measurements.player_measurements.transform.rotation.yaw
    pitch_host = measurements.player_measurements.transform.rotation.pitch
    roll_host = measurements.player_measurements.transform.rotation.roll

    x_host = measurements.player_measurements.forward_speed
    y_host = 0
    z_host = 0

    # Rotation about roll around x axis:
    sin_roll_host = math.sin(roll_host)
    cos_roll_host = math.cos(roll_host)
    y1_host = y_host * cos_roll_host - z_host * sin_roll_host
    z1_host = y_host * sin_roll_host + z_host * cos_roll_host

    # Rotation about pitch around y axis:
    sin_pitch_host = math.sin(pitch_host)
    cos_pitch_host = math.cos(pitch_host)
    x2_host = x_host * cos_pitch_host + z1_host * sin_pitch_host
    z2_host = -x_host * sin_pitch_host + z1_host * cos_pitch_host

    # Rotation about yaw around z axis:
    sin_yaw_host = math.sin(yaw_host)
    cos_yaw_host = math.cos(yaw_host)
    x3_host = x2_host * cos_yaw_host - y1_host * sin_yaw_host
    y3_host = x2_host * sin_yaw_host + y1_host * cos_yaw_host

    # EGO Vehicle Velocity with OSI Standardization
    ego_vehicle.base.velocity.x = x3_host
    ego_vehicle.base.velocity.y = y3_host
    ego_vehicle.base.velocity.z = z2_host

    # EGO Vehicle Rotation with OSI STANDARDIZATION  ( CARLA.degrees to OSI.radians)
    ego_vehicle.base.orientation.roll = measurements.player_measurements.transform.rotation.roll \
        * (math.pi / 180)
    ego_vehicle.base.orientation.pitch = measurements.player_measurements.transform.rotation.pitch \
        * (math.pi / 180)
    ego_vehicle.base.orientation.yaw = measurements.player_measurements.transform.rotation.yaw \
        * (math.pi / 180)

    # EgoVehicle BB Extent with OSI STANDARDIZATION([m] RADII DIMENSION OF THE BOUNDINGBOX(HALF))
    ego_vehicle.base.dimension.length = 2 * measurements.player_measurements.bounding_box.extent.x
    ego_vehicle.base.dimension.width = 2 * measurements.player_measurements.bounding_box.extent.y
    ego_vehicle.base.dimension.height = 2 * measurements.player_measurements.bounding_box.extent.z

    # TimeStamp with OSi Standardization
    ground_truth.timestamp.seconds = np.int64(measurements.game_timestamp / 1000)
    a_seconds = np.double(measurements.game_timestamp / 1000.0)
    b_seconds = math.floor(measurements.game_timestamp / 1000)
    ground_truth.timestamp.nanos = np.uint32((10 ** 9) * (a_seconds - b_seconds))

    for agent in measurements.non_player_agents:

        if agent.HasField('vehicle'):
            yaw = agent.vehicle.transform.rotation.yaw
            pitch = agent.vehicle.transform.rotation.pitch
            roll = agent.vehicle.transform.rotation.roll

            x_vehicle = agent.vehicle.forward_speed
            y_vehicle = 0
            z_vehicle = 0

            # Rotation about roll around x axis:
            sin_roll = math.sin(roll)
            cos_roll = math.cos(roll)
            y1_vehicle = y_vehicle * cos_roll - z_vehicle * sin_roll
            z1_vehicle = y_vehicle * sin_roll + z_vehicle * cos_roll

            # Rotation about pitch around y axis:
            sin_pitch = math.sin(pitch)
            cos_pitch = math.cos(pitch)
            x2_vehicle = x_vehicle * cos_pitch + z1_vehicle * sin_pitch
            z2_vehicle = -x_vehicle * sin_pitch + z1_vehicle * cos_pitch

            # Rotation about yaw around z axis:
            sin_yaw = math.sin(yaw)
            cos_yaw = math.cos(yaw)
            x3_vehicle = x2_vehicle * cos_yaw - y1_vehicle * sin_yaw
            y3_vehicle = x2_vehicle * sin_yaw + y1_vehicle * cos_yaw

            mov_obj = ground_truth.moving_object.add()

            # ID with OSI STANDARDIZATION
            mov_obj.id.value = agent.id

            # VELOCITY with OSI STANDARDIZATION
            mov_obj.base.velocity.x = x3_vehicle
            mov_obj.base.velocity.y = y3_vehicle
            mov_obj.base.velocity.z = z2_vehicle

            # POSITION with OSI STANDARDIZATION  (w.r.t CARTESIAN COORDINATES)
            mov_obj.base.position.x = agent.vehicle.transform.location.x \
                + agent.vehicle.bounding_box.transform.location.x
            mov_obj.base.position.y = agent.vehicle.transform.location.y \
                + agent.vehicle.bounding_box.transform.location.y
            mov_obj.base.position.z = agent.vehicle.transform.location.z \
                + agent.vehicle.bounding_box.transform.location.z

            # ROTATION with OSI STANDARDIZATION  ( CARLA.degrees to OSI.radians)
            mov_obj.base.orientation.roll = agent.vehicle.transform.rotation.roll \
                * (math.pi / 180)
            mov_obj.base.orientation.pitch = agent.vehicle.transform.rotation.pitch \
                * (math.pi / 180)
            mov_obj.base.orientation.yaw = agent.vehicle.transform.rotation.yaw \
                * (math.pi / 180)

            # BB EXTENT OF VEHICLE([m] RADII DIMENSION OF THE BOUNDING BOX(HALF))
            mov_obj.base.dimension.length = 2 * agent.vehicle.bounding_box.extent.x
            mov_obj.base.dimension.width = 2 * agent.vehicle.bounding_box.extent.y
            mov_obj.base.dimension.height = 2 * agent.vehicle.bounding_box.extent.z

    response = stub.ProcessGroundTruth(ground_truth)
    print(str(response))


def main():
    """
    Runs the CARLA Client with gRPC communication module.
    Contains the command line arguments.
    """
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Low',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')
    argparser.add_argument(
        '-c', '--carla-settings',
        metavar='PATH',
        dest='settings_filepath',
        default=None,
        help='Path to a "CarlaSettings.ini" file')

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    while True:
        try:

            run_carla_client(args)

            print('Done.')
            return

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
