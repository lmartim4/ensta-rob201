"""A simple SLAM demonstration using the "placebot" robot simulator"""

from place_bot.entities.lidar import LidarParams
from place_bot.entities.odometer import OdometerParams
from place_bot.simu_world.simulator import Simulator

from my_robot_slam import MyRobotSlam

from worlds.my_world import MyWorld

if __name__ == "__main__":
    lidar_params = LidarParams()
    lidar_params.noise_enable = False

    odometer_params = OdometerParams()
    use_shaders = True

    my_robot = MyRobotSlam(lidar_params=lidar_params,
                           odometer_params=odometer_params)
    my_world = MyWorld(robot=my_robot, use_shaders=use_shaders)
    simulator = Simulator(the_world=my_world, use_keyboard=False)

    simulator.run()
