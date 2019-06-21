#
# This is a definition of a maze environment simulation engine. It provides 
# routines to read maze configuration and build related simulation environment
# from it. Also it provides method to simulate the behavior of the navigating agent 
# and interaction with his sensors.
#
import math

import agent
import geometry

# The maximal allowed speed for the maze solver agent
MAX_AGENT_SPEED = 3.0

class Environment:
    """
    This class encapsulates the maze simulation environment.
    """
    def __init__(self, agent, walls, exit_point, exit_range=5.0):
        """
        Creates new maze environment with specified walls and exit point.
        Arguments:
            agent:      The maze navigating agent
            walls:      The maze walls
            exit_point: The maze exit point
            exit_range: The range arround exit point marking exit area
        """
        self.walls = walls
        self.exit_point = exit_point
        self.exit_range = exit_range
        # The maze navigating путе
        self.agent = agent
        # The flag to indicate if exit was found
        self.exit_found = False
        # The initial distance of agent from exit
        self.initial_distane = self.agent_distance_to_exit()

    def agent_distance_to_exit(self):
        """
        The function to estimate distance from maze solver agent to the maze exit.
        Returns:
            The distance from maze solver agent to the maze exit.
        """
        return self.agent.location.distance(self.exit_point)

    def test_wall_collision(self, loc):
        """
        The function to test if agent at specified location collides with any
        of the maze walls.
        Arguments:
            loc: The new agent location to test for collision.
        Returns:
            The True if agent at new location will collide with any of the maze walls.
        """
        for w in self.walls:
            if w.distance(loc) < self.agent.radius:
                return True

        return False
    
    def create_net_inputs(self):
        """
        The function to create the ANN input values from the simulation environment.
        Returns:
            The list of ANN inputs consist of values get from solver agent sensors.
        """
        inputs = []
        # The range finders
        for ri in self.agent.range_finders:
            inputs.append(ri)

        # The radar sensors
        for rs in self.agent.radar:
            inputs.append(rs)

        return inputs

    def apply_control_signals(self, control_signals):
        """
        The function to apply control signals received from control ANN to the
        maze solver agent.
        Arguments:
            control_signals: The control signals received from the control ANN
        """
        self.agent.angular_vel  += (control_signals[0] - 0.5)
        self.agent.speed        += (control_signals[1] - 0.5)

        # constrain the speed & angular velocity
        if self.agent.speed > MAX_AGENT_SPEED:
            self.agent.speed = MAX_AGENT_SPEED
        
        if self.agent.speed < -MAX_AGENT_SPEED:
            self.agent.speed = -MAX_AGENT_SPEED
        
        if self.agent.angular_vel > MAX_AGENT_SPEED:
            self.agent.angular_vel = MAX_AGENT_SPEED
        
        if self.agent.angular_vel < -MAX_AGENT_SPEED:
            self.agent.angular_vel = -MAX_AGENT_SPEED
    
    def update_rangefinder_sensors(self):
        """
        The function to update the agent range finder sensors.
        """
        for i, angle in enumerate(self.agent.range_finder_angles):
            rad = geometry.deg_to_rad(angle)
            # project a point from agent location outwards
            projection_point = geometry.Point(
                x = self.agent.location.x + math.cos(rad) * self.agent.range_finder_range,
                y = self.agent.location.y + math.sin(rad) * self.agent.range_finder_range
            )
            # rotate the projection point by the agent's heading angle to
            # aling it with heading direction
            projection_point.rotate(self.agent.heading, self.agent.location)
            # create the line segment from the agent location to the projected point
            projection_line = geometry.Line(
                a = self.agent.location,
                b = projection_point
            )
            # set range to maximum detection range
            min_range = self.agent.range_finder_range

            # now test against maze walls to see if projection line hits any wall
            # and find the closest hit
            for wall in self.walls:
                found, intersection = wall.intersection(projection_line)
                if found:
                    found_range = intersection.distance(self.agent.location)
                    # we are interested in the closest hit
                    if found_range < min_range:
                        min_range = found_range

            # Update sensor value
            self.agent.range_finders[i] = min_range

    def update_radars(self):
        """
        The function to update the agent radar sensors.
        """
        target = geometry.Point(self.exit_point.x, self.exit_point.y)
        # rotate target with respect to the agent's heading to aling it with heading direction
        target.rotate(self.agent.heading, self.agent.location)
        # translate with respect to the agent's location
        target.x -= self.agent.location.x
        target.y -= self.agent.location.y
        # the angle between maze exit point and the agent's heading direction
        angle = target.angle()
        # find the appropriate radar sensor to be fired
        for i, r_angles in enumerate(self.agent.radar_angles):
            self.agent.radar[i] = 0.0 # reset specific radar 

            if (angle >= r_angles[0] and angle < r_angles[1]) or (angle + 360 >= r_angles[0] and angle + 360 < r_angles[1]):
                self.agent.radar[i] = 1.0 # fire the radar

    def update(self, control_signals):
        """
        The function to run the one time step of the simulation.
        Arguments:
            control_signals: The control signals received from the control ANN
        """
        if self.exit_found:
            # Maze exit already found
            return

        # Apply control signals
        self.apply_control_signals(control_signals)

        # get X and Y velocity components
        vx = math.cos(geometry.deg_to_rad(self.agent.heading)) * self.agent.speed
        vy = math.sin(geometry.deg_to_rad(self.agent.heading)) * self.agent.speed

        # Update current Agent's heading (we consider the simulation time step size equal to 1s
        # and the angular velocity as degrees per second)
        self.agent.heading += self.agent.angular_vel

        # Enforce angular velocity bounds by wrapping
        if self.agent.heading > 360:
            self.agent.heading -= 360
        elif self.agent.heading < 0:
            self.agent.heading += 360

        # find the next location of the agent
        new_loc = geometry.Point(
            x = self.agent.x + vx, 
            y = self.agent.y + vy
        )

        if not self.test_wall_collision(new_loc):
            self.agent.location = new_loc

        # update agent's sensors
        self.update_rangefinder_sensors()
        self.update_radars()

        # check if agent reached exit point
        dist = self.agent_distance_to_exit()
        self.exit_found = (dist < self.exit_range)
            
