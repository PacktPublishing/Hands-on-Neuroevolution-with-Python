#
# This is implementation of cart-pole apparatus simulation based on the Newton laws
# which use Euler's method for numerical approximation the equations on motion.
#
import math

#
# The constants defining physics of cart-pole apparatus
#
GRAVITY = 9.8  # m/s^2
MASSCART = 1.0 # kg
MASSPOLE = 0.5 # kg
TOTAL_MASS = (MASSPOLE + MASSCART)
# The distance from the center of mass of the pole to the pivot
# (actually half the pole's length)
LENGTH = 0.5 # m
POLEMASS_LENGTH = (MASSPOLE * LENGTH) # m
FORCE_MAG = 10.0 # N
FOURTHIRDS = 4.0/3.0
# the number seconds between state updates 
TAU = 0.02 # sec

def do_step(action, x, x_dot, theta, theta_dot):
    """
    The function to perform the one step of simulation over
    provided state variables.
    Arguments:
        action:     The binary action defining direction of
                    force to be applied.
        x:          The current cart X position
        x_dot:      The velocity of the cart
        theta:      The current angle of the pole from vertical
        theta_dot:  The angular velocity of the pole.
    Returns:
        The numerically approximated values of state variables
        after current time step (TAU)
    """
    force = -FORCE_MAG if action <= 0 else FORCE_MAG
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    temp = (force + POLEMASS_LENGTH * theta_dot * theta_dot * sin_theta) / TOTAL_MASS
    # The angular acceleration of the pole
    theta_acc = (GRAVITY * sin_theta - cos_theta * temp) / (LENGTH * (FOURTHIRDS - MASSPOLE * cos_theta * cos_theta / TOTAL_MASS))
    # The linear acceleration of the cart
    x_acc = temp - POLEMASS_LENGTH * theta_acc * cos_theta / TOTAL_MASS

    # Update the four state variables, using Euler's method.
    x_ret = x + TAU * x_dot
    x_dot_ret = x_dot + TAU * x_acc
    theta_ret = theta + TAU * theta_dot
    theta_dot_ret = theta_dot + TAU * theta_acc

    return x_ret, x_dot_ret, theta_ret, theta_dot_ret