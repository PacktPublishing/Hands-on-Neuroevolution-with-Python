#
# This is implementation of cart-pole apparatus simulation based on the Newton laws
# which use Euler's method for numerical approximation the equations on motion.
#
import math
import random

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

# set random seed
random.seed(42)

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
    # Find the force direction
    force = -FORCE_MAG if action <= 0 else FORCE_MAG
    # Pre-calcuate cosine and sine to optimize performance 
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

def run_cart_pole_simulation(net, max_bal_steps, random_start=True):
    """
    The function to run cart-pole apparatus simulation for a
    certain number of time steps as maximum.
    Arguments:
        net: The ANN of the phenotype to be evaluated.
        max_bal_steps: The maximum nubmer of time steps to
            execute simulation.
        random_start: If evaluates to True than cart-pole simulation 
            starts from random initial positions.
    Returns:
        the number of steps that the control ANN was able to
        maintain the single-pole balancer in stable state.
    """
    # Set random initial state if appropriate
    x, x_dot, theta, theta_dot = 0.0, 0.0, 0.0, 0.0
    if random_start:
        x = (random.random() * 4.8 - 2.4) / 2.0 # -1.4 < x < 1.4
        x_dot = (random.random() * 3 - 1.5) / 4.0 # -0.375 < x_dot < 0.375
        theta = (random.random() * 0.42 - 0.21) / 2.0 # -0.105 < theta < 0.105
        theta_dot = (random.random() * 4 - 2) / 4.0 # -0.5 < theta_dot < 0.5

    # Run simulation for specified number of steps while
    # cart-pole system stays within contstraints
    input = [None] * 4 # the inputs
    for steps in range(max_bal_steps):
        # Load scaled inputs
        input[0] = (x + 2.4) / 4.8
        input[1] = (x_dot + 1.5) / 3
        input[2] = (theta + 0.21) / .42
        input[3] = (theta_dot + 2.0) / 4.0

        # Activate the NET
        output = net.activate(input)
        # Make action values discrete
        action = 0 if output[0] < 0.5 else 1

        # Apply action to the simulated cart-pole
        x, x_dot, theta, theta_dot = do_step(   action = action, 
                                                x = x, 
                                                x_dot = x_dot, 
                                                theta = theta, 
                                                theta_dot = theta_dot )

        # Check for failure due constraints violation. If so, return number of steps.
        if x < -2.4 or x > 2.4 or theta < -0.21 or theta > 0.21:
            return steps

    return max_bal_steps

def eval_fitness(net, max_bal_steps=500000):
    """
    The function to evaluate fitness score of phenotype produced
    provided ANN
    Arguments:
        net: The ANN of the phenotype to be evaluated.
        max_bal_steps: The maximum nubmer of time steps to
            execute simulation.
    Returns:
        The phenotype fitness score in range [0, 1]
    """
    # First we run simulation loop returning number of successfull
    # simulation steps
    steps = run_cart_pole_simulation(net, max_bal_steps)

    if steps == max_bal_steps:
        # the maximal fitness
        return 1.0
    elif steps == 0: # needed to avoid math error when taking log(0)
        # the minimal fitness
        return 0.0
    else:
        # we use logarithmic scale because most cart-pole runs fails 
        # too early - within ~100 steps, but we are testing against 
        # 500'000 balancing steps
        log_steps = math.log(steps)
        log_max_steps = math.log(max_bal_steps)
        # The loss value is in range [0, 1]
        error = (log_max_steps - log_steps) / log_max_steps
        # The fitness value is a complement of the loss value
        return 1.0 - error





