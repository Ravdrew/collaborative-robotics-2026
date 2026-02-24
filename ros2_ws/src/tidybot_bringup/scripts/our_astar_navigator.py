#!/usr/bin/env python3
# Navigation code (TidyBot2 A* trajectory tracking)

import numpy as np
import typing as T
from numpy import linalg
from scipy.interpolate import splev, splrep
import matplotlib.pyplot as plt

# Make ROS and ASL-specific imports optional so A* can be tested standalone
try:
    import time
    import math
    import rclpy                    # ROS2 client library
    from rclpy.node import Node     # ROS2 node baseclass
    from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
    from asl_tb3_lib.math_utils import wrap_angle # robot-agnostic
    from asl_tb3_lib.tf_utils import quaternion_to_yaw # robot-agnostic
    from geometry_msgs.msg import Pose2D, Twist
    from nav_msgs.msg import Odometry, OccupancyGrid
    from std_msgs.msg import Bool
    from asl_tb3_lib.grids import snap_to_grid, StochOccupancyGrid2D # robot-agnostic
    ROS_AVAILABLE = True
except Exception:
    # If ROS or ASL libraries are not present, allow importing AStar for testing
    import time
    import math
    ROS_AVAILABLE = False

    # Distinct dummy base classes so `class Navigator(Node, BaseNavigator)` is valid
    class Node:  # type: ignore[no-redef]
        pass

    class BaseNavigator:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self.t_prev = 0.0
            self.V_prev = 0.0
            self.om_prev = 0.0

    TrajectoryPlan = None
    Pose2D = None
    Twist = None
    Odometry = None
    OccupancyGrid = None
    Bool = None
    StochOccupancyGrid2D = None
    # Provide a minimal snap_to_grid placeholder if needed by other code
    def snap_to_grid(x):
        return x

class AStar(object):
    """Represents a motion planning problem to be solved using A*"""

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1):
        self.statespace_lo = statespace_lo         # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = statespace_hi         # state space upper bound (e.g., [5, 5])
        self.occupancy = occupancy                 # occupancy grid (a DetOccupancyGrid2D object)
        self.resolution = resolution               # resolution of the discretization of state space (cell/m)
        self.x_offset = x_init                     
        self.x_init = self.snap_to_grid(x_init)    # initial state
        self.x_goal = self.snap_to_grid(x_goal)    # goal state

        self.closed_set = set()    # the set containing the states that have been visited
        self.open_set = set()      # the set containing the states that are condidate for future expension

        self.est_cost_through = {}  # dictionary of the estimated cost from start to goal passing through state (often called f score)
        self.cost_to_arrive = {}    # dictionary of the cost-to-arrive at state from start (often called g score)
        self.came_from = {}         # dictionary keeping track of each state's parent to reconstruct the path

        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0
        self.est_cost_through[self.x_init] = self.distance(self.x_init,self.x_goal)

        self.path = None        # the final path as a list of states

    def is_free(self, x):
        """
        Checks if a give state x is free, meaning it is inside the bounds of the map and
        is not inside any obstacle.
        Inputs:
            x: state tuple
        Output:
            Boolean True/False
        Hint: self.occupancy is a DetOccupancyGrid2D object, take a look at its methods for what might be
              useful here
        """
        ########## Code starts here ##########
        x = np.asarray(x)
        if np.all(x >= self.statespace_lo) and np.all(x <= self.statespace_hi):
            return self.occupancy.is_free(x)
        return False
        # # check if x is within the state space bounds
        # if not (self.statespace_lo[0] <= x[0] <= self.statespace_hi[0] and self.statespace_lo[1] <= x[1] <= self.statespace_hi[1]):
        #     return False
        # check if x is in a free space according to the occupancy grid
        # if not self.occupancy.is_free(x):
        #     return False
        # return True

        # raise NotImplementedError("is_free not implemented")
        ########## Code ends here ##########

    def distance(self, x1, x2):
        """
        Computes the Euclidean distance between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float Euclidean distance

        HINT: This should take one line. Tuples can be converted to numpy arrays using np.array().
        """
        ########## Code starts here ##########
        # compute the Euclidean distance between x1 and x2
        return np.linalg.norm(np.array(x1) - np.array(x2))

        # compute the L1 norm (Manhattan distance) between x1 and x2
        # return np.linalg.norm(np.array(x1) - np.array(x2), ord=1)

        # compute the L-infinity norm (Chebyshev distance) between x1 and x2
        # return np.linalg.norm(np.array(x1) - np.array(x2), ord=np.inf)

        # raise NotImplementedError("distance not implemented")
        ########## Code ends here ##########

    def snap_to_grid(self, x):
        """ Returns the closest point on a discrete state grid
        Input:
            x: tuple state
        Output:
            A tuple that represents the closest point to x on the discrete state grid
        """
        return (
            self.resolution * round((x[0] - self.x_offset[0]) / self.resolution) + self.x_offset[0],
            self.resolution * round((x[1] - self.x_offset[1]) / self.resolution) + self.x_offset[1],
        )

    def get_neighbors(self, x):
        """
        Gets the FREE neighbor states of a given state x. Assumes a motion model
        where we can move up, down, left, right, or along the diagonals by an
        amount equal to self.resolution.
        Input:
            x: tuple state
        Ouput:
            List of neighbors that are free, as a list of TUPLES

        HINTS: Use self.is_free to check whether a given state is indeed free.
               Use self.snap_to_grid (see above) to ensure that the neighbors
               you compute are actually on the discrete grid, i.e., if you were
               to compute neighbors by adding/subtracting self.resolution from x,
               numerical errors could creep in over the course of many additions
               and cause grid point equality checks to fail. To remedy this, you
               should make sure that every neighbor is snapped to the grid as it
               is computed.
        """
        neighbors = []
        ########## Code starts here ##########
        # Define the possible movements (8 directions)
        movements = [
            (self.resolution, 0),   # right
            (-self.resolution, 0),  # left
            (0, self.resolution),   # up
            (0, -self.resolution),  # down
            (self.resolution, self.resolution),    # up-right
            (self.resolution, -self.resolution),   # down-right
            (-self.resolution, self.resolution),   # up-left
            (-self.resolution, -self.resolution)   # down-left
        ]
        for move in movements:
            neighbor = self.snap_to_grid((x[0] + move[0], x[1] + move[1]))
            if self.is_free(neighbor):
                neighbors.append(neighbor)
        return neighbors
        # raise NotImplementedError("get_neighbors not implemented")
        ########## Code ends here ##########

    def find_best_est_cost_through(self):
        """
        Gets the state in open_set that has the lowest est_cost_through
        Output: A tuple, the state found in open_set that has the lowest est_cost_through
        """
        return min(self.open_set, key=lambda x: self.est_cost_through[x])

    def reconstruct_path(self):
        """
        Use the came_from map to reconstruct a path from the initial location to
        the goal location
        Output:
            A list of tuples, which is a list of the states that go from start to goal
        """
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        return list(reversed(path))

    def plot_path(self, fig_num=0, show_init_label=True):
        """Plots the path found in self.path and the obstacles"""
        if not self.path:
            return

        self.occupancy.plot(fig_num)

        solution_path = np.asarray(self.path)
        plt.plot(solution_path[:,0],solution_path[:,1], color="green", linewidth=2, label="A* solution path", zorder=10)
        plt.scatter([self.x_init[0], self.x_goal[0]], [self.x_init[1], self.x_goal[1]], color="green", s=30, zorder=10)
        if show_init_label:
            plt.annotate(r"$x_{init}$", np.array(self.x_init) + np.array([.2, .2]), fontsize=16)
        plt.annotate(r"$x_{goal}$", np.array(self.x_goal) + np.array([.2, .2]), fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)

        plt.axis([0, self.occupancy.width, 0, self.occupancy.height])

    def plot_tree(self, point_size=15):
        # plot_line_segments([(x, self.came_from[x]) for x in self.open_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
        # plot_line_segments([(x, self.came_from[x]) for x in self.closed_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
        # px = [x[0] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
        # py = [x[1] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
        # plt.scatter(px, py, color="blue", s=point_size, zorder=10, alpha=0.2)
        print("Tree plotting not implemented")

    def solve(self):
        """
        Solves the planning problem using the A* search algorithm. It places
        the solution as a list of tuples (each representing a state) that go
        from self.x_init to self.x_goal inside the variable self.path
        Input:
            None
        Output:
            Boolean, True if a solution from x_init to x_goal was found

        HINTS:  We're representing the open and closed sets using python's built-in
                set() class. This allows easily adding and removing items using
                .add(item) and .remove(item) respectively, as well as checking for
                set membership efficiently using the syntax "if item in set".
        """
        ########## Code starts here ##########
        while self.open_set:
            current = self.find_best_est_cost_through()
            if current == self.x_goal:
                self.path = self.reconstruct_path()
                return True

            self.open_set.remove(current)
            self.closed_set.add(current)

            for neighbor in self.get_neighbors(current):
                if neighbor in self.closed_set:
                    continue

                tentative_g_score = self.cost_to_arrive[current] + self.distance(current, neighbor)

                if neighbor not in self.open_set:
                    self.open_set.add(neighbor)
                elif tentative_g_score >= self.cost_to_arrive.get(neighbor, float('inf')):
                    continue

                self.came_from[neighbor] = current
                self.cost_to_arrive[neighbor] = tentative_g_score
                self.est_cost_through[neighbor] = tentative_g_score + self.distance(neighbor, self.x_goal)
        # raise NotImplementedError("solve not implemented")
        ########## Code ends here ##########

#----------------------------------------------------------------------------------------------------------
class Navigator(Node, BaseNavigator):
    def __init__(self, kpx: float = 6.0, kpy: float = 6.0, kdx: float = 3.0, kdy: float = 3.0,
                 V_max: float = 0.5, om_max: float = 1.0) -> None:
        Node.__init__(self, 'astar_navigator')
        BaseNavigator.__init__(self)
        # set the proportional control gain
        self.kp = 2.0
        # set the constants for trajectory tracking
        self.kpx = kpx
        self.kpy = kpy
        self.kdx = kdx
        self.kdy = kdy
        self.V_max = V_max
        self.om_max = om_max
        self.V_PREV_THRES = 0.0001

        # Subscriptions
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.goal_sub = self.create_subscription(Pose2D, '/cmd_nav', self.goal_callback, 10)
        self.goal_sub_legacy = self.create_subscription(Pose2D, '/nav_goal', self.goal_callback, 10)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.end_pose_pub = self.create_publisher(Pose2D, '/end_nav_pose', 10)
        self.nav_success_pub = self.create_publisher(Bool, '/nav_success', 10)

        # Internal state
        self.current_state = None
        self.occupancy = None
        self.current_plan = None
        self.plan_start_time = None
        self.plan_active = False
        self.goal_tolerance = 0.05    # metres
        self.yaw_tolerance  = 0.05    # radians

        self.create_timer(0.02, self.control_loop)
        self.get_logger().info('A* Navigator ready')

    def reset(self) -> None:
        """ Reset internal variables for trajectory tracking controller """
        self.t_prev = 0.0
        self.V_prev = 0.0
        self.om_prev = 0.0
        
    # override the compute_heading_control() method from BaseHeadingController
    #change to tidybot control and state
    def compute_heading_control(self, state: Pose2D, goal: Pose2D) -> Twist:
        # compute the heading error
        heading_error = wrap_angle(goal.theta - state.theta)

        # required angular velocity
        angular_velocity = self.kp * heading_error
        # create a Twist message for angular velocity
        cmd = Twist()
        cmd.angular.z = float(angular_velocity)
        return cmd
    
    def compute_trajectory_tracking_control(self, state: Pose2D, plan: TrajectoryPlan, t: float) -> Twist:
        """ Compute control target using a trajectory tracking controller

        Args:
            state (Pose2D): current robot state (x, y, theta)
            plan (TrajectoryPlan): planned trajectory
            t (float): current timestep

        Returns:
            Twist: control command (linear velocity, angular velocity)
        """

        dt = t - self.t_prev

        # Sample desired state and derivatives from spline parameters
        x_d   = splev(t, plan.path_x_spline, der=0)
        xd_d  = splev(t, plan.path_x_spline, der=1)
        xdd_d = splev(t, plan.path_x_spline, der=2)
        y_d   = splev(t, plan.path_y_spline, der=0)
        yd_d  = splev(t, plan.path_y_spline, der=1)
        ydd_d = splev(t, plan.path_y_spline, der=2)

        ########## Code starts here ##########
        # avoid singularity
        if abs(self.V_prev) < self.V_PREV_THRES:
            self.V_prev = self.V_PREV_THRES

        xd = self.V_prev*np.cos(state.theta)
        yd = self.V_prev*np.sin(state.theta)

        # compute virtual controls
        u = np.array([xdd_d + self.kpx*(x_d-state.x) + self.kdx*(xd_d-xd),
                      ydd_d + self.kpy*(y_d-state.y) + self.kdy*(yd_d-yd)])

        # compute real controls
        J = np.array([[np.cos(state.theta), -self.V_prev*np.sin(state.theta)],
                          [np.sin(state.theta), self.V_prev*np.cos(state.theta)]])
        a, om = linalg.solve(J, u)
        V = self.V_prev + a*dt
        ########## Code ends here ##########

        # apply control limits
        V = np.clip(V, -self.V_max, self.V_max)
        om = np.clip(om, -self.om_max, self.om_max)

        # save the commands that were applied and the time
        self.t_prev = t
        self.V_prev = V
        self.om_prev = om

        cmd = Twist()
        cmd.linear.x = float(V)
        cmd.angular.z = float(om)
        return cmd

    def compute_trajectory_plan(self, state: Pose2D, goal: Pose2D, occupancy: StochOccupancyGrid2D, resolution: float, horizon: float) -> T.Optional[TrajectoryPlan]:
        """ Compute a trajectory plan using A* and cubic spline fitting
        
        Args:
            state (Pose2D): current robot state (x, y, theta)
            goal (Pose2D): goal state (x, y, theta)
            occupancy (StochOccupancyGrid2D): occupancy
            resolution (float): resolution
            horizon (float): horizon

        Returns:
            T.Optional[TrajectoryPlan]:
        """
        
        x_init = (state.x, state.y)
        x_goal = (goal.x, goal.y)

        astar = AStar(statespace_lo=(state.x-horizon,state.y-horizon), statespace_hi=(state.x+horizon,state.y+horizon), x_init=x_init, x_goal=x_goal, occupancy=occupancy, resolution=resolution)
        if not astar.solve() or len(astar.path) < 4:
            print("No valid path found")
            return None
        self.reset()
        path = np.array(astar.path)

        # path time 
        v_desired = 0.15
        spline_alpha = 0.05
        ts = [0.0]
        path = np.array(path)
        for i in range(len(path)-1):
            dist = linalg.norm(path[i+1]-path[i])
            dt = dist / v_desired
            ts.append(ts[-1] + dt)
        ts = np.array(ts)

        x_path = splrep(ts, path[:,0], k=3, s=spline_alpha)
        y_path = splrep(ts, path[:,1], k=3, s=spline_alpha)

        # code from Cole
        plan = TrajectoryPlan(
            path = path,
            path_x_spline=x_path,
            path_y_spline=y_path,
            duration = ts[-1]
        )
        
        return plan

    def odom_callback(self, msg: Odometry) -> None:
        """Extract robot pose from odometry (matches TrajectoryTracker convention)."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        theta = 2.0 * math.atan2(q.z, q.w) - math.pi / 2
        self.current_state = Pose2D(x=float(x), y=float(y), theta=float(theta))

    def map_callback(self, msg: OccupancyGrid) -> None:
        """Convert ROS2 OccupancyGrid to StochOccupancyGrid2D for A* planning."""
        self.occupancy = StochOccupancyGrid2D(
            resolution=msg.info.resolution,
            size_xy=np.array([msg.info.width, msg.info.height]),
            origin_xy=np.array([msg.info.origin.position.x, msg.info.origin.position.y]),
            window_size=3,
            probs=msg.data,
        )

    def goal_callback(self, msg: Pose2D) -> None:
        """Receive a navigation goal, run A*, and start trajectory tracking."""
        if self.current_state is None or self.occupancy is None:
            self.get_logger().warn('Goal received but state/map not yet available')
            return
        self.current_plan = self.compute_trajectory_plan(
            self.current_state, msg, self.occupancy,
            resolution=0.1, horizon=10.0,
        )
        if self.current_plan:
            self.plan_start_time = time.time()
            self.plan_active = True
            self.get_logger().info(f'A* plan computed ({len(self.current_plan.path)} waypoints)')
        else:
            self.get_logger().warn('A* failed to find a path to goal')

    def control_loop(self) -> None:
        """50 Hz control loop: track the current trajectory plan."""
        if not self.plan_active or self.current_state is None:
            return
        t = time.time() - self.plan_start_time
        if t > self.current_plan.duration:
            self._finish_goal()
            return
        cmd = self.compute_trajectory_tracking_control(self.current_state, self.current_plan, t)
        self.cmd_vel_pub.publish(cmd)

    def _finish_goal(self) -> None:
        """Stop the robot and signal success to FrontierExplorer."""
        self.cmd_vel_pub.publish(Twist())
        pose = Pose2D(
            x=self.current_state.x,
            y=self.current_state.y,
            theta=self.current_state.theta,
        )
        self.end_pose_pub.publish(pose)
        self.nav_success_pub.publish(Bool(data=True))
        self.plan_active = False
        self.get_logger().info('Goal reached')


def main(argv=None):
    """Entry point for running the Navigator as a ROS2 node or a standalone demo."""
    if ROS_AVAILABLE:
        rclpy.init(args=argv)
        node = Navigator()
        try:
            rclpy.spin(node)
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        print("ROS not available — running standalone A* test")
        try:
            from mod_asl3_lib.grids import StochOccupancyGrid2D
        except Exception as e:
            print("Could not import local StochOccupancyGrid2D:", e)
            raise

        resolution = 1.0
        size_xy = np.array([20, 20], dtype=int)
        origin_xy = np.array([0.0, 0.0])

        probs = np.zeros(size_xy[0] * size_xy[1], dtype=float)
        probs = probs.reshape((size_xy[1], size_xy[0]))
        probs[8:12, 8:12] = 100.0
        probs = probs.flatten().tolist()

        occupancy = StochOccupancyGrid2D(
            resolution=resolution,
            size_xy=size_xy,
            origin_xy=origin_xy,
            window_size=3,
            probs=probs,
            thresh=0.5,
        )

        x_init = (1.0, 1.0)
        x_goal = (18.0, 18.0)

        astar = AStar(statespace_lo=(0.0, 0.0), statespace_hi=(20.0, 20.0),
                      x_init=x_init, x_goal=x_goal, occupancy=occupancy, resolution=1.0)

        found = astar.solve()
        print("Found path:", found)
        if found:
            print(astar.path)
            try:
                astar.plot_path(fig_num=1)
                plt.show()
            except Exception:
                pass


if __name__ == "__main__":
    main()
