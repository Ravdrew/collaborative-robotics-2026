#!/usr/bin/env python3
"""
TidyBot2 Bimanual Shelf Demo - Realistic Water Bottle Meshes

Same bimanual pick-and-place demo as shelf_demo_sim.py, but uses water bottle
meshes instead of primitive shapes. Both arms pick up water bottles from the
ground, drive toward a shelf, and place them on top.

Usage:
    cd simulation/scripts
    uv run python shelf_demo_sim_realistic.py
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path

import mink


# Phase durations: (cumulative_start_time, duration)
PHASE_TIMING = {
    "approach":       (0.0,  5.0),
    "descend":        (5.0,  3.0),
    "grasp":          (8.0,  2.0),
    "lift":           (10.0, 3.0),
    "drive":          (13.0, 3.0),
    "place":          (16.0, 3.0),
    "release":        (19.0, 1.5),
    "retract":        (20.5, 3.0),
    "done":           (23.5, float("inf")),
}

DRIVE_DISTANCE = 0.45  # base drives from X=0 to X=0.45

GRIPPER_OPEN = 0.037
# Grip waist radius ~0.019m; gripper closed to match
RIGHT_GRIPPER_CLOSED = 0.022
LEFT_GRIPPER_CLOSED = 0.022

SLEEP_POSE = [0.0, -1.80, 1.55, 0.0, 0.8, 0.0]


def set_grippers(data, gripper_actuators, value):
    """Set all gripper finger actuators to the same value."""
    for act_id in gripper_actuators.values():
        data.ctrl[act_id] = value


def lerp(a, b, t):
    """Linear interpolation between arrays a and b by factor t in [0,1]."""
    return a + np.clip(t, 0.0, 1.0) * (b - a)


def main():
    # ==========================================================================
    # Load Model
    # ==========================================================================
    script_dir = Path(__file__).parent
    model_path = script_dir / "../assets/mujoco/scene_shelf_demo_realistic.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path.resolve()))
    data = mujoco.MjData(model)

    # ==========================================================================
    # Get IDs
    # ==========================================================================
    right_obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_object")
    left_obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_object")

    right_arm_joints = [
        "right_waist", "right_shoulder", "right_elbow",
        "right_forearm_roll", "right_wrist_angle", "right_wrist_rotate",
    ]
    left_arm_joints = [
        "left_waist", "left_shoulder", "left_elbow",
        "left_forearm_roll", "left_wrist_angle", "left_wrist_rotate",
    ]
    all_arm_joints = right_arm_joints + left_arm_joints

    joint_qpos_addrs = {}
    for jname in all_arm_joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        joint_qpos_addrs[jname] = model.jnt_qposadr[jid]

    arm_actuator_ids = {}
    for jname in all_arm_joints:
        arm_actuator_ids[jname] = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname
        )

    right_gripper_actuators = {
        "right_left_finger": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_left_finger"),
        "right_right_finger": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_right_finger"),
    }
    left_gripper_actuators = {
        "left_left_finger": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_left_finger"),
        "left_right_finger": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_right_finger"),
    }
    all_gripper_actuators = {**right_gripper_actuators, **left_gripper_actuators}

    right_obj_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_object_joint")
    left_obj_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_object_joint")
    right_obj_qpos_addr = model.jnt_qposadr[right_obj_jnt_id]
    left_obj_qpos_addr = model.jnt_qposadr[left_obj_jnt_id]
    right_obj_dof_addr = model.jnt_dofadr[right_obj_jnt_id]
    left_obj_dof_addr = model.jnt_dofadr[left_obj_jnt_id]

    right_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_pinch_site")
    left_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_pinch_site")

    gripper_joint_qpos = {}
    for fname in ["right_left_finger", "right_right_finger",
                   "left_left_finger", "left_right_finger"]:
        fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, fname)
        gripper_joint_qpos[fname] = model.jnt_qposadr[fid]

    base_x_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint_x")
    base_x_qpos_addr = model.jnt_qposadr[base_x_jnt_id]
    base_x_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint_x")

    theta_qpos_addr = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint_th")]
    theta_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint_th")

    # ==========================================================================
    # Hide world origin axes and site (cosmetic geoms in the robot model)
    # ==========================================================================
    for axis_name in ["world_x_axis", "world_y_axis", "world_z_axis"]:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, axis_name)
        if gid >= 0:
            model.geom_rgba[gid] = [0, 0, 0, 0]  # fully transparent
    origin_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "world_origin")
    if origin_sid >= 0:
        model.site_rgba[origin_sid] = [0, 0, 0, 0]

    # ==========================================================================
    # Disable collision on finger link geoms (objects tracked kinematically)
    # ==========================================================================
    finger_bodies = []
    for side in ["right", "left"]:
        finger_bodies.extend([
            f"{side}_left_finger_link",
            f"{side}_right_finger_link",
        ])
    finger_body_ids = set()
    for bname in finger_bodies:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        if bid >= 0:
            finger_body_ids.add(bid)
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] in finger_body_ids:
            model.geom_contype[gid] = 0
            model.geom_conaffinity[gid] = 0

    # ==========================================================================
    # Helper functions
    # ==========================================================================
    def reset_to_initial():
        """Reset robot to initial state (sleep pose, theta=pi/2, grippers open)."""
        mujoco.mj_resetData(model, data)
        data.qpos[theta_qpos_addr] = np.pi / 2
        data.ctrl[theta_act_id] = np.pi / 2
        for arm_joints in [right_arm_joints, left_arm_joints]:
            for jname, val in zip(arm_joints, SLEEP_POSE):
                data.qpos[joint_qpos_addrs[jname]] = val
                data.ctrl[arm_actuator_ids[jname]] = val
        for fname in gripper_joint_qpos:
            data.qpos[gripper_joint_qpos[fname]] = GRIPPER_OPEN
        set_grippers(data, all_gripper_actuators, GRIPPER_OPEN)
        mujoco.mj_forward(model, data)

    def set_arm_joints(right_q, left_q):
        """Apply arm joint angles from numpy arrays to qpos and ctrl."""
        for i, jname in enumerate(right_arm_joints):
            data.qpos[joint_qpos_addrs[jname]] = right_q[i]
            data.ctrl[arm_actuator_ids[jname]] = right_q[i]
        for i, jname in enumerate(left_arm_joints):
            data.qpos[joint_qpos_addrs[jname]] = left_q[i]
            data.ctrl[arm_actuator_ids[jname]] = left_q[i]

    def get_arm_joints():
        """Read current arm joint angles as (right_q, left_q) numpy arrays."""
        r = np.array([data.qpos[joint_qpos_addrs[j]] for j in right_arm_joints])
        l = np.array([data.qpos[joint_qpos_addrs[j]] for j in left_arm_joints])
        return r, l

    # ==========================================================================
    # Initialize and read object positions
    # ==========================================================================
    reset_to_initial()

    right_obj_pos = data.xpos[right_obj_body_id].copy()
    left_obj_pos = data.xpos[left_obj_body_id].copy()
    print(f"Right bottle position: {right_obj_pos}")
    print(f"Left bottle position:  {left_obj_pos}")

    # ==========================================================================
    # Setup Mink IK (for offline pre-computation only)
    # ==========================================================================
    configuration = mink.Configuration(model)
    configuration.update(data.qpos)

    right_ee_task = mink.FrameTask(
        frame_name="right_pinch_site", frame_type="site",
        position_cost=1.0, orientation_cost=0.1,
    )
    left_ee_task = mink.FrameTask(
        frame_name="left_pinch_site", frame_type="site",
        position_cost=1.0, orientation_cost=0.1,
    )
    # Position-only tasks (no orientation constraint) for approach
    right_ee_task_pos = mink.FrameTask(
        frame_name="right_pinch_site", frame_type="site",
        position_cost=1.0, orientation_cost=0.0,
    )
    left_ee_task_pos = mink.FrameTask(
        frame_name="left_pinch_site", frame_type="site",
        position_cost=1.0, orientation_cost=0.0,
    )

    top_down_quat = np.array([0, 1, 0, 0])  # wxyz: gripper pointing down

    # Freeze non-arm joints during IK pre-computation
    arm_joint_set = set(all_arm_joints)
    vel_limits = {}
    for i in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if jname:
            jtype = model.jnt_type[i]
            if jtype == mujoco.mjtJoint.mjJNT_FREE:
                continue
            if jname not in arm_joint_set:
                vel_limits[jname] = np.array([1e-10])
    freeze_limit = mink.VelocityLimit(model, vel_limits)
    config_limit = mink.ConfigurationLimit(model)

    def solve_ik_to(right_target, left_target, n_steps=500,
                    r_task=None, l_task=None):
        """Iteratively solve IK from current data.qpos toward targets.
        Uses only joint limits and freeze constraint (no collision avoidance)
        for reliable convergence. Returns (right_q, left_q) arrays.
        Pass r_task/l_task to override the default orientation-aware tasks."""
        r_task = r_task or right_ee_task
        l_task = l_task or left_ee_task
        ik_dt = 0.02
        for step in range(n_steps):
            configuration.update(data.qpos)
            r_pose = mink.SE3.from_rotation_and_translation(
                rotation=mink.SO3(top_down_quat), translation=right_target)
            l_pose = mink.SE3.from_rotation_and_translation(
                rotation=mink.SO3(top_down_quat), translation=left_target)
            r_task.set_target(r_pose)
            l_task.set_target(l_pose)
            try:
                vel = mink.solve_ik(
                    configuration, [r_task, l_task],
                    dt=ik_dt, solver="quadprog", damping=1e-1,
                    limits=[freeze_limit, config_limit])
                configuration.integrate_inplace(vel, ik_dt)
                for jname in all_arm_joints:
                    addr = joint_qpos_addrs[jname]
                    data.qpos[addr] = configuration.q[addr]
                    data.ctrl[arm_actuator_ids[jname]] = configuration.q[addr]
            except mink.exceptions.NoSolutionFound:
                print(f"    IK pre-solve stopped at step {step}")
                break
        mujoco.mj_forward(model, data)
        return get_arm_joints()

    # ==========================================================================
    # Define waypoints (world-frame positions for pinch sites)
    # Bottles: 14cm tall, sitting on floor (bottom z=0, top z=0.14)
    #   Body cylinder: z=0.008-0.082
    #   Grip waist:    z=0.048-0.058
    #   Shoulder:      z=0.088-0.100
    # Grasp at the grip waist for a secure hold (~z=0.053)
    # ==========================================================================
    right_grasp_z = 0.13
    left_grasp_z = 0.13

    right_approach = np.array([right_obj_pos[0], right_obj_pos[1], right_obj_pos[2] + 0.18])
    left_approach = np.array([left_obj_pos[0], left_obj_pos[1], left_obj_pos[2] + 0.18])

    right_grasp = np.array([right_obj_pos[0], right_obj_pos[1], right_grasp_z])
    left_grasp = np.array([left_obj_pos[0], left_obj_pos[1], left_grasp_z])

    right_lift = np.array([0.32, right_obj_pos[1], 0.55])
    left_lift = np.array([0.32, left_obj_pos[1], 0.55])

    shelf_place_x = 0.85
    shelf_z = 0.30
    # Gripper at grasp_z on bottle, so bottle bottom is grasp_z below gripper.
    # Place so bottle clears shelf: pinch z = shelf_z + grasp_z + margin
    right_place = np.array([shelf_place_x, -0.10, shelf_z + right_grasp_z + 0.05])
    left_place = np.array([shelf_place_x, 0.10, shelf_z + left_grasp_z + 0.05])

    print(f"\nWaypoints:")
    print(f"  Right approach:  {right_approach}")
    print(f"  Right grasp:     {right_grasp}")
    print(f"  Right lift:      {right_lift}")
    print(f"  Right shelf:     {right_place}")
    print(f"  Left approach:   {left_approach}")
    print(f"  Left grasp:      {left_grasp}")
    print(f"  Left lift:       {left_lift}")
    print(f"  Left shelf:      {left_place}")

    # ==========================================================================
    # Pre-compute joint angles for all waypoints (offline, no collision checks)
    # Each solve starts from the previous waypoint's configuration.
    # ==========================================================================
    SLEEP_Q = np.array(SLEEP_POSE)
    print("\nPre-computing joint angles...")

    # Approach (from sleep, base at X=0) - position only, no orientation constraint
    right_approach_q, left_approach_q = solve_ik_to(
        right_approach, left_approach,
        r_task=right_ee_task_pos, l_task=left_ee_task_pos)
    print(f"  approach:      right={np.round(right_approach_q, 3)}")

    # Grasp (from approach)
    right_grasp_q, left_grasp_q = solve_ik_to(
        right_grasp, left_grasp,
        r_task=right_ee_task_pos, l_task=left_ee_task_pos)
    print(f"  grasp:         right={np.round(right_grasp_q, 3)}")

    # Lift (from grasp)
    right_lift_q, left_lift_q = solve_ik_to(
        right_lift, left_lift,
        r_task=right_ee_task_pos, l_task=left_ee_task_pos)
    print(f"  lift:          right={np.round(right_lift_q, 3)}")

    # --- Move base forward for post-drive waypoints ---
    data.qpos[base_x_qpos_addr] = DRIVE_DISTANCE
    data.ctrl[base_x_act_id] = DRIVE_DISTANCE
    mujoco.mj_forward(model, data)

    # Place (from lift joints, base at DRIVE_DISTANCE)
    right_place_q, left_place_q = solve_ik_to(
        right_place, left_place,
        r_task=right_ee_task_pos, l_task=left_ee_task_pos)
    print(f"  place:         right={np.round(right_place_q, 3)}")

    # Retract: return to sleep pose (no IK needed)
    right_retract_q = SLEEP_Q.copy()
    left_retract_q = SLEEP_Q.copy()
    print(f"  retract:       right={np.round(right_retract_q, 3)}")

    print("Pre-computation complete.\n")

    # ==========================================================================
    # Reset to initial state for the actual simulation
    # ==========================================================================
    reset_to_initial()

    # ==========================================================================
    # Run Simulation (joint-space interpolation only - no real-time IK)
    # ==========================================================================
    print("Starting bimanual shelf demo (realistic water bottles)...")

    # Start viewer maximized via GLFW hint
    try:
        import glfw
        glfw.init()
        glfw.window_hint(glfw.MAXIMIZED, glfw.TRUE)
    except Exception:
        pass

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 1.5
        viewer.cam.elevation = -25
        viewer.cam.azimuth = 160
        viewer.cam.lookat[:] = [0.35, 0.0, 0.30]
        viewer.opt.frame = 0  # hide coordinate frames
        viewer.opt.sitegroup[:] = 0  # hide all sites (world_origin, pinch sites)

        phase = "approach"
        prev_phase = None
        dt = model.opt.timestep
        sim_time = -3.0  # 3-second pause before motion starts
        step_count = 0

        objects_attached = False
        right_obj_offset = np.zeros(3)
        left_obj_offset = np.zeros(3)

        while viewer.is_running():
            step_start = time.time()
            elapsed = sim_time

            # ==================================================================
            # State Machine - interpolate pre-computed joint angles
            # ==================================================================
            right_gripper_target = None
            left_gripper_target = None

            if phase == "approach":
                t_start, t_dur = PHASE_TIMING["approach"]
                t = (elapsed - t_start) / t_dur
                right_q = lerp(SLEEP_Q, right_approach_q, t)
                left_q = lerp(SLEEP_Q, left_approach_q, t)
                if elapsed >= t_start + t_dur:
                    phase = "descend"

            elif phase == "descend":
                t_start, t_dur = PHASE_TIMING["descend"]
                t = (elapsed - t_start) / t_dur
                right_q = lerp(right_approach_q, right_grasp_q, t)
                left_q = lerp(left_approach_q, left_grasp_q, t)
                if elapsed >= t_start + t_dur:
                    phase = "grasp"

            elif phase == "grasp":
                right_q = right_grasp_q
                left_q = left_grasp_q
                t_start, t_dur = PHASE_TIMING["grasp"]
                progress = min((elapsed - t_start) / t_dur, 1.0)
                right_gripper_target = GRIPPER_OPEN + progress * (RIGHT_GRIPPER_CLOSED - GRIPPER_OPEN)
                left_gripper_target = GRIPPER_OPEN + progress * (LEFT_GRIPPER_CLOSED - GRIPPER_OPEN)
                if elapsed >= t_start + t_dur:
                    rp = data.site_xpos[right_site_id].copy()
                    lp = data.site_xpos[left_site_id].copy()
                    right_obj_offset = data.xpos[right_obj_body_id].copy() - rp
                    left_obj_offset = data.xpos[left_obj_body_id].copy() - lp
                    objects_attached = True
                    phase = "lift"

            elif phase == "lift":
                t_start, t_dur = PHASE_TIMING["lift"]
                t = (elapsed - t_start) / t_dur
                right_q = lerp(right_grasp_q, right_lift_q, t)
                left_q = lerp(left_grasp_q, left_lift_q, t)
                right_gripper_target = RIGHT_GRIPPER_CLOSED
                left_gripper_target = LEFT_GRIPPER_CLOSED
                if elapsed >= t_start + t_dur:
                    phase = "drive"

            elif phase == "drive":
                t_start, t_dur = PHASE_TIMING["drive"]
                t = min((elapsed - t_start) / t_dur, 1.0)
                base_x = t * DRIVE_DISTANCE
                data.qpos[base_x_qpos_addr] = base_x
                data.ctrl[base_x_act_id] = base_x
                right_q = right_lift_q
                left_q = left_lift_q
                right_gripper_target = RIGHT_GRIPPER_CLOSED
                left_gripper_target = LEFT_GRIPPER_CLOSED
                if elapsed >= t_start + t_dur:
                    phase = "place"

            elif phase == "place":
                t_start, t_dur = PHASE_TIMING["place"]
                t = (elapsed - t_start) / t_dur
                right_q = lerp(right_lift_q, right_place_q, t)
                left_q = lerp(left_lift_q, left_place_q, t)
                right_gripper_target = RIGHT_GRIPPER_CLOSED
                left_gripper_target = LEFT_GRIPPER_CLOSED
                if elapsed >= t_start + t_dur:
                    # Release objects immediately so they fall under gravity
                    objects_attached = False
                    data.qvel[right_obj_dof_addr:right_obj_dof_addr + 6] = 0
                    data.qvel[left_obj_dof_addr:left_obj_dof_addr + 6] = 0
                    phase = "release"

            elif phase == "release":
                right_q = right_place_q
                left_q = left_place_q
                t_start, t_dur = PHASE_TIMING["release"]
                progress = min((elapsed - t_start) / t_dur, 1.0)
                right_gripper_target = RIGHT_GRIPPER_CLOSED + progress * (GRIPPER_OPEN - RIGHT_GRIPPER_CLOSED)
                left_gripper_target = LEFT_GRIPPER_CLOSED + progress * (GRIPPER_OPEN - LEFT_GRIPPER_CLOSED)
                if elapsed >= t_start + t_dur:
                    phase = "retract"

            elif phase == "retract":
                t_start, t_dur = PHASE_TIMING["retract"]
                t = (elapsed - t_start) / t_dur
                right_q = lerp(right_place_q, right_retract_q, t)
                left_q = lerp(left_place_q, left_retract_q, t)
                right_gripper_target = GRIPPER_OPEN
                left_gripper_target = GRIPPER_OPEN
                if elapsed >= t_start + t_dur:
                    phase = "done"

            else:  # done
                right_q = right_retract_q
                left_q = left_retract_q
                right_gripper_target = GRIPPER_OPEN
                left_gripper_target = GRIPPER_OPEN

            # Print phase transitions
            if phase != prev_phase:
                print(f"Phase: {phase} (t={elapsed:.1f}s)")
                prev_phase = phase

            # ==================================================================
            # Apply joint angles directly (no IK needed)
            # ==================================================================
            set_arm_joints(right_q, left_q)

            # Apply grippers
            if right_gripper_target is not None:
                set_grippers(data, right_gripper_actuators, right_gripper_target)
                for fname, addr in gripper_joint_qpos.items():
                    if fname.startswith("right_"):
                        data.qpos[addr] = right_gripper_target
            if left_gripper_target is not None:
                set_grippers(data, left_gripper_actuators, left_gripper_target)
                for fname, addr in gripper_joint_qpos.items():
                    if fname.startswith("left_"):
                        data.qpos[addr] = left_gripper_target

            # Track objects when attached to grippers
            if objects_attached:
                rp = data.site_xpos[right_site_id]
                lp = data.site_xpos[left_site_id]
                data.qpos[right_obj_qpos_addr:right_obj_qpos_addr + 3] = rp + right_obj_offset
                data.qpos[left_obj_qpos_addr:left_obj_qpos_addr + 3] = lp + left_obj_offset
                data.qvel[right_obj_dof_addr:right_obj_dof_addr + 6] = 0
                data.qvel[left_obj_dof_addr:left_obj_dof_addr + 6] = 0

            # ==================================================================
            # Step Simulation
            # ==================================================================
            mujoco.mj_step(model, data)
            step_count += 1
            sim_time += dt

            # Camera follows the robot
            base_x_now = data.qpos[base_x_qpos_addr]
            viewer.cam.lookat[0] = 0.35 + base_x_now

            # Hide all coordinate frames every frame (viewer may reset options)
            viewer.opt.frame = 0

            viewer.sync()

            # Real-time sync
            elapsed_step = time.time() - step_start
            sleep_time = dt - elapsed_step
            if sleep_time > 0:
                time.sleep(sleep_time)

    print("Simulation ended.")


if __name__ == "__main__":
    main()
