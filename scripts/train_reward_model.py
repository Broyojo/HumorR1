from __future__ import annotations

import time
from timeit import default_timer as timer

import jax
import jax.numpy as jnp
import mpx.config.config_rope as config
import mpx.utils.models_rope as rope_models
import mujoco
import mujoco.viewer
import numpy as np
from mpx.primal_dual_ilqr.primal_dual_ilqr.admm_tvlqr import ADMMConfig
from mpx.primal_dual_ilqr.primal_dual_ilqr.fast_sls import SLSConfig
from mpx.primal_dual_ilqr.primal_dual_ilqr.optimizers import SQPConfig
from mpx.utils.rope_mpc_wrapper import RopeMPCControllerWrapper

# Obstacle definition — solver-native via the constraint and SLS paths.
# Disabled while we confirm baseline gripper tracking; re-enable by replacing
# with `np.array([[cx, cy, cz, r]], dtype=np.float64)`.
OBSTACLES_XYZR_NP = np.zeros((0, 4), dtype=np.float64)
GRIPPER_OBSTACLE_CLEARANCE = 0.030
PLANNER_CONSTRAIN_ROPE_POINTS = False

# Small-motion test goal. Once convergence is clean here, scale up. The
# 30 cm shift is past what this surrogate-XPBD + 0.2 s horizon controller
# can handle without overshoot — see the bad linearization symptoms when
# the arms are far from their initial pose.
LEFT_SHIFT = jnp.array([0.08, 0.0, -0.02], dtype=jnp.float32)
RIGHT_SHIFT = jnp.array([0.05, 0.0, -0.005], dtype=jnp.float32)
PRINT_MODEL_FEASIBILITY = True
PRINT_SIM_FEASIBILITY = True


jax.config.update("jax_compilation_cache_dir", str(config.WORKSPACE_ROOT / "jax_cache"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


class RopeSimulator:
    """MuJoCo simulation where only the KUKA joints are actuated."""

    def __init__(self, model, dt, initial_data: mujoco.MjData | None = None):
        self.model = model
        self.data = mujoco.MjData(model)
        self.dt = dt
        self.pinch_site_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "iiwa14_1/pinch_site"),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "iiwa14/pinch_site"),
        ]
        self.anchor_site_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "actuatedS_first"),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "actuatedS_last"),
        ]
        self.rope_body_ids_all = []
        for body_id in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if name and name.startswith("actuatedB_"):
                self.rope_body_ids_all.append(body_id)

        if not self.rope_body_ids_all:
            raise ValueError("No rope bodies found")

        # The solver state uses every rope body position, not a downsampled rope.
        self.rope_body_ids = list(self.rope_body_ids_all)
        self.sample_count = len(self.rope_body_ids)
        if self.sample_count != config.sample_count:
            raise ValueError(
                "config.sample_count must equal the number of rope bodies so the "
                "solver state is [left_xyz, right_xyz, xyz of every rope body]."
            )

        arm_joint_names = [f"iiwa14_1/joint{i}" for i in range(1, 8)] + [
            f"iiwa14/joint{i}" for i in range(1, 8)
        ]
        arm_actuator_names = [f"iiwa14_1/actuator{i}" for i in range(1, 8)] + [
            f"iiwa14/actuator{i}" for i in range(1, 8)
        ]
        arm_joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in arm_joint_names
        ]
        arm_actuator_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in arm_actuator_names
        ]
        self.arm_dof_ids = np.array(
            [model.jnt_dofadr[jid] for jid in arm_joint_ids], dtype=np.int32
        )
        self.arm_qpos_ids = np.array(
            [model.jnt_qposadr[jid] for jid in arm_joint_ids], dtype=np.int32
        )
        self.arm_joint_ids = np.array(arm_joint_ids, dtype=np.int32)
        self.arm_actuator_ids = np.array(arm_actuator_ids, dtype=np.int32)
        self.arm_joint_names = arm_joint_names
        self.q_lo = model.jnt_range[self.arm_joint_ids, 0].copy()
        self.q_hi = model.jnt_range[self.arm_joint_ids, 1].copy()
        self.arm_slices = (slice(0, 7), slice(7, 14))
        self.n_substeps = max(1, int(round(dt / model.opt.timestep)))
        # torquescale="0" in the weld already breaks the torque path from gripper to
        # rope, so locking roll joints to prevent twist is unnecessary. All 7 joints
        # are free, giving full redundancy for better IK solutions near workspace limits.
        self._roll_mask = np.zeros(len(arm_joint_ids), dtype=bool)

        if initial_data is not None:
            self.data.qpos[:] = initial_data.qpos
            self.data.qvel[:] = initial_data.qvel
            self.data.act[:] = initial_data.act
            if self.data.ctrl.size:
                self.data.ctrl[:] = initial_data.ctrl
            if self.data.mocap_pos.size:
                self.data.mocap_pos[:] = initial_data.mocap_pos
            if self.data.mocap_quat.size:
                self.data.mocap_quat[:] = initial_data.mocap_quat
            mujoco.mj_forward(self.model, self.data)

        self.ctrl_nominal = self.data.qpos[self.arm_qpos_ids].copy()

        if self.data.ctrl.size:
            self.data.ctrl[self.arm_actuator_ids] = self.ctrl_nominal

        self.ik_cartesian_gain = 4.0
        self.ik_joint_blend = 0.4
        self.ik_damping = 5e-2

    def get_coordinate_state(self):
        left = jnp.asarray(
            self.data.site_xpos[self.pinch_site_ids[0]], dtype=jnp.float32
        )
        right = jnp.asarray(
            self.data.site_xpos[self.pinch_site_ids[1]], dtype=jnp.float32
        )
        rope = jnp.asarray(self.data.xpos[self.rope_body_ids], dtype=jnp.float32)
        return rope_models.join_state(left, right, rope)

    def get_state(self):
        left = jnp.asarray(
            self.data.site_xpos[self.pinch_site_ids[0]], dtype=jnp.float32
        )
        right = jnp.asarray(
            self.data.site_xpos[self.pinch_site_ids[1]], dtype=jnp.float32
        )
        rope = jnp.asarray(self.data.xpos[self.rope_body_ids], dtype=jnp.float32)
        rope_quat = jnp.asarray(self.data.xquat[self.rope_body_ids], dtype=jnp.float32)
        return rope_models.join_pose_state(left, right, rope, rope_quat)

    def _pinch_site_step(
        self,
        site_id: int,
        qpos_idx: np.ndarray,
        dof_idx: np.ndarray,
        goal_pos: np.ndarray,
    ) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)

        J = jacp[:, dof_idx]
        pos = np.asarray(self.data.site_xpos[site_id]).copy()
        err = goal_pos - pos

        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(
            JJt + self.ik_damping * np.eye(3),
            self.ik_cartesian_gain * err,
        )

        q_curr = np.asarray(self.data.qpos[qpos_idx]).copy()
        return q_curr + self.ik_joint_blend * dq

    def step(self, u_cmd, x_goal: jnp.ndarray | None = None):
        del x_goal
        u_cmd = np.asarray(u_cmd, dtype=np.float64)

        if u_cmd.shape[0] == 6:
            left_now = np.asarray(self.data.site_xpos[self.pinch_site_ids[0]], dtype=np.float64)
            right_now = np.asarray(self.data.site_xpos[self.pinch_site_ids[1]], dtype=np.float64)
            left_goal = left_now + self.dt * u_cmd[:3]
            right_goal = right_now + self.dt * u_cmd[3:6]

            left_slice, right_slice = self.arm_slices
            q_left_des = self._pinch_site_step(
                self.pinch_site_ids[0],
                self.arm_qpos_ids[left_slice],
                self.arm_dof_ids[left_slice],
                left_goal,
            )
            q_right_des = self._pinch_site_step(
                self.pinch_site_ids[1],
                self.arm_qpos_ids[right_slice],
                self.arm_dof_ids[right_slice],
                right_goal,
            )
            ctrl_target = np.concatenate([q_left_des, q_right_des], axis=0)
        elif u_cmd.shape[0] == len(self.arm_actuator_ids):
            ctrl_target = u_cmd
        else:
            raise ValueError(
                f"Expected 6 Cartesian gripper velocities or {len(self.arm_actuator_ids)} arm actuator targets, got {u_cmd.shape[0]}"
            )

        if self.data.ctrl.size:
            ctrl_target = np.clip(ctrl_target, self.q_lo, self.q_hi)
            ctrl_target[self._roll_mask] = self.ctrl_nominal[self._roll_mask]
            self.data.ctrl[self.arm_actuator_ids] = ctrl_target

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        mujoco.mj_forward(self.model, self.data)
        return self.get_state()

    def max_attachment_error(self) -> float:
        errs = [
            np.linalg.norm(
                self.data.site_xpos[pinch_id] - self.data.site_xpos[anchor_id]
            )
            for pinch_id, anchor_id in zip(self.pinch_site_ids, self.anchor_site_ids)
        ]
        return float(max(errs))

    def hold_current_pose(self) -> None:
        if self.data.ctrl.size:
            hold_q = self.data.qpos[self.arm_qpos_ids].copy()
            hold_q[self._roll_mask] = self.ctrl_nominal[self._roll_mask]
            self.data.ctrl[self.arm_actuator_ids] = hold_q
        self.data.qvel[self.arm_dof_ids] = 0.0


def _task_state(x: jnp.ndarray) -> jnp.ndarray:
    x = jnp.asarray(x, dtype=jnp.float32)
    if x.ndim == 0:
        if hasattr(config, "x0_task"):
            return jnp.asarray(config.x0_task, dtype=jnp.float32)
        raise ValueError(
            "Expected rope state to be a 1-D array, but received a scalar. "
            "Check that config_rope.x0 is exported as the planner/task state."
        )
    if x.shape[0] == config.task_n:
        return x
    return config.state_to_task(x)


def make_gripper_constraints(
    u_min: jnp.ndarray,
    u_max: jnp.ndarray,
    obstacles_xyzr: jnp.ndarray,
    sample_count: int,
    constrain_rope_points: bool = True,
):
    """Control bounds plus 3D obstacle avoidance for grippers, optionally rope."""
    has_obstacles = obstacles_xyzr.shape[0] > 0

    def constraints(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        del t
        u_box = jnp.concatenate([u - u_max, u_min - u], axis=0)

        if not has_obstacles:
            return u_box

        left, right, rope = rope_models.split_state(_task_state(x), sample_count)
        pts_xyz = jnp.concatenate([left[None], right[None]], axis=0)
        if constrain_rope_points:
            pts_xyz = jnp.concatenate([pts_xyz, rope], axis=0)
        obs_centers = obstacles_xyzr[:, :3]
        obs_radii = obstacles_xyzr[:, 3]
        diff = pts_xyz[:, None, :] - obs_centers[None, :, :]
        dist = jnp.linalg.norm(diff, axis=-1) + 1e-6
        obs_c = (obs_radii[None, :] - dist).reshape(-1)

        return jnp.concatenate([u_box, obs_c], axis=0)

    return constraints


def make_constant_disturbance(n: int, alpha: float):
    def disturbance(x_prefix: jnp.ndarray) -> jnp.ndarray:
        t_horizon = x_prefix.shape[0]
        diag = jnp.full(n, alpha, dtype=x_prefix.dtype)
        e0 = jnp.diag(diag)
        return jnp.broadcast_to(e0, (t_horizon, n, n))

    return disturbance


def build_controller(x_init: jnp.ndarray, x_goal: jnp.ndarray):
    x_in = jnp.tile(x_init, (config.N + 1, 1))
    # u_ref holds the settled joint targets — zeros would mean "drive every
    # joint to angle 0" which is far from the home pose. Warm-start at the
    # current pose so the first solve isn't fighting a giant torque transient.
    u_warm = jnp.asarray(config.u_ref, dtype=jnp.float32)
    u_in = jnp.tile(u_warm, (config.N, 1))

    obstacles_xyzr = jnp.asarray(OBSTACLES_XYZR_NP, dtype=jnp.float32)
    obstacles_xyzr = obstacles_xyzr.at[:, 3].add(GRIPPER_OBSTACLE_CLEARANCE)
    constraints = make_gripper_constraints(
        config.u_min,
        config.u_max,
        obstacles_xyzr,
        config.sample_count,
        constrain_rope_points=PLANNER_CONSTRAIN_ROPE_POINTS,
    )
    # SLS robustification against residual XPBD-vs-MuJoCo gap. After fixing
    # the catenary-blending hacks in _xpbd_cosserat_step the per-step gap
    # should be much smaller, but keeping a non-zero disturbance is what
    # makes SLS produce sign-stable controls under any remaining mismatch.
    disturbance = make_constant_disturbance(config.n, alpha=5e-3)

    # 3D obstacles are enforced via `constraints` (make_gripper_constraints).
    # The SLS solver also needs a planar [cx, cy, radius] obstacle list; pass a
    # single far-away dummy so the shape is fixed for JIT.
    obstacles = jnp.array([[1.0e6, 1.0e6, 1.0e-3]], dtype=obstacles_xyzr.dtype)
    num_constraints = (
        int(2 * config.m)
        + (2 + (config.sample_count if PLANNER_CONSTRAIN_ROPE_POINTS else 0))
        * obstacles_xyzr.shape[0]
        + int(obstacles.shape[0])
    )

    admm_config = ADMMConfig(
        eps_abs=5e-3, eps_rel=5e-3, condense_block_size=5, rho_max=120
    )
    # SQP=1 like mjx_quad: re-linearizing XPBD dynamics within one solve makes
    # ADMM diverge (the rope-step Jacobian is poorly conditioned away from the
    # current operating point). Trust one good linearization per MPC tick and
    # let the resolve frequency do the rest.
    sls_config = SLSConfig(
        max_sls_iterations=2, sls_primal_tol=5e-2, enable_fastsls=True
    )
    sqp_config = SQPConfig(max_sqp_iterations=1, line_search=False)

    return RopeMPCControllerWrapper(
        config=config,
        sls_config=sls_config,
        sqp_config=sqp_config,
        admm_config=admm_config,
        constraints=constraints,
        obstacles=obstacles,
        disturbance=disturbance,
        X_in=x_in,
        U_in=u_in,
        num_constraints=num_constraints,
        limited_memory=True,
    )


def print_summary(label: str, x: jnp.ndarray):
    left, right, rope = rope_models.split_state(_task_state(x), config.sample_count)
    rope_mid = rope[config.sample_count // 2]
    rope_tip = rope[-1]
    print(label)
    print("  left pinch :", left)
    print("  right pinch:", right)
    print("  rope mid   :", rope_mid)
    print("  rope tip   :", rope_tip)


def clear_user_geoms(viewer: mujoco.viewer.Handle) -> None:
    viewer.user_scn.ngeom = 0


def add_sphere(
    viewer: mujoco.viewer.Handle,
    pos_xyz: np.ndarray,
    radius: float,
    rgba,
) -> None:
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return

    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, 0.0, 0.0], dtype=np.float64),
        pos=np.array(pos_xyz, dtype=np.float64),
        mat=np.eye(3, dtype=np.float64).ravel(),
        rgba=np.array(rgba, dtype=np.float32),
    )
    viewer.user_scn.ngeom += 1


def add_chain(
    viewer: mujoco.viewer.Handle,
    pts_xyz: np.ndarray,
    radius: float,
    rgba,
    samples_per_segment: int = 4,
) -> None:
    if pts_xyz.shape[0] == 0:
        return
    add_sphere(viewer, pts_xyz[0], radius, rgba)
    for i in range(pts_xyz.shape[0] - 1):
        p0 = pts_xyz[i]
        p1 = pts_xyz[i + 1]
        for s in range(1, samples_per_segment + 1):
            alpha = s / float(samples_per_segment + 1)
            add_sphere(viewer, (1.0 - alpha) * p0 + alpha * p1, radius, rgba)
        add_sphere(viewer, p1, radius, rgba)


def draw_state(
    viewer: mujoco.viewer.Handle,
    x: jnp.ndarray,
    x_goal: jnp.ndarray,
    sim: RopeSimulator | None = None,
) -> None:
    clear_user_geoms(viewer)
    left, right, rope = rope_models.split_state(_task_state(x), config.sample_count)
    left_g, right_g, rope_g = rope_models.split_state(x_goal, config.sample_count)

    left = np.asarray(left)
    right = np.asarray(right)
    rope = np.asarray(rope)
    left_g = np.asarray(left_g)
    right_g = np.asarray(right_g)
    rope_g = np.asarray(rope_g)

    # Red/blue are gripper markers. Orange/green chains are the current/goal rope.
    add_sphere(viewer, left, 0.016, [0.9, 0.1, 0.1, 0.85])
    add_sphere(viewer, right, 0.016, [0.1, 0.1, 0.9, 0.85])
    add_chain(viewer, rope, 0.010, [0.98, 0.68, 0.12, 0.95], samples_per_segment=5)

    add_sphere(viewer, left_g, 0.012, [0.9, 0.6, 0.6, 0.5])
    add_sphere(viewer, right_g, 0.012, [0.6, 0.6, 0.9, 0.5])
    add_chain(viewer, rope_g, 0.007, [0.3, 0.95, 0.3, 0.40], samples_per_segment=3)

    if sim is not None:
        pinch_left = np.asarray(
            sim.data.site_xpos[sim.pinch_site_ids[0]], dtype=np.float64
        )
        pinch_right = np.asarray(
            sim.data.site_xpos[sim.pinch_site_ids[1]], dtype=np.float64
        )
        anchor_left = np.asarray(
            sim.data.site_xpos[sim.anchor_site_ids[0]], dtype=np.float64
        )
        anchor_right = np.asarray(
            sim.data.site_xpos[sim.anchor_site_ids[1]], dtype=np.float64
        )

        # Attachment debug:
        # magenta/cyan = pinch sites on grippers
        # yellow/green = actual rope endpoint sites
        add_sphere(viewer, pinch_left, 0.009, [1.0, 0.0, 1.0, 0.95])
        add_sphere(viewer, pinch_right, 0.009, [0.0, 1.0, 1.0, 0.95])
        add_sphere(viewer, anchor_left, 0.008, [1.0, 1.0, 0.0, 0.95])
        add_sphere(viewer, anchor_right, 0.008, [0.2, 1.0, 0.2, 0.95])
        add_chain(
            viewer,
            np.stack([pinch_left, anchor_left], axis=0),
            0.004,
            [1.0, 0.7, 0.0, 0.85],
            samples_per_segment=2,
        )
        add_chain(
            viewer,
            np.stack([pinch_right, anchor_right], axis=0),
            0.004,
            [0.2, 1.0, 0.6, 0.85],
            samples_per_segment=2,
        )

    # Navigation obstacles: physical sphere + semi-transparent constraint radius
    for obs in OBSTACLES_XYZR_NP:
        cx, cy, cz, r_constraint = (
            float(obs[0]),
            float(obs[1]),
            float(obs[2]),
            float(obs[3]),
        )
        r_physical = r_constraint - 0.015
        add_sphere(viewer, np.array([cx, cy, cz]), r_physical, [1.0, 0.35, 0.05, 0.85])
        add_sphere(
            viewer, np.array([cx, cy, cz]), r_constraint, [1.0, 0.35, 0.05, 0.12]
        )


def _build_reference(x_cur: jnp.ndarray, x_goal: jnp.ndarray, t_ref: jnp.ndarray) -> jnp.ndarray:
    """Linear path to a feasibility-capped terminal state.

    Goals farther than `u_max * N * dt` are physically unreachable in one
    horizon. Asking for them anyway makes the stage cost fight the dynamics
    constraint and ADMM diverges. We rescale the whole goal-delta by a single
    scalar so the terminal lies on the horizon-reach boundary in the goal
    direction — same idea as mjx_quad's velocity command, just expressed as a
    target. Asymmetry between left and right is preserved.
    """
    left_cur, right_cur, rope_cur = rope_models.split_state(
        _task_state(x_cur), config.sample_count
    )
    left_goal, right_goal, _ = rope_models.split_state(x_goal, config.sample_count)
    dleft = left_goal - left_cur
    dright = right_goal - right_cur

    # MJX path: config.u_max is now joint-target ranges (rad), not Cartesian
    # velocities, so derive reach from a sensible Cartesian gripper-speed cap.
    horizon_time = float(config.N) * float(config.dt)
    cart_speed_cap = jnp.array([0.20, 0.20, 0.10], dtype=dleft.dtype)  # m/s per axis
    reach = cart_speed_cap * horizon_time  # (3,)
    over_l = jnp.max(jnp.abs(dleft) / (reach + 1e-9))
    over_r = jnp.max(jnp.abs(dright) / (reach + 1e-9))
    scale_l = jnp.minimum(1.0, 1.0 / (over_l + 1e-9))
    scale_r = jnp.minimum(1.0, 1.0 / (over_r + 1e-9))

    left_term = left_cur + scale_l * dleft
    right_term = right_cur + scale_r * dright

    tau = t_ref.reshape(-1)
    left_path = left_cur[None, :] + tau[:, None] * (left_term - left_cur)[None, :]
    right_path = right_cur[None, :] + tau[:, None] * (right_term - right_cur)[None, :]
    rope_path = jax.vmap(
        lambda l_ref, r_ref: rope_models.rope_reference_from_current(
            left_cur, right_cur, rope_cur, l_ref, r_ref, config.sample_count
        )
    )(left_path, right_path)
    return jax.vmap(rope_models.join_state)(left_path, right_path, rope_path)


def _gripper_goal_error(x_cur: jnp.ndarray, x_goal: jnp.ndarray) -> float:
    left_cur, right_cur, _ = rope_models.split_state(
        _task_state(x_cur), config.sample_count
    )
    left_goal, right_goal, _ = rope_models.split_state(x_goal, config.sample_count)
    delta = jnp.concatenate([left_goal - left_cur, right_goal - right_cur])
    return float(jnp.linalg.norm(delta))


def _gripper_goal_error_max_axis(x_cur: jnp.ndarray, x_goal: jnp.ndarray) -> float:
    left_cur, right_cur, _ = rope_models.split_state(
        _task_state(x_cur), config.sample_count
    )
    left_goal, right_goal, _ = rope_models.split_state(x_goal, config.sample_count)
    delta = jnp.concatenate([left_goal - left_cur, right_goal - right_cur])
    return float(jnp.max(jnp.abs(delta)))


def _print_goal_tracking(x_cur: jnp.ndarray, x_goal: jnp.ndarray) -> None:
    left_cur, right_cur, _ = rope_models.split_state(
        _task_state(x_cur), config.sample_count
    )
    left_goal, right_goal, _ = rope_models.split_state(x_goal, config.sample_count)
    left_err = np.asarray(left_goal - left_cur)
    right_err = np.asarray(right_goal - right_cur)
    print("  left cur   :", np.asarray(left_cur))
    print("  right cur  :", np.asarray(right_cur))
    print("  left goal  :", np.asarray(left_goal))
    print("  right goal :", np.asarray(right_goal))
    print("  left error :", left_err, f"|norm|={np.linalg.norm(left_err):.4f}")
    print("  right error:", right_err, f"|norm|={np.linalg.norm(right_err):.4f}")


def _summarize_residuals(label: str, residuals: np.ndarray) -> None:
    sq_norms = np.sum(residuals * residuals, axis=1)
    worst_idx = int(np.argmax(sq_norms))
    print(
        f"{label}: rms={float(np.sqrt(np.mean(sq_norms))):.4e}, "
        f"max={float(np.sqrt(np.max(sq_norms))):.4e} at step {worst_idx}, "
        f"final={float(np.sqrt(sq_norms[-1])):.4e}"
    )


def _model_horizon_residuals(
    X: jnp.ndarray,
    U: jnp.ndarray,
    parameter: jnp.ndarray | None,
) -> np.ndarray:
    residuals = []
    for t in range(U.shape[0]):
        x_next_pred = config.dynamics(
            X[t], U[t], jnp.asarray(float(t), dtype=X.dtype), parameter
        )
        residuals.append(np.asarray(X[t + 1] - x_next_pred, dtype=np.float64))
    return np.stack(residuals, axis=0)


def _sim_horizon_residuals(
    sim: RopeSimulator,
    X: jnp.ndarray,
    U: jnp.ndarray,
) -> np.ndarray:
    probe = RopeSimulator(sim.model, sim.dt, initial_data=sim.data)
    if probe.data.ctrl.size:
        probe.data.ctrl[:] = sim.data.ctrl

    residuals = []
    for t in range(U.shape[0]):
        x_next_sim = probe.step(np.asarray(U[t], dtype=np.float64))
        residuals.append(np.asarray(X[t + 1] - x_next_sim, dtype=np.float64))
    return np.stack(residuals, axis=0)


def main() -> None:
    x0 = jnp.asarray(config.x0, dtype=jnp.float32)
    if x0.ndim == 0:
        if hasattr(config, "x0"):
            maybe_x0 = jnp.asarray(config.x0, dtype=jnp.float32)
            if maybe_x0.ndim == 1:
                x0 = maybe_x0
        if x0.ndim == 0 and hasattr(config, "x0_task"):
            x0 = jnp.asarray(config.x0_task, dtype=jnp.float32)
        if x0.ndim == 0:
            raise ValueError(
                "config.x0 is scalar; expected planner state vector. "
                "Re-sync config_rope.py so x0 is built from rope positions/quaternions."
            )

    left_shift = LEFT_SHIFT
    right_shift = RIGHT_SHIFT

    left_goal = config.left0 + left_shift
    right_goal = config.right0 + right_shift
    left_start, right_start, rope_start = rope_models.split_state(
        _task_state(x0), config.sample_count
    )
    rope_goal = rope_models.rope_reference_from_current(
        left_start,
        right_start,
        rope_start,
        left_goal,
        right_goal,
        config.sample_count,
    )
    x_goal = rope_models.join_state(left_goal, right_goal, rope_goal)

    print_summary("Initial state", x0)
    print("Active left shift :", np.asarray(left_shift))
    print("Active right shift:", np.asarray(right_shift))
    print_summary("Goal state", x_goal)

    sim = RopeSimulator(config.model, config.dt, initial_data=config.data)
    x = sim.get_state()
    parameter = None

    mpc = build_controller(x0, x_goal)

    t_ref = jnp.linspace(0.0, 1.0, config.N + 1, dtype=jnp.float32)[:, None]
    reference = _build_reference(x, x_goal, t_ref)
    u0, X, U, V, backoffs, Phi_x, Phi_u = mpc.run(x, reference, parameter)
    del V, backoffs, Phi_x, Phi_u

    print("\nPlanner solved (warm-up).")
    print(f"State dim={config.n}, control dim={config.m}, horizon={config.N}")
    print("First control:", u0)
    print_summary("Planned next state", X[1])
    print_summary("Planned terminal state", X[-1])

    q0 = sim.data.qpos[sim.arm_qpos_ids]
    arm_names = [f"iiwa14_1/j{i}" for i in range(1, 8)] + [
        f"iiwa14/j{i}" for i in range(1, 8)
    ]
    print("\nInitial joint margins (distance to each limit):")
    for name, q, lo, hi in zip(arm_names, q0, sim.q_lo, sim.q_hi):
        bar = "#" * int(20 * (q - lo) / max(hi - lo, 1e-6))
        print(f"  {name:14s} [{lo:+.2f} .. {q:+.2f} .. {hi:+.2f}]  |{bar:<20}|")

    sim_hz = 50.0
    requested_mpc_hz = float(config.mpc_frequency)
    mpc_hz = min(requested_mpc_hz, 10.0)
    sim_steps_per_mpc = max(1, int(round(sim_hz / min(mpc_hz, sim_hz))))
    counter = 0
    u_cmd = jnp.zeros(config.m)
    goal_reached = False
    x_track = x_goal

    print(
        f"Simulation Hz: {sim_hz}, MPC Hz: {mpc_hz} (requested {requested_mpc_hz}), Steps per MPC: {sim_steps_per_mpc}"
    )

    with mujoco.viewer.launch_passive(config.model, sim.data) as viewer:
        while viewer.is_running():
            tic = time.time()

            if (counter % sim_steps_per_mpc == 0) and not goal_reached:
                reference = _build_reference(x, x_goal, t_ref)
                parameter = None
                # XPBD path needed cold-start because the surrogate disagreed
                # with MuJoCo. With exact MJX dynamics the shifted warm-start
                # is valid — kept the call commented for easy fallback.
                # mpc._reset_warm_start(x)
                start = timer()
                u_mpc, X, U, V, backoffs, Phi_x, Phi_u = mpc.run(
                    x, reference, parameter
                )
                x_track = X[1]
                u_cmd = jnp.clip(u_mpc, config.u_min, config.u_max)
                solve_dt = timer() - start
                print(
                    f"MPC solve time: {solve_dt:.4f}s, "
                    f"mpc_control={np.asarray(u_mpc)}, executed_control={np.asarray(u_cmd)}"
                )
                if PRINT_MODEL_FEASIBILITY:
                    _summarize_residuals(
                        "Model horizon residual",
                        _model_horizon_residuals(X, U, parameter),
                    )
                if PRINT_SIM_FEASIBILITY:
                    _summarize_residuals(
                        "Sim horizon residual",
                        _sim_horizon_residuals(sim, X, U),
                    )
                print(f"Attachment error: {sim.max_attachment_error():.4f} m")
                _print_goal_tracking(x, x_goal)
                del X, U, V, backoffs, Phi_x, Phi_u

                goal_err = _gripper_goal_error(x, x_goal)
                goal_err_axis = _gripper_goal_error_max_axis(x, x_goal)
                if goal_err < 0.025 or goal_err_axis < 0.015:
                    goal_reached = True
                    u_cmd = jnp.zeros((config.m,), dtype=jnp.float32)
                    x_track = x_goal
                    sim.hold_current_pose()
                    print(
                        "Goal reached, holding pose. "
                        f"Gripper error: {goal_err:.4f} m, max-axis error: {goal_err_axis:.4f} m"
                    )
            elif goal_reached:
                u_cmd = jnp.zeros((config.m,), dtype=jnp.float32)
                x_track = x_goal
                sim.hold_current_pose()

            x = sim.step(u_cmd, x_goal)

            draw_state(viewer, x, x_goal, sim)
            viewer.sync()
            counter += 1

            elapsed = time.time() - tic
            sleep_dt = (1.0 / sim_hz) - elapsed
            if sleep_dt > 0:
                time.sleep(sleep_dt)


if __name__ == "__main__":
    main()
