# E028 Draft

This is a non-canonical evidence draft for Gate B. It is intended for review, not for direct promotion.

## Headline

Gate B does **not** support a pure "teacher target is not executable in DrawerRobotEnv" story.

Observed aggregate:
- `open_loop_attach_rate = 0.9`
- `open_loop_success_rate = 0.5`
- `state_anchored_attach_rate = 1.0`
- `state_anchored_success_rate = 0.6`
- `oracle_reachability_rate = 1.0`
- `decision_support = teacher_actions_executable_policy_gap_remains`

## What this draft supports

- Teacher actions are substantially executable in the current env.
- Oracle reachability shows the current env task is stably reachable.
- The remaining failure pattern is mostly `pull_no_open`, not broad failure to approach or attach.

## What this draft does not support

- Final root cause solved
- Physics confirmed
- Image domain eliminated
- Policy learnability confirmed

## Important caveat

The Gate B top-10 NPZs are legacy rollout files and do not include full native teacher-rollout metadata such as `grasp_pose_local_*`. The collector reconstructs handle semantics from the current env. This makes the audit useful and discriminative, but it is not a perfect proof of exact source-grasp replay fidelity.
