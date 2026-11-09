# RG Service Center · Third-party notices

## Franka Emika Panda model and meshes

Source: https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda
Vendored files: `public/models/panda/panda.xml`, `public/models/panda/assets/*`.
License: Apache-2.0. Full license: `public/models/panda/LICENSE`.
The upstream model README is retained. Source model derives from Franka ROS descriptions.
The model is loaded from local assets; the workcell does not require external asset requests.
Our runtime adds the conveyor, receptacle, target can, TCP site and gravity compensation.

## Franka analytical inverse kinematics

Source: https://github.com/Chidanand26/franka-arm-ik-sim
Adapted file: `src/sim/FrankaAnalyticalIK.ts`.
The source file carries `SPDX-License-Identifier: Apache-2.0`, preserved in the file.
The upstream repository additionally declares MIT; its full repository license is retained at
`public/models/panda/IK_REPOSITORY_LICENSE`. The Apache-2.0 text is retained above.
Modification: TCP offset aligned to the retained Panda model (0.1034 m).
The solver credits Yanhao He and Steven Liu's analytical inverse-kinematics work.

## Runtime dependencies

MuJoCo WASM (`mujoco-js`) is Apache-2.0. Three.js and React are MIT licensed.
Their package notices remain in installed dependencies.

Spark (`@sparkjsdev/spark`) is MIT licensed; its full license is retained at
`public/worlds/factory/SPARK_LICENSE.txt`.

## Generated factory environment

The active `public/worlds/factory-conveyor-hq/` environment is a can recycling and
conveyor sorting factory generated with World Labs Marble 1.1 from a purpose-built
OpenAI image_gen cube-face panorama assembled at4096 ×2048, with neutral gray bales. The panorama is AI-generated, not a real facility
photograph. Source, prompt, model, world identity, API cost, input hash and scale
provenance are retained in that package. Spark's MIT license is included there.
The collider and full SPZ are used offline for alignment; they do not replace the
native task's MuJoCo collision geometry or change policy camera observations.

Earlier `public/worlds/factory/` Draft assets retain their own world.json identity.
Earlier `public/worlds/factory-standard/` uses the real Poly Haven University
Workshop reference by Savva Zakharov, CC0 1.0; see its ATTRIBUTION.txt. These are
separate packages and are not reused as the active can-recycling reference.

## Demonstration semantics

### Offline browser demo

The offline autonomous controller is a scripted IK state machine, not a trained VLA.
MuJoCo owns robot and target-can physics. No object attachment is used to imitate a grasp.
Success requires the target can to be physically inside the metal receptacle.
Conveyor transport is an approximation: a MuJoCo free-body velocity servo moves the can during feeding. The belt then stops before contact-based grasping. It is not a detailed frictional belt model.
The offline factory scene is custom and is not a LIBERO benchmark task.

### Native GPU mode

The separate native mode runs the pinned pretrained SmolVLA checkpoint in actual LIBERO simulation. The browser renders native state; the generated factory backdrop does not change the simulator physics or policy observations. Human operators perform recovery, and task success comes from native LIBERO `check_success()`. This mode does not establish a trained recovery policy or real-world safety. See the [inference guide](../inference/README.md) for framework and checkpoint pins.

## LIBERO alphabet-soup visual asset

Source: https://github.com/Lifelong-Robot-Learning/LIBERO/tree/master/libero/libero/assets/stable_hope_objects/alphabet_soup
Original scanned visual mesh, XML and texture are retained under `public/models/libero-alphabet-soup/`.
The LIBERO repository license is retained there as `LIBERO_LICENSE`. The asset derives from the HOPE object collection used by LIBERO.
The offline demo fits the scanned mesh to its existing cylindrical collision dimensions; physics XML is unchanged.
The native renderer instead uses server MJCF geometry, material texture pixels and indexed mesh UVs without changing object dimensions or relative poses.
