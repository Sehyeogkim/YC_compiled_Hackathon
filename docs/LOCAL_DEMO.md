# Current presentation UI

Run `npm ci` and `npm run dev` from `control-ui/`, then open **http://127.0.0.1:5173/?local=1**.

The header reads **RECORDED SMOLVLA → LOCAL HUMAN CONTROL**. The right panel contains **Play recorded SmolVLA**, **Take Control · record human**, **Recovery complete · keep control**, and **Finish & download**.

## Native scene prerequisite

The local workcell needs a native scene exported by the pinned LIBERO runtime. The development plugin reads it from:

```text
control-ui/.cache/native-scenes/4a7eb0dbbe6e6b968ee9e5589df8b537a72659635525a18fc631b3d231723104.json
```

This large generated scene is not included in GitHub. It is required even though local playback/control does not require a running GPU once prepared. The plugin serves it at `/local-scene?expected_scene_sha256=...`. Use the matching scene from the pinned native runtime; an arbitrary MuJoCo scene is not interchangeable. See [native runtime setup](SETUP.md) and [the scene API](../inference/PROTOCOL.md).

The small recorded policy trajectory is included in `control-ui/public/replay/base-flail.json`. Its metadata identifies the base checkpoint and LIBERO I/O adapter. It is recorded playback, not live inference.

## Operator flow

1. Play the recorded SmolVLA motion.
2. Take control to record human recovery in local MuJoCo.
3. Mark recovery complete while retaining human control for task continuation.
4. Finish and download the local intervention and video capture.

Mouse controls XY, the wheel adjusts height, and click/Space operates the gripper. WASD moves XY, R/F adjusts height, Q/E changes yaw, and arrow keys control pitch/roll. Enter marks recovery complete; Esc pauses/releases the mouse.

Local records preserve the control interaction and phase annotation. They require native-camera reconstruction, action/schema checks, and qualification before VLA training. A replay or manual demonstration is not evidence of learned recovery improvement.
