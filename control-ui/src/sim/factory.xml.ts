export function factoryXml(panda: string) {
  let robot = panda.replace(/<keyframe>[\s\S]*?<\/keyframe>/, '')
    .replace(/<body name="([^"]+)"/g, '<body gravcomp="1" name="$1"')
    .replace('<body name="left_finger"', '<body name="left_finger"');
  robot = robot.replace('<body gravcomp="1" name="left_finger"', '<site name="tcp" pos="0 0 0.1034" size="0.004" rgba="0 0 0 0"/><body gravcomp="1" name="left_finger"');
  robot = robot.replace('<option integrator="implicitfast"/>', '<option timestep="0.002" integrator="implicitfast" cone="elliptic" iterations="60"/>');
  robot = robot.replace('</worldbody>', `
    <geom name="floor" type="plane" size="4 4 .1" rgba=".035 .055 .075 1" friction="1 .005 .0001"/>
    <geom name="belt" type="box" pos=".48 .12 .045" size=".21 .7 .045" rgba=".07 .095 .12 1" friction="1 .005 .0001"/>
    <geom name="belt_left" type="box" pos=".25 .12 .072" size=".012 .72 .035" rgba=".22 .28 .32 1"/>
    <geom name="belt_right" type="box" pos=".71 .12 .072" size=".012 .72 .035" rgba=".22 .28 .32 1"/>
    <geom name="basket_floor" type="box" pos=".32 -.4 .055" size=".135 .115 .02" rgba=".07 .29 .34 1"/>
    <geom name="basket_wall1" type="box" pos=".18 -.4 .11" size=".008 .12 .075" rgba=".1 .42 .48 1"/>
    <geom name="basket_wall2" type="box" pos=".46 -.4 .11" size=".008 .12 .075" rgba=".1 .42 .48 1"/>
    <geom name="basket_wall3" type="box" pos=".32 -.52 .11" size=".14 .008 .075" rgba=".1 .42 .48 1"/>
    <geom name="basket_wall4" type="box" pos=".32 -.28 .11" size=".14 .008 .075" rgba=".1 .42 .48 1"/>
    <geom name="bottle" type="capsule" pos=".55 .45 .118" size=".025 .035" quat=".707 0 .707 0" rgba=".1 .62 .52 1"/>
    <geom name="carton" type="box" pos=".38 -.08 .12" size=".035 .025 .03" rgba=".65 .48 .29 1"/>
    <body name="can" pos=".48 .38 .15">
      <freejoint name="can_free"/>
      <geom name="can_body" type="cylinder" size=".027 .055" mass=".035" rgba=".88 .18 .10 1" friction="3 .01 .001" condim="4" solref=".004 1"/>
      <geom name="can_top" type="cylinder" pos="0 0 .055" size=".0275 .001" mass=".001" rgba=".74 .79 .82 1" contype="0" conaffinity="0"/>
      <geom name="can_bottom" type="cylinder" pos="0 0 -.055" size=".0275 .001" mass=".001" rgba=".74 .79 .82 1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>`);
  return robot;
}
