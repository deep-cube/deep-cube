import React from 'react';
import { render } from '@testing-library/react';
import App from './App';

import { RotationManager, Rotation } from './lib/Rotation'
import { Quaternion, Vector3 } from 'three';

test('updates orientation correctly', () => {
  let rm = new RotationManager(new Quaternion())

  //console.log("closest face of U", Rotation.getClosestFace(new Vector3(0, 1, 0.3)));
  let rot = rm.updateQuatAndCalcRotation(new Quaternion().setFromAxisAngle(new Vector3( 1, 0, 0), Math.PI / 2))
  console.log("reported rotation on x = ", rot)
  console.assert(rot === "x'")
});