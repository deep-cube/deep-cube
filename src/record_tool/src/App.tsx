import './App.css';

import {
  BluetoothPuzzle,
  MoveEvent,
  OrientationEvent,
} from './cubing.js/bluetooth/bluetooth-puzzle';
import { Cube, Move } from './lib/CubeLib';
import { Matrix4, Quaternion, Vector3 } from 'three';
import React, { useEffect } from 'react';

import Button from 'react-bootstrap/Button';
import Col from 'react-bootstrap/Col';
import Container from 'react-bootstrap/Container';
import CubeSim from './components/CubeSim';
import { CubieT } from './lib/Defs';
import { GanCube } from './cubing.js/bluetooth/gan';
import Row from 'react-bootstrap/Row';
import { connect } from './cubing.js/bluetooth/connect';

export type CubeModel = {
  cube: CubieT;
  orientation: Quaternion;
};

export type Move = [string, number];
export type Moves = Move[];

const repr = (v: number[]) => {
  return '(' + v.map((x) => x.toFixed(3)).join(',') + ')';
};

function processData(moves: Moves) {
  return {
    moves,
  };
}

const DEBUG = false;

function App() {
  const moveRef = React.useRef<Moves>([]);
  const [cube, setCube] = React.useState<CubeModel>({
    cube: Cube.id,
    orientation: new Quaternion(),
  });
  const [puzzle, setPuzzle] = React.useState<BluetoothPuzzle | null>(null);
  const [pause, setPause] = React.useState(false);

  const state = React.useRef({ cube, puzzle, pause });

  useEffect(() => {
    state.current = { cube, puzzle, pause };
  });

  const onMove = (e: MoveEvent) => {
    //debugLog("Move", e)
    let { cube, puzzle, pause } = state.current!;

    if (!'xyz'.includes(e.latestMove[0])) {
      const newCube = {
        ...cube,
        cube: Cube.apply(cube.cube, Move.parse(e.latestMove)),
      };
      setCube(newCube);
      state.current.cube = newCube;
    }
    const moveRelative = puzzle
      ? puzzle.rotationManager.calcCurrentMove(e.latestMove)
      : e.latestMove;
    //console.log(e.latestMove, moveRelative)
    const newMove: Move = [moveRelative, e.timeStamp];
    if (!pause) moveRef.current.push(newMove);
  };

  const onOri = (e: OrientationEvent) => {
    const { cube } = state.current!;
    const { x, y, z, w } = e.quaternion;
    const q = new Quaternion(x, y, z, w);
    //debugLog("ori", e)

    const newCube = {
      ...cube,
      orientation: q,
    };
    setCube(newCube);
    state.current.cube = newCube;
  };
  if (DEBUG)
    (window as any).trigger = () =>
      onMove({ latestMove: 'U', timeStamp: new Date().getTime() });

  const onClear = () => {
    moveRef.current = [];
  };
  const onPause = () => {
    setPause(!pause);
  };
  const onSave = () => {
    const element = document.createElement('a');
    const file = new Blob(
      [JSON.stringify(processData(moveRef.current), null, 2)],
      { type: 'text/plain' }
    );
    element.href = URL.createObjectURL(file);
    element.download = 'recording.json';
    document.body.appendChild(element); // Required for this to work in FireFox
    element.click();
  };

  const connectBluetooth = async () => {
    const puzzle: GanCube = (await connect()) as GanCube;
    puzzle.addMoveListener(onMove);
    puzzle.addOrientationListener(onOri);
    setPuzzle(puzzle);
    alert('battery (weirdly fluctuating)= ' + (await puzzle.getBattery()));
  };

  const calibrate = (mode: string) => () => {
    let { puzzle } = state.current!;
    if (!puzzle) return;
    if (mode === 'base') puzzle.setBase();
    else if (mode === 'x') puzzle.setXAxis();
    else if (mode === 'y') puzzle.setYAxis();
  };

  const x = new Vector3(1, 0, 0).applyQuaternion(cube.orientation).toArray();
  const y = new Vector3(0, 1, 0).applyQuaternion(cube.orientation).toArray();
  const z = new Vector3(0, 0, 1).applyQuaternion(cube.orientation).toArray();
  const s = `Current cube: \n x axis = ${repr(x)}\n y axis = ${repr(
    y
  )}\n z axis = ${repr(z)}`;

  const moveView = moveRef.current.map((m: [string, number], i: number) => {
    return <div key={i}> {m[0] + ' , ' + m[1]} </div>;
  });
  moveView.reverse();
  const cubeToDisplay = Cube.to_facelet(cube.cube);

  return (
    <Container className='App'>
      <div>
        <Button onClick={connectBluetooth}>Connect to Bluetooth </Button>
        <span style={{ color: 'gray', fontSize: '0.8rem', margin: '10px' }}>
          Shake 5 times to wake the cube. If cube is drifting, wait or refresh
          until it stops.
        </span>
      </div>

      {/* <div style={{ height: '10px' }} /> */}
      <div>
        <span style={{ color: 'gray', fontSize: '0.8rem' }}>
          The cube's absolute rotation tends to drift at startup, but goes away
          after a while. So the recommended calibration steps are:
          <ol>
            <li>connect</li>
            <li>calibrate step 123</li>
            <li>free play for a while</li>
            <li>calibrate step 123 (Kevin: I usually do 1231)</li>
            <li>record your clips. every time a clip finishes, re-calibrate</li>
          </ol>
        </span>
      </div>
      <div>
        <Button variant='secondary' onClick={calibrate('base')}>
          1.Calibrate base (White top Green front facing you){' '}
        </Button>{' '}
        <Button variant='secondary' onClick={calibrate('x')}>
          2.Calibrate after x'{' '}
        </Button>{' '}
        <Button variant='secondary' onClick={calibrate('y')}>
          3.Undo x'. Then calibrate after y'.{' '}
        </Button>{' '}
      </div>

      <hr></hr>
      <Row>
        <Col>
          <div className='cubeContainer'>
            <CubeSim
              height={300}
              width={300}
              cube={cubeToDisplay}
              rotation={new Matrix4().makeRotationFromQuaternion(
                cube.orientation
              )}
            />
          </div>
          <div className='stats'>
            <p>
              <span style={{ whiteSpace: 'pre-line' }}>
                Quaternion:{' '}
                {cube.orientation
                  .toArray()
                  .map((x) => x.toFixed(3))
                  .toString()}
                <br />
                {s}
                <br />
              </span>
            </p>
            <p>
              Recent Moves:{' '}
              {moveRef.current
                .slice(moveRef.current.length - 5)
                .map(([x]) => x)
                .join(' ')}
            </p>
            <p>
              Orientation:{' '}
              {puzzle && puzzle.rotationManager.getCurrentRotation()}
            </p>
          </div>
        </Col>
        <Col>
          <div>
            <Button variant='outline-danger' onClick={onClear}>
              Clear{' '}
            </Button>{' '}
            <Button variant='outline-secondary' onClick={onPause}>
              {' '}
              {pause ? 'Resume Recording' : 'Pause Recording'}{' '}
            </Button>{' '}
            <Button variant='outline-primary' onClick={onSave}>
              Save to File{' '}
            </Button>{' '}
          </div>
          <div>{moveView}</div>
        </Col>
      </Row>
    </Container>
  );
}

// Debug
// const xbasis = puzzle && repr(puzzle.ax.toArray())
// const ybasis = puzzle && repr(puzzle.ay.toArray())
// const zbasis = puzzle && repr(puzzle.az.toArray())
// const basisStr = `In raw data, xbasis = ${xbasis}\nybasis = ${ybasis}\nzbasis = ${zbasis}`

export default App;
