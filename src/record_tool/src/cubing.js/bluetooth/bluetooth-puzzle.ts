//import { BlockMove } from "../alg";
//import { Transformation } from "../kpuzzle";
// import { BasicRotationTransformer, StreamTransformer } from "./transformer";

import { Quaternion, Vector3 } from "three";
import { RotationManager } from "../../lib/Rotation";

/******** BluetoothPuzzle ********/

// TODO: Make compatible with Twisty.
export type OP = {
  orientation: number[],
  permutation: number[]
}
export type PuzzleState = {
  CORNER: OP,
  EDGE: OP,
  CENTER: OP
}

// TODO: Use actual `CustomEvent`s?
// https://developer.mozilla.org/en-US/docs/Web/API/CustomEvent
export interface MoveEvent {
  latestMove: string;
  timeStamp: number;
  debug?: object;
  //state?: PuzzleState;
  quaternion?: any; // TODO: Unused
}

// TODO: Only use the `quaternion` field in the `MoveEvent`?
export interface OrientationEvent {
  quaternion: { x: number, y: number, z: number, w: number };
  timeStamp: number;
  debug?: object;
}

export interface BluetoothConfig {
  filters: BluetoothRequestDeviceFilter[];
  optionalServices: BluetoothServiceUUID[];
}

// TODO: Expose device name (and/or globally unique identifier)?
export abstract class BluetoothPuzzle {
  // public transformers: StreamTransformer[] = [];
  protected listeners: Array<(e: MoveEvent) => void> = []; // TODO: type
  protected orientationListeners: Array<(e: OrientationEvent) => void> = []; // TODO: type

  public abstract name(): string | undefined;

  // TODO: require subclasses to implement this?
  //public async getState(): Promise<PuzzleState> {
    //throw new Error("cannot get state");
  //}

  public ax: Vector3= new Vector3(1, 0, 0);
  public ay: Vector3= new Vector3(0, 1, 0);
  public az: Vector3= new Vector3(0, 0, 1);

  private prevQuat: Quaternion = new Quaternion();
  private baseQuat: Quaternion = new Quaternion();

  public rotationManager: RotationManager = new RotationManager(new Quaternion());

  static axisFromQuaternion(q: Quaternion) {
    const axis = new Vector3(q.x, q.y, q.z).normalize()
    if (q.w < 0) {
      axis.negate()
    }
    return axis
  }
  private updateZ() {
    this.az = this.ax.clone().cross(this.ay)
    // this.rot = new Matrix4().getInverse(new Matrix4().fromArray([
    //   this.qx.x, this.qy.x, this.qz.x, 0,
    //   this.qx.y, this.qy.y, this.qz.y, 0,
    //   this.qx.z, this.qy.z, this.qz.z, 0,
    //   0, 0, 0, 1
    // ]) )
  }
  public setXAxis() {
    this.ax = BluetoothPuzzle.axisFromQuaternion(this.baseQuat.clone().multiply(this.prevQuat))
    this.updateZ()
  }

  public setYAxis() {
    this.ay = BluetoothPuzzle.axisFromQuaternion(this.baseQuat.clone().multiply(this.prevQuat))
    this.updateZ()
  }

  public setBase() {
    this.baseQuat = this.prevQuat.clone().inverse()
  }

  public addMoveListener(listener: (e: MoveEvent) => void): void {
    this.listeners.push(listener);
  }

  public addOrientationListener(listener: (e: OrientationEvent) => void): void {
    this.orientationListeners.push(listener);
  }

  // public experimentalAddBasicRotationTransformer(): void {
  //   this.transformers.push(new BasicRotationTransformer());
  // }

  protected dispatchMove(moveEvent: MoveEvent): void {
    // for (const transformer of this.transformers) {
    //   transformer.transformMove(moveEvent);
    // }
    // moveEvent.latestMove = this.rotationManager.calcCurrentMove(moveEvent.latestMove)
    for (const l of this.listeners) {
      l(moveEvent);
    }
  }

  protected dispatchOrientation(orientationEvent: OrientationEvent): void {
    // for (const transformer of this.transformers) {
    //   transformer.transformOrientation(orientationEvent);
    // }
    let {x, y, z, w} = orientationEvent.quaternion

    let q = new Quaternion(x, y, z, w)
    this.prevQuat = q.clone()

    q = this.baseQuat.clone().multiply(q)

    let axis = new Vector3(q.x, q.y, q.z)

    let cubeAxis = new Vector3(axis.dot(this.ax), axis.dot(this.ay), axis.dot(this.az))

    cubeAxis.normalize().multiplyScalar(axis.length())

    q = new Quaternion(cubeAxis.x, cubeAxis.y, cubeAxis.z, q.w)

    orientationEvent.quaternion = {x: q.x, y: q.y, z: q.z, w: q.w}

    for (const l of this.orientationListeners) {
      // TODO: Convert quaternion.
      l(orientationEvent);
    }

    const rot = this.rotationManager.updateQuatAndCalcRotation(q)
    if (rot !== "") {
      const moveEvent : MoveEvent = {
        latestMove: rot,
        timeStamp: orientationEvent.timeStamp
      }
      for (const l of this.listeners) {
        l(moveEvent)
      }
    }
  }

}
