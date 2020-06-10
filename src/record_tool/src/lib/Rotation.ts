/*
 * The rotation group of the cube. (S4)
 * Translates moves relative to orientation.
 * Also monitors edges where orientation changes
 * and emit events
 */

import { Quaternion, Vector3 } from "three"

const decoder = "URFDLB"
const encoder : { [key: string] : number } =
    { "U": 0, "R": 1, "F": 2, "D": 3, "L": 4, "B": 5 }

export class Rotation {
    static id = [0, 1, 2, 3, 4, 5]
    static x = [5, 1, 0, 2, 4, 3]
    static y = [0, 2, 4, 3, 5, 1]
    static z = [1, 3, 2, 4, 0, 5];
    static faceVec : { [key: string] : Vector3 }= {
        "U": new Vector3(0, 1, 0),
        "D": new Vector3(0, -1, 0),
        "F": new Vector3(0, 0, 1),
        "B": new Vector3(0, 0, -1),
        "L": new Vector3(-1, 0, 0),
        "R": new Vector3(1, 0, 0),
    }

    static name2perm = (Rotation.generate_all_forward());
    static permstr2name = Rotation.generate_all_backward();

    static getRelativeMove(p: number[], move: string) {
        p = Rotation.invert(p)
        return decoder[p[encoder[move[0]]]] + move.slice(1)
    }

    static getClosestFace(v: Vector3): string {
        let max = -1, argmax = -1
        for (let i = 0; i < 6; i++) {
          const res = Rotation.faceVec[decoder[i]].dot(v)
          if (res > max) {
            max = res;
            argmax = i;
          }
        }
        return decoder[argmax]
    }

    static fromQuaternion(q: Quaternion) {
        const { getClosestFace, faceVec, permstr2name, name2perm, invert } = Rotation
        const invPermStr = getClosestFace(faceVec["U"].clone().applyQuaternion(q))
        + getClosestFace(faceVec["F"].clone().applyQuaternion(q))

        //console.log("invpermstr ", invPermStr)
        const invPerm = name2perm[permstr2name.get(invPermStr) || ""]
        const perm = invert(invPerm)
        return perm
    }

    static getRotationName(p1: number[], p2: number[]) {
        for (let i = 0; i < 6; i++) {
            if (p1[i] !== p2[i]) {
                const p = Rotation.apply( p1, Rotation.invert(p2) )
                return Rotation.permstr2name.get(Rotation.toString(p)) || ""
            }
        }
        return ""
    }

    static toString(p: number[]) {
        return decoder[p[0]] + decoder[p[2]]
    }

    static invert(p: number[]) {
        const arr : number[] = Array(6).fill(0)
        for (let i = 0;i < 6; i++) {
            arr[p[i]] = i;
        }
        return arr
    }

    static apply(p1: number[], p2: number[]) {
        const arr : number[] = Array(6).fill(0)
        for (let i = 0; i < 6; i++) {
            arr[i] = p2[p1[i]]
        }
        return arr
    }

    static compose(p1: number[], p2: number[]) {
        return Rotation.apply(p1, p2)
    }

    static repeat(a: number[], n: number) {
        let base = a;
        for (let i = 0; i < n - 1; i++) {
            base = Rotation.apply(base, a);
        }
        return base
    }

    static generate_all_forward() {
        let map : {[key: string]: number[]}
        let {id, x, y, z, repeat, compose: _} = Rotation
        let xp = repeat(x, 3), x2 = repeat(x, 2)
        let yp = repeat(y, 3), y2 = repeat(y, 2)
        let zp = repeat(z, 3), z2 = repeat(z, 2)
        map = {
            "": id, "y": y, "y'": yp, "y2": y2, // U
            "z2": z2, "x2": x2, "x2 y": _(x2, y), "x2 y'": _(x2, yp), // D
            "x": x, "x y": _(x, y), "x y'": _(x, yp), "x y2": _(x, y2), //F
            "x'": xp, "x' y": _(xp, y), "x' y2": _(xp, y2), "x' y'": _(xp, yp), // B
            "z": z, "z y": _(z, y), "z y2": _(z, y2), "z y'": _(z, yp), // L
            "z'": zp, "z' y": _(zp, y), "z' y2": _(zp, y2), "z' y'": _(zp, yp), // R
        }
        return map
    }

    static generate_all_backward() {
        let {toString} = Rotation
        let map = this.generate_all_forward()
        // invert
        let map_inv = new Map<string, string>()
        Object.entries(map).forEach( ([x, y]) => { map_inv.set( toString(y), x); } )
        // sanity check
        console.assert( map_inv.size === 24)
        console.log(map_inv)
        return map_inv
    }
}

export class RotationManager {
    private prev : number[] = Rotation.id;
    constructor(initialQuat: Quaternion) {
        this.prev = Rotation.fromQuaternion(initialQuat)
    }
    updateQuatAndCalcRotation(quat: Quaternion): string {
        const p = Rotation.fromQuaternion(quat);
        // TODO: figure out why the order has to be like this...
        const s = Rotation.getRotationName(this.prev, p);
        this.prev = [...p];
        return s;
    }
    getCurrentRotation(): string {
        const p = Rotation.invert(this.prev)
        return Rotation.permstr2name.get(Rotation.toString(p)) || ""
    }
    calcCurrentMove(move: string): string {
        if (move[0] === "x" || move[0] === "y" || move[0] === "z") {
            return move
        }
        return Rotation.getRelativeMove(this.prev, move)
    }
}