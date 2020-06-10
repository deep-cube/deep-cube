export type BlockMove = string;

export function BareBlockMove(prefix: string, amt?: number) {
    amt = amt || 1
    return (amt === -1) ? prefix + "'" : prefix
}