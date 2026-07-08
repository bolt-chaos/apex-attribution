// The E[finish] surface is a 2-D grid over (driver_skill, car_pace); `exp_finish` in the research
// pipeline sets both roots and marginalizes circuit_type, so this is a clean smooth function we can
// bilinearly interpolate. Credible bands come from replaying the posterior draws through it.

import type { Mesh } from "./data";

const FINISH_MIN = 1;
const FINISH_MAX = 20;

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

// Locate `v` in an ascending axis: return the lower cell index and the fractional offset t in [0,1].
function locate(axis: number[], v: number): { i: number; t: number } {
  const n = axis.length;
  if (v <= axis[0]) return { i: 0, t: 0 };
  if (v >= axis[n - 1]) return { i: n - 2, t: 1 };
  let i = 0;
  while (i < n - 2 && axis[i + 1] < v) i++;
  const t = (v - axis[i]) / (axis[i + 1] - axis[i]);
  return { i, t };
}

/** Bilinearly interpolate E[finish] at (skill, pace), clamped to a valid finishing position. */
export function interpFinish(mesh: Mesh, skill: number, pace: number): number {
  const { i, t: ts } = locate(mesh.skill_axis, skill);
  const { i: j, t: tp } = locate(mesh.pace_axis, pace);
  const z = mesh.z;
  const v =
    (1 - ts) * (1 - tp) * z[i][j] +
    ts * (1 - tp) * z[i + 1][j] +
    (1 - ts) * tp * z[i][j + 1] +
    ts * tp * z[i + 1][j + 1];
  return clamp(v, FINISH_MIN, FINISH_MAX);
}

export interface FinishBand {
  lo: number;
  med: number;
  hi: number;
  samples: number[];
}

/** Propagate posterior uncertainty in skill and pace through the mesh into an E[finish] band. */
export function finishBand(mesh: Mesh, skillDraws: number[], paceDraws: number[]): FinishBand {
  const n = Math.min(skillDraws.length, paceDraws.length);
  const samples: number[] = new Array(n);
  for (let k = 0; k < n; k++) samples[k] = interpFinish(mesh, skillDraws[k], paceDraws[k]);
  const sorted = [...samples].sort((a, b) => a - b);
  const q = (p: number) => sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
  return { lo: q(0.05), med: q(0.5), hi: q(0.95), samples };
}

/** Band over skill draws at a single fixed pace (e.g. one specific car) — for the cross-era query. */
export function finishBandAtPace(mesh: Mesh, skillDraws: number[], pace: number): FinishBand {
  return finishBand(mesh, skillDraws, new Array(skillDraws.length).fill(pace));
}

export { FINISH_MIN, FINISH_MAX };
