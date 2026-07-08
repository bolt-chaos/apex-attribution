// Closed-form helpers over downsampled posterior draws. Lower skill = faster (a better driver),
// so "A ahead of B" means A's skill draw is below B's.

export function quantile(sorted: number[], p: number): number {
  if (sorted.length === 0) return NaN;
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor(p * sorted.length)))];
}

/** P(A is faster than B) by pairing posterior draws index-for-index. */
export function pAhead(drawsA: number[], drawsB: number[]): number {
  const n = Math.min(drawsA.length, drawsB.length);
  if (n === 0) return NaN;
  let wins = 0;
  for (let k = 0; k < n; k++) if (drawsA[k] < drawsB[k]) wins++;
  return wins / n;
}

export function fmtPos(p: number): string {
  return `P${p.toFixed(1)}`;
}
