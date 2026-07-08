// A P1..P20 finishing-position track with a shaded 90% credible band and a median marker.
// Uncertainty is always visible — never a bare point estimate (the project's design DNA).

import { FINISH_MAX, FINISH_MIN } from "../../lib/mesh";

interface Props {
  lo: number;
  med: number;
  hi: number;
  /** optional reference marker (e.g. the driver's typical finish), drawn as a hollow tick */
  reference?: { pos: number; label: string };
}

const W = 640;
const H = 96;
const PAD = 28;

function x(pos: number): number {
  const t = (pos - FINISH_MIN) / (FINISH_MAX - FINISH_MIN);
  return PAD + t * (W - 2 * PAD);
}

export function CredibleBand({ lo, med, hi, reference }: Props) {
  const ticks = [1, 5, 10, 15, 20];
  const trackY = H / 2;
  return (
    <svg className="band" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Expected finish P${med.toFixed(1)}, 90% range P${lo.toFixed(1)} to P${hi.toFixed(1)}`}>
      {/* baseline */}
      <line x1={PAD} y1={trackY} x2={W - PAD} y2={trackY} className="band__axis" />
      {ticks.map((t) => (
        <g key={t}>
          <line x1={x(t)} y1={trackY - 5} x2={x(t)} y2={trackY + 5} className="band__tick" />
          <text x={x(t)} y={trackY + 22} className="band__ticklabel">
            P{t}
          </text>
        </g>
      ))}
      {/* 90% band */}
      <rect x={x(lo)} y={trackY - 12} width={Math.max(2, x(hi) - x(lo))} height={24} rx={4} className="band__range" />
      {/* median */}
      <line x1={x(med)} y1={trackY - 18} x2={x(med)} y2={trackY + 18} className="band__median" />
      <text x={x(med)} y={trackY - 24} className="band__medlabel">
        P{med.toFixed(1)}
      </text>
      {reference && (
        <g>
          <line x1={x(reference.pos)} y1={trackY - 14} x2={x(reference.pos)} y2={trackY + 14} className="band__ref" />
          <text x={x(reference.pos)} y={trackY + 38} className="band__reflabel">
            {reference.label}
          </text>
        </g>
      )}
    </svg>
  );
}
