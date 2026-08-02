// Inline caveat for a season that is still being raced. The model's newest season may be only
// half-complete (2026 was 11 of 22 rounds at the time of writing), which means its car pace and
// driver skills rest on roughly half the usual evidence. Rather than shout this site-wide, each
// feature renders the note only when the CURRENT selection actually leans on that partial season.
//
// `manifest.partialSeason` is null once the season finishes (export_site.py reads the real round
// counts from f1db), so this component disappears on its own — nothing to remember to remove.

import type { Manifest } from "../../lib/data";

export function PartialSeasonNote({ manifest, when }: { manifest: Manifest; when: boolean }) {
  const p = manifest.partialSeason;
  if (!p || !when) return null;
  return (
    <p className="partial-note" role="note">
      <strong>{p.year} is still being raced.</strong> {p.roundsComplete} of {p.roundsTotal} rounds
      are in the model, so {p.year} figures rest on about half the usual evidence — expect them to
      move as the season plays out.
    </p>
  );
}
