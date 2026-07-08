// The shared-teammate graph: two drivers are linked if they ever drove for the same team in the same
// season (same car). Chaining those links lets us compare drivers who never met — the connectivity
// that makes the whole skill scale identifiable. BFS gives the shortest such chain.

export interface TeamEdge {
  a: string;
  b: string;
  teams: string[];
}
export interface TeamNode {
  id: string;
  name: string;
}
export interface Teammates {
  nodes: TeamNode[];
  edges: TeamEdge[];
}

type Adj = Map<string, { to: string; teams: string[] }[]>;

export function buildAdjacency(edges: TeamEdge[]): Adj {
  const adj: Adj = new Map();
  const add = (from: string, to: string, teams: string[]) => {
    if (!adj.has(from)) adj.set(from, []);
    adj.get(from)!.push({ to, teams });
  };
  for (const e of edges) {
    add(e.a, e.b, e.teams);
    add(e.b, e.a, e.teams);
  }
  return adj;
}

export interface Path {
  nodes: string[];
  links: { a: string; b: string; teams: string[] }[];
}

/** Breadth-first shortest shared-teammate path from `a` to `b`; null if disconnected. */
export function shortestPath(adj: Adj, a: string, b: string): Path | null {
  if (a === b) return { nodes: [a], links: [] };
  const prev = new Map<string, { from: string; teams: string[] }>();
  const seen = new Set<string>([a]);
  let frontier = [a];
  while (frontier.length) {
    const next: string[] = [];
    for (const node of frontier) {
      for (const { to, teams } of adj.get(node) ?? []) {
        if (seen.has(to)) continue;
        seen.add(to);
        prev.set(to, { from: node, teams });
        if (to === b) {
          // reconstruct
          const nodes = [b];
          const links: Path["links"] = [];
          let cur = b;
          while (cur !== a) {
            const p = prev.get(cur)!;
            links.unshift({ a: p.from, b: cur, teams: p.teams });
            nodes.unshift(p.from);
            cur = p.from;
          }
          return { nodes, links };
        }
        next.push(to);
      }
    }
    frontier = next;
  }
  return null;
}
