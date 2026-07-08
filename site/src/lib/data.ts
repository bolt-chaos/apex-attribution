// Typed loaders for the committed JSON artifacts (see ../../DATA.md for the schema of each).
// Everything the site needs at runtime lives in these files — there is no backend.

export interface Names {
  drivers: Record<string, string>;
  constructors: Record<string, string>;
}

export interface Band {
  lo: number;
  med: number;
  hi: number;
}

export interface DriverEntry {
  name: string;
  seasons: number[];
  bySeason: Record<string, Band>;
  career: Band & { draws: number[] };
}
export type Drivers = Record<string, DriverEntry>;

export interface CarEntry extends Band {
  constructor: string;
  year: number;
  name: string;
  draws: number[];
}
export type Cars = Record<string, CarEntry>;

export interface Mesh {
  skill_axis: number[];
  pace_axis: number[];
  z: number[][];
  note?: string;
}

export interface EraRow {
  label: string;
  start: number;
  end: number;
  carPct: number;
  driverPct: number;
  carSpread: number;
  driverSpread: number;
}

export interface Manifest {
  generated: string;
  nDraws: number;
  meshN: number;
  expN: number;
  mainModel: string;
  crossEraModel: string;
  eras: string[];
  meshRanges: { skill: [number, number]; pace: [number, number] };
}

// Fetch a JSON artifact from the deployment's base path (works in dev "/" and on Pages).
export async function loadJSON<T>(name: string): Promise<T> {
  const res = await fetch(`${import.meta.env.BASE_URL}data/${name}.json`);
  if (!res.ok) throw new Error(`failed to load ${name}.json (${res.status})`);
  return (await res.json()) as T;
}

// The bundle of artifacts the current (PR2) feature set needs.
export interface CoreData {
  drivers: Drivers;
  cars: Cars;
  mesh: Mesh;
  manifest: Manifest;
}

export async function loadCore(): Promise<CoreData> {
  const [drivers, cars, mesh, manifest] = await Promise.all([
    loadJSON<Drivers>("drivers"),
    loadJSON<Cars>("cars"),
    loadJSON<Mesh>("finish_mesh"),
    loadJSON<Manifest>("manifest"),
  ]);
  return { drivers, cars, mesh, manifest };
}
