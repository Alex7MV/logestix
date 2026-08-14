// Mosaic — decomposes a complex user prompt into discrete task tiles and maps
// each tile to a routed agent node. Decomposition is deterministic and offline:
// explicit `agent:`/`->` line markers produce one tile per marker; an unmarked
// prompt tessellates into a single tile bound to the default router node.

const DEFAULT_AGENT = 'router';

// Each tile carries: { agent, task, ref } where `ref` is a human-readable id
// the server may use for routing the generation burst.
export function decompose(prompt) {
  const lines = prompt
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);

  const tiles = [];

  for (const line of lines) {
    const m = /^(?:agent\s*:)?([a-z][a-z0-9-_]*)\s*(?:->|--|:|-)\s*(.+)$/i.exec(line);
    if (m) {
      tiles.push({
        agent: m[1].toLowerCase(),
        task: m[2].trim(),
        ref: `agent:${m[1].toLowerCase()}`,
      });
    } else {
      // A bare line folds into the previous tile's task when one exists,
      // otherwise it seeds a new router tile.
      if (tiles.length) {
        tiles[tiles.length - 1].task += ` ${line}`;
      } else {
        tiles.push({ agent: DEFAULT_AGENT, task: line, ref: `agent:${DEFAULT_AGENT}` });
      }
    }
  }

  if (!tiles.length) {
    tiles.push({ agent: DEFAULT_AGENT, task: prompt, ref: `agent:${DEFAULT_AGENT}` });
  }
  return tiles;
}

export const AGENT_REGISTRY = ['router', 'researcher', 'encoder', 'decoder', 'writer', 'verifier'];
