interface DenseLayer {
  weights: number[][];
  bias: number[];
  activation: 'clip' | 'scores';
}

export interface Network {
  inputSize: number;
  layers: DenseLayer[];
}

const WIDTH = 480;
const HEIGHT = 300;
const CENTER = 128 / 255;
const FIELD = 'x="0" y="0" width="4" height="4"';
const fmt = (n: number) => Number(n.toFixed(9)).toString();
const segments = [
  'abcdef',
  'bc',
  'abged',
  'abgcd',
  'fgbc',
  'afgcd',
  'afgecd',
  'abc',
  'abcdefg',
  'abfgcd',
];
// All segments share 12-unit thickness and 6-unit bevels at the same joints.
const paths: Record<string, string> = {
  a: 'M12 0H42L48 6L42 12H12L6 6Z',
  b: 'M48 6L54 12V42L48 48L42 42V12Z',
  c: 'M48 48L54 54V84L48 90L42 84V54Z',
  d: 'M12 84H42L48 90L42 96H12L6 90Z',
  e: 'M6 48L12 54V84L6 90L0 84V54Z',
  f: 'M6 6L12 12V42L6 48L0 42V12Z',
  g: 'M12 42H42L48 48L42 54H12L6 48Z',
};

function digit(n: number, x: number, y: number, scale = 1, fill = '#e9ff77'): string {
  // A single compound path avoids antialiasing seams along shared edges.
  return `<path transform="translate(${x} ${y}) scale(${scale})" fill="${fill}" d="${[...segments[n]].map((segment) => paths[segment]).join(' ')}"/>`;
}

/** Shared by training and the compiler: reserve headroom for every partial sum. */
function layerScales(layer: DenseLayer): number[] {
  const norms = layer.weights.map(
    (row, i) => Math.abs(layer.bias[i]) + row.reduce((s, w) => s + Math.abs(w), 0),
  );
  if (layer.activation === 'scores') return norms.map(() => 0.4 / Math.max(1, ...norms));
  return norms.map((n) => 0.4 / Math.max(1, n));
}

/** Compile the graph once; the returned function supplies each frame's input pixels. */
export function compileSvg(model: Network) {
  const count = model.layers.at(-1)!.bias.length;
  const graph: string[] = [];
  let serial = 0;
  function op(tag: string, attrs: string, children = '', bounds = FIELD): string {
    const id = `v${serial++}`;
    graph.push(
      `<${tag} ${bounds} ${attrs} result="${id}"${children ? `>${children}</${tag}>` : '/>'}`,
    );
    return id;
  }
  function matrix(input: string, rows: number[][]): string {
    const value = op(
      'feColorMatrix',
      `in="${input}" type="matrix" values="${[...rows.flat(), 0, 0, 0, 0, 1].map(fmt).join(' ')}"`,
    );
    return materialize(value);
  }
  // Dilation preserves uniform fields and is intended to separate reused calculations.
  // The browser still controls caching; this does not make direct SVG display reliable.
  function materialize(input: string): string {
    return op('feMorphology', 'in="' + input + '" operator="dilate" radius="1"');
  }
  function scalar(input: string, channel: number): string {
    const row = [0, 0, 0, 0, 0];
    row[channel] = 1;
    return matrix(input, [row, row, row]);
  }
  function alpha(input: string, color: number[] = [233 / 255, 1, 119 / 255]): string {
    return op(
      'feColorMatrix',
      `in="${input}" values="0 0 0 0 ${color[0]} 0 0 0 0 ${color[1]} 0 0 0 0 ${color[2]} 1 0 0 0 0"`,
    );
  }
  const box = (x: number, y: number, w: number, h: number) =>
    `x="${x}" y="${y}" width="${w}" height="${h}"`;
  const tile = (input: string, x: number, y: number, w: number, h: number) =>
    op('feTile', `in="${input}"`, '', box(x, y, w, h));

  let previous = Array.from({ length: Math.ceil(model.inputSize / 3) }, (_, i) => `input-${i}`);
  for (const [layerIndex, layer] of model.layers.entries()) {
    const scales = layerScales(layer);
    const next: string[] = [];
    // Missing channels belong to the final, partially filled RGB group.
    for (let group = 0; group < layer.bias.length; group += 3) {
      let accumulator = '';
      for (let block = 0; block < previous.length; block++) {
        const rows = [0, 1, 2].map((channel) => {
          const neuron = group + channel;
          const scale = scales[neuron] ?? 1;
          const weights = layer.weights[neuron];
          return [0, 1, 2]
            .map((c) => (weights?.[block * 3 + c] ?? 0) * scale)
            .concat([0, CENTER + (block === 0 ? (layer.bias[neuron] ?? 0) * scale : 0)]);
        });
        const term = matrix(previous[block], rows);
        accumulator =
          block === 0
            ? term
            : materialize(
                op(
                  'feComposite',
                  `in="${accumulator}" in2="${term}" operator="arithmetic" k2="1" k3="1" k4="${fmt(-CENTER)}"`,
                ),
              );
      }
      if (layer.activation === 'clip') {
        accumulator = matrix(
          accumulator,
          [0, 1, 2].map((c) => {
            const scale = scales[group + c] ?? 1;
            const row = [0, 0, 0, 0, -CENTER / scale];
            row[c] = 1 / scale;
            return row;
          }),
        );
      }
      graph.push(
        `<!-- layer ${layerIndex}, neurons ${group}..${Math.min(group + 2, layer.bias.length - 1)}: ${accumulator} -->`,
      );
      next.push(accumulator);
    }
    previous = next;
  }

  const scores = Array.from({ length: count }, (_, i) =>
    scalar(previous[Math.floor(i / 3)], i % 3),
  );
  const stripBounds = box(0, 0, count * 4, 4);
  function pack(fields: string[]): string {
    const placed = fields.map((id, i) =>
      op('feOffset', 'in="' + id + '" dx="' + i * 4 + '" dy="0"', '', box(i * 4, 0, 4, 4)),
    );
    return op(
      'feMerge',
      '',
      placed.map((id) => '<feMergeNode in="' + id + '"/>').join(''),
      stripBounds,
    );
  }
  const lastBin = '0 '.repeat(255) + '1';
  function equalStrip(left: string, right: string): string {
    const difference = op(
      'feComposite',
      'in="' + left + '" in2="' + right + '" operator="arithmetic" k2="1" k3="-1" k4="1"',
      '',
      stripBounds,
    );
    return op(
      'feComponentTransfer',
      'in="' + difference + '"',
      ['R', 'G', 'B']
        .map((c) => '<feFunc' + c + ' type="discrete" tableValues="' + lastBin + '"/>')
        .join(''),
      stripBounds,
    );
  }
  function maximumStrip(input: string): string {
    return op(
      'feMorphology',
      'in="' + input + '" operator="dilate" radius="' + count * 4 + ' 0"',
      '',
      stripBounds,
    );
  }

  // Spatial SIMD: ten scores share one strip. Dilation finds the maximum once.
  // A fixed priority strip breaks ties in favor of the smallest digit.
  const scoreStrip = pack(scores);
  const maxima = equalStrip(scoreStrip, maximumStrip(scoreStrip));
  const priorities = pack(
    Array.from({ length: count }, (_, i) => {
      const value = 255 - i * 20;
      return op('feFlood', 'flood-color="rgb(' + [value, value, value].join(',') + ')"');
    }),
  );
  const ranked = op(
    'feComposite',
    'in="' + maxima + '" in2="' + priorities + '" operator="arithmetic" k1="1"',
    '',
    stripBounds,
  );
  let winners = equalStrip(ranked, maximumStrip(ranked));

  let ink = 'input-0';
  for (let i = 1; i < Math.ceil(model.inputSize / 3); i++)
    ink = op('feBlend', 'in="' + ink + '" in2="input-' + i + '" mode="lighten"');
  ink = matrix(
    ink,
    Array.from({ length: 3 }, () => [255, 255, 255, 0, 0]),
  );
  const inkStrip = tile(ink, 0, 0, count * 4, 4);
  winners = op(
    'feComposite',
    'in="' + winners + '" in2="' + inkStrip + '" operator="arithmetic" k1="1"',
    '',
    stripBounds,
  );
  const output: string[] = [];
  for (let i = 0; i < count; i++) {
    const winner = op('feOffset', 'in="' + winners + '" dx="' + -i * 4 + '" dy="0"');
    const mask = alpha(winner);
    const destination = box(62, 80, 54, 96);
    const glyph = op(
      'feOffset',
      'in="SourceGraphic" dx="' +
        (62 - (i % 5) * 64) +
        '" dy="' +
        (80 - Math.floor(i / 5) * 104) +
        '"',
      '',
      destination,
    );
    const largeMask = tile(mask, 62, 80, 54, 96);
    output.push(
      op(
        'feComposite',
        'in="' + glyph + '" in2="' + largeMask + '" operator="in"',
        '',
        destination,
      ),
    );
    output.push(tile(mask, 24 + i * 44, 244, 28, 4));
    output.push(tile(scores[i], 24 + i * 44, 226, 28, 8));
  }

  // The trainer and handwriting preprocessor both produce an 8×8 input.
  for (let i = 0; i < model.inputSize; i++) {
    const value = scalar(`input-${Math.floor(i / 3)}`, i % 3);
    output.push(tile(value, 238 + (i % 8) * 21, 48 + Math.floor(i / 8) * 21, 18, 18));
  }
  const blank = matrix(
    ink,
    Array.from({ length: 3 }, () => [-1, 0, 0, 0, 1]),
  );
  output.push(tile(alpha(blank, [0.35, 0.38, 0.4]), 62, 124, 54, 8));
  op(
    'feMerge',
    '',
    output.map((id) => `<feMergeNode in="${id}"/>`).join(''),
    box(0, 0, WIDTH, HEIGHT),
  );

  const header = `<svg xmlns="http://www.w3.org/2000/svg" width="${WIDTH}" height="${HEIGHT}" viewBox="0 0 ${WIDTH} ${HEIGHT}" color-interpolation="sRGB">
<title>Brain.svg - native SVG neural network</title>
<desc>Input colors encode the pixels. Native filter operations compute the scores and select the displayed digit. No script, external resource, or precomputed prediction.</desc>
<defs><filter id="brain" filterUnits="userSpaceOnUse" primitiveUnits="userSpaceOnUse" x="0" y="0" width="${WIDTH}" height="${HEIGHT}" color-interpolation-filters="sRGB">`;
  const body = `${graph.join('\n')}
</filter></defs>
<rect width="${WIDTH}" height="${HEIGHT}" rx="16" fill="#111713"/>
<g fill="#8c998f" font-family="monospace" font-size="10" letter-spacing="2"><text x="28" y="32">PREDICTION</text><text x="238" y="32">8 × 8 INPUT</text><text x="28" y="205" font-size="8" letter-spacing="1">RESPONSES</text></g>
<g filter="url(#brain)">${Array.from({ length: count }, (_, i) => digit(i, (i % 5) * 64, Math.floor(i / 5) * 104)).join('')}</g>
${Array.from({ length: count }, (_, i) => digit(i, 34 + i * 44, 261, 0.12, '#8c998f')).join('')}
</svg>`;

  return (pixels: readonly number[]) => {
    const inputs: string[] = [];
    for (let i = 0; i < model.inputSize; i += 3) {
      // A final partial RGB group is padded with zeroes.
      const rgb = [0, 1, 2].map((c) => Math.round((pixels[i + c] ?? 0) * 255));
      inputs.push(
        `<feFlood ${FIELD} flood-color="rgb(${rgb.join(',')})" result="input-${i / 3}"/>`,
      );
    }
    return `${header}\n${inputs.join('\n')}\n${body}`;
  };
}
