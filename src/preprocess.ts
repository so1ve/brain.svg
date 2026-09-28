/** Center, deskew and area-sample ink; the same transform prepares training images. */
export function normalizeInk(canvas: HTMLCanvasElement): number[] {
  const context = canvas.getContext('2d', { willReadFrequently: true })!;
  const { width, height } = canvas;
  const data = context.getImageData(0, 0, width, height).data;
  let mass = 0,
    momentX = 0,
    momentY = 0;
  let hasInk = false;
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const a = data[(y * width + x) * 4 + 3];
      mass += a;
      momentX += (x + 0.5) * a;
      momentY += (y + 0.5) * a;
      if (a > 24) hasInk = true;
    }
  if (!hasInk) return new Array(64).fill(0);
  const cx = momentX / mass,
    cy = momentY / mass;
  let xy = 0,
    yy = 0;
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const a = data[(y * width + x) * 4 + 3];
      xy += (x + 0.5 - cx) * (y + 0.5 - cy) * a;
      yy += (y + 0.5 - cy) ** 2 * a;
    }
  const shear = yy > 0 ? Math.max(-2, Math.min(2, xy / yy)) : 0;
  let left = Infinity,
    top = Infinity,
    right = -Infinity,
    bottom = -Infinity;
  const halfWidth = 0.5 * (1 + Math.abs(shear));
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      if (data[(y * width + x) * 4 + 3] <= 24) continue;
      const u = x + 0.5 - shear * (y + 0.5 - cy);
      left = Math.min(left, u - halfWidth);
      right = Math.max(right, u + halfWidth);
      top = Math.min(top, y);
      bottom = Math.max(bottom, y + 1);
    }
  const scale = 28 / Math.max(right - left, bottom - top);
  const marginX = (32 - (right - left) * scale) / 2;
  const marginY = (32 - (bottom - top) * scale) / 2;
  let originX = marginX - left * scale,
    originY = marginY - top * scale;
  originX += Math.max(-marginX, Math.min(marginX, 16 - (originX + cx * scale)));
  originY += Math.max(-marginY, Math.min(marginY, 16 - (originY + cy * scale)));
  const alpha = (x: number, y: number) =>
    x >= 0 && x < width && y >= 0 && y < height ? data[(y * width + x) * 4 + 3] : 0;
  const pixels = new Array<number>(64).fill(0);
  // 8 × 8 samples per output pixel preserve thin strokes without GPU resize differences.
  for (let y = 0; y < 64; y++)
    for (let x = 0; x < 64; x++) {
      const sourceY = ((y + 0.5) / 2 - originY) / scale;
      const sourceX = ((x + 0.5) / 2 - originX) / scale + shear * (sourceY - cy);
      const ix = Math.floor(sourceX - 0.5),
        iy = Math.floor(sourceY - 0.5);
      const fx = sourceX - 0.5 - ix,
        fy = sourceY - 0.5 - iy;
      const value =
        (alpha(ix, iy) * (1 - fx) + alpha(ix + 1, iy) * fx) * (1 - fy) +
        (alpha(ix, iy + 1) * (1 - fx) + alpha(ix + 1, iy + 1) * fx) * fy;
      pixels[Math.floor(y / 8) * 8 + Math.floor(x / 8)] += value;
    }
  return pixels.map((value) => Math.round(value / 64) / 255);
}
