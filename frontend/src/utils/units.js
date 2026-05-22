// Display-only z-score → physical-unit projection.
//
// The Bosch dataset is shipped pre-standardized and the original scaler
// (mean, std) per channel is not persisted with the project. The model
// keeps operating in z-space; these constants exist purely so the UI can
// render plausible engineering-typical values that end users can reason
// about (RPM in rev/min, λ as a ratio, °C, ppm, etc.) instead of raw σ.
//
// Numbers below are sensible defaults for a small naturally-aspirated
// gasoline engine at part load. Tweak `CALIBRATION` if you want a
// different presentation — nothing in the inference path depends on it.

export const CALIBRATION = {
  lambda:        { mean: 1.00, std: 0.05, unit: '',       prefix: 'λ ', decimals: 3, min: 0 },
  rpm:           { mean: 2500, std: 500,  unit: 'rpm',    decimals: 0,  min: 0 },
  speed:         { mean: 2500, std: 500,  unit: 'rpm',    decimals: 0,  min: 0 },
  load:          { mean: 50,   std: 20,   unit: '%',      decimals: 0,  min: 0, max: 100 },
  ignitionAngle: { mean: 15,   std: 5,    unit: '° BTDC', decimals: 1 },
  co:            { mean: 0.50, std: 0.40, unit: '% vol',  decimals: 2,  min: 0 },
  hc:            { mean: 200,  std: 150,  unit: 'ppm',    decimals: 0,  min: 0 },
  nox:           { mean: 800,  std: 400,  unit: 'ppm',    decimals: 0,  min: 0 },
  coBaseline:    { mean: 0.50, std: 0.40, unit: '% vol',  decimals: 2,  min: 0 },
  hcBaseline:    { mean: 200,  std: 150,  unit: 'ppm',    decimals: 0,  min: 0 },
  tempExhaust:   { mean: 650,  std: 80,   unit: '°C',     decimals: 0 },
  tempCatalyst:  { mean: 550,  std: 70,   unit: '°C',     decimals: 0 },
}

export function toPhysical(channel, z) {
  const c = CALIBRATION[channel]
  if (!c || z == null || !Number.isFinite(z)) return null
  let v = c.mean + z * c.std
  if (c.min !== undefined) v = Math.max(c.min, v)
  if (c.max !== undefined) v = Math.min(c.max, v)
  return v
}

export function unitOf(channel) {
  return CALIBRATION[channel]?.unit ?? ''
}

// "λ 0.925", "2925 rpm", "180 ppm"
export function formatPhysical(channel, z) {
  const c = CALIBRATION[channel]
  const v = toPhysical(channel, z)
  if (v === null) return '—'
  const num = v.toFixed(c.decimals)
  if (c.prefix) return `${c.prefix}${num}`
  if (c.unit)   return `${num} ${c.unit}`
  return num
}

// Compact dual readout — physical primary, σ secondary.
//   formatBoth('lambda', -1.23)  →  "λ 0.939 · −1.23σ"
export function formatBoth(channel, z) {
  if (z == null || !Number.isFinite(z)) return '—'
  const sigma = `${z >= 0 ? '+' : ''}${z.toFixed(2)}σ`
  return `${formatPhysical(channel, z)} · ${sigma}`
}
