import { useMemo } from 'react'
import * as THREE from 'three'
import Piston from './Piston'

// ── Shared materials ──────────────────────────────────────────────────────
// `fuelHighlight` / `ignitionHighlight` (0..~1.6) drive the red emissive
// glow on the parts associated with each fault family. Zero means no glow.
const HIGHLIGHT_COLOR = '#ff2211'

function useMaterials(headEmissiveColor, headEmissiveStrength,
                      fuelHighlight, ignitionHighlight) {
  const block = useMemo(() => new THREE.MeshStandardMaterial({
    color: '#8a8e94', metalness: 0.82, roughness: 0.42,
  }), [])
  const head = useMemo(() => new THREE.MeshStandardMaterial({
    color: '#33363c', metalness: 0.78, roughness: 0.45,
  }), [])
  head.emissive = new THREE.Color(headEmissiveColor)
  head.emissiveIntensity = headEmissiveStrength

  const valveCover = useMemo(() => new THREE.MeshStandardMaterial({
    color: '#1c1f24', metalness: 0.55, roughness: 0.55,
  }), [])
  const crank = useMemo(() => new THREE.MeshStandardMaterial({
    color: '#1c1e22', metalness: 0.95, roughness: 0.22,
  }), [])
  const chrome = useMemo(() => new THREE.MeshStandardMaterial({
    color: '#cfd3d8', metalness: 0.95, roughness: 0.18,
  }), [])
  // Intake manifold / plenum / air-filter — glows red on fuel faults
  const manifold = useMemo(() => new THREE.MeshStandardMaterial({
    color: '#444a52', metalness: 0.70, roughness: 0.55,
  }), [])
  manifold.emissive = new THREE.Color(HIGHLIGHT_COLOR)
  manifold.emissiveIntensity = fuelHighlight * 0.85

  const exhaust = useMemo(() => new THREE.MeshStandardMaterial({
    color: '#2a1813', metalness: 0.55, roughness: 0.82,
  }), [])
  const bolt = useMemo(() => new THREE.MeshStandardMaterial({
    color: '#3a3e44', metalness: 0.85, roughness: 0.45,
  }), [])
  const pulley = useMemo(() => new THREE.MeshStandardMaterial({
    color: '#16181c', metalness: 0.85, roughness: 0.35,
  }), [])
  const belt = useMemo(() => new THREE.MeshStandardMaterial({
    color: '#0a0c0f', metalness: 0.10, roughness: 0.95,
  }), [])
  const oilPan = useMemo(() => new THREE.MeshStandardMaterial({
    color: '#4a4e54', metalness: 0.55, roughness: 0.60,
  }), [])

  // Dedicated emissive-capable materials for parameter-specific highlights.
  // Separate from `chrome`/`bolt`/`crank` so unrelated parts don't light up.
  const fuelRail = useMemo(() => new THREE.MeshStandardMaterial({
    color: '#cfd3d8', metalness: 0.95, roughness: 0.18,
  }), [])
  fuelRail.emissive = new THREE.Color(HIGHLIGHT_COLOR)
  fuelRail.emissiveIntensity = fuelHighlight * 1.6

  const injector = useMemo(() => new THREE.MeshStandardMaterial({
    color: '#3a3e44', metalness: 0.85, roughness: 0.45,
  }), [])
  injector.emissive = new THREE.Color(HIGHLIGHT_COLOR)
  injector.emissiveIntensity = fuelHighlight * 1.3

  const sparkPlugBody = useMemo(() => new THREE.MeshStandardMaterial({
    color: '#1c1e22', metalness: 0.95, roughness: 0.22,
  }), [])
  sparkPlugBody.emissive = new THREE.Color(HIGHLIGHT_COLOR)
  sparkPlugBody.emissiveIntensity = ignitionHighlight * 1.2

  const sparkPlugTop = useMemo(() => new THREE.MeshStandardMaterial({
    color: '#cfd3d8', metalness: 0.95, roughness: 0.18,
  }), [])
  sparkPlugTop.emissive = new THREE.Color(HIGHLIGHT_COLOR)
  sparkPlugTop.emissiveIntensity = ignitionHighlight * 1.7

  const coilPack = useMemo(() => new THREE.MeshStandardMaterial({
    color: '#1c1f24', metalness: 0.55, roughness: 0.55,
  }), [])
  coilPack.emissive = new THREE.Color(HIGHLIGHT_COLOR)
  coilPack.emissiveIntensity = ignitionHighlight * 1.1

  return { block, head, valveCover, crank, chrome, manifold,
           exhaust, bolt, pulley, belt, oilPan,
           fuelRail, injector, sparkPlugBody, sparkPlugTop, coilPack }
}

function boltRow(count, x0, x1, y, z, material, size = 0.045) {
  const out = []
  const step = (x1 - x0) / Math.max(count - 1, 1)
  for (let i = 0; i < count; i++) {
    out.push(
      <mesh key={`b${i}`} material={material} position={[x0 + i * step, y, z]}>
        <cylinderGeometry args={[size, size * 0.85, size * 1.35, 8]} />
      </mesh>
    )
  }
  return out
}

// ── Engine 1 / gengine1 : Inline-6 (refined) ──────────────────────────────
function Inline6({ rpmHz, faultClass, severity, mats }) {
  const CYL_COUNT = 6
  const CYL_SPACING = 0.78
  const CYL_RADIUS = 0.30
  const CYL_HEIGHT = 1.10
  const BLOCK_X = CYL_COUNT * CYL_SPACING + 0.4
  const BLOCK_Z = 1.10
  const BLOCK_Y = 1.20

  const bores = useMemo(() => {
    const arr = []
    const x0 = -(CYL_COUNT - 1) * CYL_SPACING / 2
    for (let i = 0; i < CYL_COUNT; i++) {
      arr.push({ x: x0 + i * CYL_SPACING, phase: (i % 2 ? Math.PI : 0) + i * Math.PI / 6 })
    }
    return arr
  }, [])

  return (
    <group>
      {/* Block */}
      <mesh material={mats.block} position={[0, 0.2, 0]} castShadow receiveShadow>
        <boxGeometry args={[BLOCK_X, BLOCK_Y, BLOCK_Z]} />
      </mesh>
      {/* Casting ribs on block sides */}
      {[...Array(5)].map((_, i) => (
        <mesh key={`rib${i}`} material={mats.block}
              position={[-BLOCK_X / 2 + 0.4 + i * ((BLOCK_X - 0.8) / 4), 0.2, BLOCK_Z / 2 + 0.015]}>
          <boxGeometry args={[0.05, BLOCK_Y - 0.20, 0.03]} />
        </mesh>
      ))}

      {/* Oil pan + flange bolts */}
      <mesh material={mats.oilPan} position={[0, -0.50, 0]} castShadow>
        <boxGeometry args={[BLOCK_X - 0.2, 0.42, BLOCK_Z - 0.25]} />
      </mesh>
      {boltRow(CYL_COUNT + 1, -BLOCK_X / 2 + 0.2, BLOCK_X / 2 - 0.2, -0.30, BLOCK_Z / 2 - 0.18, mats.bolt, 0.032)}
      {boltRow(CYL_COUNT + 1, -BLOCK_X / 2 + 0.2, BLOCK_X / 2 - 0.2, -0.30, -(BLOCK_Z / 2 - 0.18), mats.bolt, 0.032)}

      {/* Head + valve cover + head bolts */}
      <mesh material={mats.head} position={[0, 0.95, 0]} castShadow>
        <boxGeometry args={[BLOCK_X - 0.05, 0.32, BLOCK_Z - 0.05]} />
      </mesh>
      <mesh material={mats.valveCover} position={[0, 1.22, 0]} castShadow>
        <boxGeometry args={[BLOCK_X - 0.18, 0.20, BLOCK_Z - 0.18]} />
      </mesh>
      {[...Array(5)].map((_, i) => (
        <mesh key={`vcr${i}`} material={mats.chrome}
              position={[0, 1.33, -0.30 + i * 0.15]}>
          <boxGeometry args={[BLOCK_X - 0.35, 0.02, 0.05]} />
        </mesh>
      ))}
      {boltRow(CYL_COUNT * 2, -BLOCK_X / 2 + 0.22, BLOCK_X / 2 - 0.22, 1.13, BLOCK_Z / 2 - 0.06, mats.bolt)}
      {boltRow(CYL_COUNT * 2, -BLOCK_X / 2 + 0.22, BLOCK_X / 2 - 0.22, 1.13, -(BLOCK_Z / 2 - 0.06), mats.bolt)}

      {/* Cylinder bores */}
      {bores.map((b, i) => (
        <group key={`bore-${i}`} position={[b.x, 0.55, 0]}>
          <mesh material={mats.block}>
            <cylinderGeometry args={[CYL_RADIUS, CYL_RADIUS, CYL_HEIGHT, 24, 1, true]} />
          </mesh>
        </group>
      ))}
      {/* Spark plugs poking out the top of the valve cover */}
      {bores.map((b, i) => (
        <group key={`plug-${i}`} position={[b.x, 1.42, 0]}>
          <mesh material={mats.sparkPlugBody}>
            <cylinderGeometry args={[0.05, 0.07, 0.14, 12]} />
          </mesh>
          <mesh material={mats.sparkPlugTop} position={[0, 0.10, 0]}>
            <cylinderGeometry args={[0.035, 0.035, 0.06, 8]} />
          </mesh>
        </group>
      ))}

      {/* Pistons */}
      {bores.map((b, i) => (
        <Piston
          key={`piston-${i}`}
          x={b.x} phase={b.phase} rpmHz={rpmHz} radius={CYL_RADIUS - 0.04}
          faultClass={faultClass} severity={severity} index={i}
        />
      ))}

      {/* Crankshaft */}
      <mesh material={mats.crank}
            position={[0, -0.45, 0]}
            rotation={[0, 0, Math.PI / 2]}
            castShadow>
        <cylinderGeometry args={[0.16, 0.16, BLOCK_X + 0.6, 22]} />
      </mesh>
      {/* Harmonic balancer / front pulley + belt */}
      <group position={[BLOCK_X / 2 + 0.32, -0.45, 0]}>
        <mesh material={mats.pulley} rotation={[0, 0, Math.PI / 2]}>
          <cylinderGeometry args={[0.32, 0.32, 0.18, 28]} />
        </mesh>
        {[-0.04, 0.04].map((zOff, j) => (
          <mesh key={j} material={mats.belt}
                position={[0, 0, zOff]}
                rotation={[0, 0, Math.PI / 2]}>
            <torusGeometry args={[0.28, 0.014, 8, 28]} />
          </mesh>
        ))}
      </group>

      {/* Intake plenum + runners + throttle body */}
      <mesh material={mats.manifold} position={[0, 0.75, BLOCK_Z / 2 + 0.22]}>
        <boxGeometry args={[BLOCK_X - 0.4, 0.22, 0.42]} />
      </mesh>
      {bores.map((b, i) => (
        <mesh key={`intake-${i}`} material={mats.manifold}
              position={[b.x, 0.85, BLOCK_Z / 2 + 0.05]}
              rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.08, 0.08, 0.30, 14]} />
        </mesh>
      ))}
      <mesh material={mats.crank}
            position={[BLOCK_X / 2 - 0.25, 0.75, BLOCK_Z / 2 + 0.55]}
            rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.18, 0.18, 0.25, 18]} />
      </mesh>
      <mesh material={mats.chrome}
            position={[BLOCK_X / 2 - 0.25, 0.75, BLOCK_Z / 2 + 0.70]}
            rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.16, 0.16, 0.04, 18]} />
      </mesh>

      {/* Exhaust manifold + downpipe + tailpipe */}
      <mesh material={mats.exhaust} position={[0, 0.55, -(BLOCK_Z / 2 + 0.22)]}>
        <boxGeometry args={[BLOCK_X - 0.2, 0.34, 0.32]} />
      </mesh>
      {bores.map((b, i) => (
        <mesh key={`ex-${i}`} material={mats.exhaust}
              position={[b.x, 0.65, -(BLOCK_Z / 2 + 0.06)]}
              rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.10, 0.10, 0.34, 14]} />
        </mesh>
      ))}
      <mesh material={mats.exhaust}
            position={[BLOCK_X / 2 - 0.2, 0.55, -(BLOCK_Z / 2 + 1.15)]}
            rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.12, 0.12, 1.6, 18]} />
      </mesh>
      <mesh material={mats.chrome}
            position={[BLOCK_X / 2 - 0.2, 0.55, -(BLOCK_Z / 2 + 1.95)]}
            rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.14, 0.12, 0.08, 18]} />
      </mesh>

      {/* Fuel rail + injectors */}
      <mesh material={mats.fuelRail}
            position={[0, 1.02, BLOCK_Z / 2 + 0.20]}
            rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.05, 0.05, BLOCK_X - 0.5, 14]} />
      </mesh>
      {bores.map((b, i) => (
        <mesh key={`inj-${i}`} material={mats.injector}
              position={[b.x, 0.92, BLOCK_Z / 2 + 0.20]}>
          <cylinderGeometry args={[0.04, 0.04, 0.18, 10]} />
        </mesh>
      ))}

      {/* Coil pack on top of the valve cover */}
      <mesh material={mats.coilPack}
            position={[0, 1.45, BLOCK_Z / 2 - 0.40]}>
        <boxGeometry args={[BLOCK_X - 0.6, 0.10, 0.18]} />
      </mesh>
    </group>
  )
}

// ── Engine 2 / gengine2 : V8 ──────────────────────────────────────────────
function V8({ rpmHz, faultClass, severity, mats }) {
  const CYL_PER_BANK = 4
  const CYL_SPACING = 0.78
  const CYL_RADIUS = 0.28
  const CYL_HEIGHT = 0.95
  const BLOCK_X = CYL_PER_BANK * CYL_SPACING + 0.4
  const BLOCK_Z = 1.45
  const BLOCK_Y = 1.10
  const BANK_TILT = Math.PI / 6

  const bores = useMemo(() => {
    const arr = []
    const x0 = -(CYL_PER_BANK - 1) * CYL_SPACING / 2
    for (let i = 0; i < CYL_PER_BANK; i++) {
      arr.push({
        x: x0 + i * CYL_SPACING,
        phaseL: i * Math.PI / 4,
        phaseR: i * Math.PI / 4 + Math.PI / 2,
      })
    }
    return arr
  }, [])

  const Bank = ({ sign }) => (
    <group rotation={[sign * BANK_TILT, 0, 0]}>
      {/* Head */}
      <mesh material={mats.head} position={[0, 1.20, 0]} castShadow>
        <boxGeometry args={[BLOCK_X - 0.05, 0.28, 0.78]} />
      </mesh>
      {/* Valve cover */}
      <mesh material={mats.valveCover} position={[0, 1.45, 0]} castShadow>
        <boxGeometry args={[BLOCK_X - 0.20, 0.18, 0.62]} />
      </mesh>
      {[...Array(4)].map((_, i) => (
        <mesh key={`vcr${i}`} material={mats.chrome}
              position={[0, 1.55, -0.22 + i * 0.14]}>
          <boxGeometry args={[BLOCK_X - 0.35, 0.018, 0.04]} />
        </mesh>
      ))}
      {boltRow(CYL_PER_BANK * 2, -BLOCK_X / 2 + 0.18, BLOCK_X / 2 - 0.18, 1.34, 0.30, mats.bolt, 0.04)}
      {boltRow(CYL_PER_BANK * 2, -BLOCK_X / 2 + 0.18, BLOCK_X / 2 - 0.18, 1.34, -0.30, mats.bolt, 0.04)}

      {/* Bores */}
      {bores.map((b, i) => (
        <group key={`bore-${i}`} position={[b.x, 0.55, 0]}>
          <mesh material={mats.block}>
            <cylinderGeometry args={[CYL_RADIUS, CYL_RADIUS, CYL_HEIGHT, 24, 1, true]} />
          </mesh>
        </group>
      ))}
      {bores.map((b, i) => (
        <group key={`plug-${i}`} position={[b.x, 1.62, 0]}>
          <mesh material={mats.sparkPlugBody}>
            <cylinderGeometry args={[0.045, 0.06, 0.12, 12]} />
          </mesh>
          <mesh material={mats.sparkPlugTop} position={[0, 0.09, 0]}>
            <cylinderGeometry args={[0.030, 0.030, 0.05, 8]} />
          </mesh>
        </group>
      ))}
      {bores.map((b, i) => (
        <Piston
          key={`piston-${i}`}
          x={b.x} phase={sign > 0 ? b.phaseL : b.phaseR}
          rpmHz={rpmHz} radius={CYL_RADIUS - 0.04}
          faultClass={faultClass} severity={severity}
          index={i + (sign > 0 ? 0 : CYL_PER_BANK)}
        />
      ))}

      {/* Exhaust manifold on outer face */}
      <mesh material={mats.exhaust} position={[0, 0.55, -0.55]}>
        <boxGeometry args={[BLOCK_X - 0.2, 0.30, 0.22]} />
      </mesh>
      {bores.map((b, i) => (
        <mesh key={`ex-${i}`} material={mats.exhaust}
              position={[b.x, 0.70, -0.46]}
              rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.09, 0.09, 0.30, 14]} />
        </mesh>
      ))}
    </group>
  )

  return (
    <group>
      {/* Compact lower block */}
      <mesh material={mats.block} position={[0, 0.05, 0]} castShadow receiveShadow>
        <boxGeometry args={[BLOCK_X, BLOCK_Y, BLOCK_Z - 0.55]} />
      </mesh>
      {[...Array(4)].map((_, i) => (
        <mesh key={`rib${i}`} material={mats.block}
              position={[-BLOCK_X / 2 + 0.3 + i * ((BLOCK_X - 0.6) / 3), 0.05, (BLOCK_Z - 0.55) / 2 + 0.015]}>
          <boxGeometry args={[0.05, BLOCK_Y - 0.20, 0.03]} />
        </mesh>
      ))}
      {/* Wide oil pan */}
      <mesh material={mats.oilPan} position={[0, -0.65, 0]} castShadow>
        <boxGeometry args={[BLOCK_X - 0.2, 0.50, 0.85]} />
      </mesh>
      {boltRow(CYL_PER_BANK + 1, -BLOCK_X / 2 + 0.18, BLOCK_X / 2 - 0.18, -0.42, 0.40, mats.bolt, 0.035)}
      {boltRow(CYL_PER_BANK + 1, -BLOCK_X / 2 + 0.18, BLOCK_X / 2 - 0.18, -0.42, -0.40, mats.bolt, 0.035)}

      {/* Crankshaft */}
      <mesh material={mats.crank}
            position={[0, -0.20, 0]}
            rotation={[0, 0, Math.PI / 2]}
            castShadow>
        <cylinderGeometry args={[0.18, 0.18, BLOCK_X + 0.5, 22]} />
      </mesh>
      {/* Pulley + belt */}
      <group position={[BLOCK_X / 2 + 0.30, -0.20, 0]}>
        <mesh material={mats.pulley} rotation={[0, 0, Math.PI / 2]}>
          <cylinderGeometry args={[0.34, 0.34, 0.18, 28]} />
        </mesh>
        <mesh material={mats.belt} rotation={[0, 0, Math.PI / 2]}>
          <torusGeometry args={[0.30, 0.018, 8, 28]} />
        </mesh>
      </group>

      <Bank sign={+1} />
      <Bank sign={-1} />

      {/* Intake plenum in the valley between the banks */}
      <mesh material={mats.manifold} position={[0, 1.30, 0]} castShadow>
        <boxGeometry args={[BLOCK_X - 0.4, 0.24, 0.55]} />
      </mesh>
      {/* Big central throttle body */}
      <mesh material={mats.crank}
            position={[BLOCK_X / 2 + 0.05, 1.30, 0]}
            rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.22, 0.22, 0.30, 18]} />
      </mesh>
      <mesh material={mats.chrome}
            position={[BLOCK_X / 2 + 0.22, 1.30, 0]}
            rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.19, 0.19, 0.04, 18]} />
      </mesh>

      {/* Tailpipe out the back, single exit */}
      <mesh material={mats.exhaust}
            position={[BLOCK_X / 2 - 0.2, -0.30, -(BLOCK_Z / 2 + 0.20)]}
            rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.14, 0.14, 1.5, 18]} />
      </mesh>
      <mesh material={mats.chrome}
            position={[BLOCK_X / 2 - 0.2, -0.30, -(BLOCK_Z / 2 + 1.00)]}
            rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.17, 0.14, 0.10, 18]} />
      </mesh>
    </group>
  )
}

// ── Engine 3 / pengines : Flat-4 (boxer) ──────────────────────────────────
function Flat4({ rpmHz, faultClass, severity, mats }) {
  const CYL_PER_BANK = 2
  const CYL_SPACING = 0.95
  const CYL_RADIUS = 0.32
  const CYL_HEIGHT = 1.05
  const BLOCK_X = CYL_PER_BANK * CYL_SPACING + 0.5
  const BLOCK_Z = 1.00
  const BLOCK_Y = 0.95

  const bores = useMemo(() => {
    const arr = []
    const x0 = -(CYL_PER_BANK - 1) * CYL_SPACING / 2
    for (let i = 0; i < CYL_PER_BANK; i++) {
      arr.push({
        x: x0 + i * CYL_SPACING,
        phaseL: i * Math.PI,
        phaseR: i * Math.PI + Math.PI,
      })
    }
    return arr
  }, [])

  // Bank rotated 90° around X. sign=+1 → bank on +Z (faces front), -1 → -Z.
  const Bank = ({ sign }) => (
    <group rotation={[sign * Math.PI / 2, 0, 0]}>
      <mesh material={mats.head} position={[0, 1.15, 0]} castShadow>
        <boxGeometry args={[BLOCK_X - 0.10, 0.30, 1.00]} />
      </mesh>
      <mesh material={mats.valveCover} position={[0, 1.42, 0]} castShadow>
        <boxGeometry args={[BLOCK_X - 0.25, 0.20, 0.80]} />
      </mesh>
      {[...Array(3)].map((_, i) => (
        <mesh key={`vcr${i}`} material={mats.chrome}
              position={[0, 1.53, -0.22 + i * 0.22]}>
          <boxGeometry args={[BLOCK_X - 0.45, 0.02, 0.05]} />
        </mesh>
      ))}
      {boltRow(CYL_PER_BANK * 3, -BLOCK_X / 2 + 0.20, BLOCK_X / 2 - 0.20, 1.30, 0.42, mats.bolt)}
      {boltRow(CYL_PER_BANK * 3, -BLOCK_X / 2 + 0.20, BLOCK_X / 2 - 0.20, 1.30, -0.42, mats.bolt)}

      {bores.map((b, i) => (
        <group key={`bore-${i}`} position={[b.x, 0.55, 0]}>
          <mesh material={mats.block}>
            <cylinderGeometry args={[CYL_RADIUS, CYL_RADIUS, CYL_HEIGHT, 24, 1, true]} />
          </mesh>
        </group>
      ))}
      {bores.map((b, i) => (
        <group key={`plug-${i}`} position={[b.x, 1.62, 0]}>
          <mesh material={mats.sparkPlugBody}>
            <cylinderGeometry args={[0.05, 0.07, 0.14, 12]} />
          </mesh>
          <mesh material={mats.sparkPlugTop} position={[0, 0.10, 0]}>
            <cylinderGeometry args={[0.035, 0.035, 0.06, 8]} />
          </mesh>
        </group>
      ))}
      {bores.map((b, i) => (
        <Piston
          key={`piston-${i}`}
          x={b.x} phase={sign > 0 ? b.phaseL : b.phaseR}
          rpmHz={rpmHz} radius={CYL_RADIUS - 0.04}
          faultClass={faultClass} severity={severity}
          index={i + (sign > 0 ? 0 : CYL_PER_BANK)}
        />
      ))}

      {/* Exhaust manifold on the underside of the bank */}
      <mesh material={mats.exhaust} position={[0, 0.45, -0.58]}>
        <boxGeometry args={[BLOCK_X - 0.2, 0.28, 0.22]} />
      </mesh>
      {bores.map((b, i) => (
        <mesh key={`ex-${i}`} material={mats.exhaust}
              position={[b.x, 0.50, -0.48]}
              rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.10, 0.10, 0.30, 14]} />
        </mesh>
      ))}
    </group>
  )

  return (
    <group>
      {/* Central crankcase */}
      <mesh material={mats.block} position={[0, 0, 0]} castShadow receiveShadow>
        <boxGeometry args={[BLOCK_X, BLOCK_Y, BLOCK_Z]} />
      </mesh>
      {/* Shallow sump */}
      <mesh material={mats.oilPan} position={[0, -BLOCK_Y / 2 - 0.16, 0]} castShadow>
        <boxGeometry args={[BLOCK_X - 0.15, 0.24, BLOCK_Z - 0.10]} />
      </mesh>
      {boltRow(CYL_PER_BANK * 3, -BLOCK_X / 2 + 0.18, BLOCK_X / 2 - 0.18, -BLOCK_Y / 2 - 0.05, BLOCK_Z / 2 - 0.08, mats.bolt, 0.032)}
      {boltRow(CYL_PER_BANK * 3, -BLOCK_X / 2 + 0.18, BLOCK_X / 2 - 0.18, -BLOCK_Y / 2 - 0.05, -(BLOCK_Z / 2 - 0.08), mats.bolt, 0.032)}

      {/* Crankshaft along X */}
      <mesh material={mats.crank}
            position={[0, 0, 0]}
            rotation={[0, 0, Math.PI / 2]}
            castShadow>
        <cylinderGeometry args={[0.16, 0.16, BLOCK_X + 0.6, 22]} />
      </mesh>
      {/* Front pulley */}
      <group position={[BLOCK_X / 2 + 0.30, 0, 0]}>
        <mesh material={mats.pulley} rotation={[0, 0, Math.PI / 2]}>
          <cylinderGeometry args={[0.30, 0.30, 0.16, 28]} />
        </mesh>
        <mesh material={mats.belt} rotation={[0, 0, Math.PI / 2]}>
          <torusGeometry args={[0.27, 0.015, 8, 28]} />
        </mesh>
      </group>

      <Bank sign={+1} />
      <Bank sign={-1} />

      {/* Air filter / intake box sitting on top of the crankcase */}
      <mesh material={mats.manifold} position={[0, BLOCK_Y / 2 + 0.30, 0]} castShadow>
        <boxGeometry args={[1.50, 0.34, 0.55]} />
      </mesh>
      <mesh material={mats.chrome}
            position={[BLOCK_X / 2 + 0.05, BLOCK_Y / 2 + 0.30, 0]}
            rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.10, 0.10, 0.40, 14]} />
      </mesh>

      {/* Twin tailpipes — one each side, like a flat-4 should have */}
      {[+1, -1].map(s => (
        <group key={s}>
          <mesh material={mats.exhaust}
                position={[BLOCK_X / 2 - 0.10, -0.10, s * 1.45]}
                rotation={[Math.PI / 2, 0, 0]}>
            <cylinderGeometry args={[0.11, 0.11, 1.30, 16]} />
          </mesh>
          <mesh material={mats.chrome}
                position={[BLOCK_X / 2 - 0.10, -0.10, s * 2.05]}
                rotation={[Math.PI / 2, 0, 0]}>
            <cylinderGeometry args={[0.13, 0.11, 0.08, 16]} />
          </mesh>
        </group>
      ))}
    </group>
  )
}

// ── Main export ───────────────────────────────────────────────────────────
export default function EngineBlock({
  engineId, rpmHz, faultClass, severity,
  headEmissiveColor, headEmissiveStrength,
  fuelHighlight = 0, ignitionHighlight = 0,
}) {
  const mats = useMaterials(headEmissiveColor, headEmissiveStrength,
                            fuelHighlight, ignitionHighlight)
  const common = { rpmHz, faultClass, severity, mats }

  if (engineId === 'gengine2') return <V8 {...common} />
  if (engineId === 'pengines') return <Flat4 {...common} />
  return <Inline6 {...common} />
}

// ── Per-engine dimensions/anchors used by FaultEffects ────────────────────
export function getEngineDims(engineId) {
  if (engineId === 'gengine2') {
    const BLOCK_X = 4 * 0.78 + 0.4   // 3.52
    const BLOCK_Z = 1.45
    return {
      BLOCK_X, BLOCK_Z,
      SPARK_COUNT: 8,
      SPARK_POSITIONS: [
        // Both banks of 4, tilted ±30° from vertical. Approximate world positions.
        ...Array.from({ length: 4 }, (_, i) => {
          const x = -(3 * 0.78) / 2 + i * 0.78
          // Tilted plug along +Z bank: rotated ~30°, plug local y=1.62
          const y = 1.62 * Math.cos(Math.PI / 6)
          const z =  1.62 * Math.sin(Math.PI / 6)
          return { x, y, z }
        }),
        ...Array.from({ length: 4 }, (_, i) => {
          const x = -(3 * 0.78) / 2 + i * 0.78
          const y = 1.62 * Math.cos(Math.PI / 6)
          const z = -1.62 * Math.sin(Math.PI / 6)
          return { x, y, z }
        }),
      ],
      TAILPIPE_END: [BLOCK_X / 2 - 0.2, -0.30, -(BLOCK_Z / 2 + 1.10)],
      FUEL_RAIL_Y: 1.30,
      FUEL_RAIL_Z: 0,
    }
  }
  if (engineId === 'pengines') {
    const BLOCK_X = 2 * 0.95 + 0.5   // 2.40
    const BLOCK_Z = 1.00
    return {
      BLOCK_X, BLOCK_Z,
      SPARK_COUNT: 4,
      SPARK_POSITIONS: [
        // Banks rotated ±90° around X → plugs face ±Z.
        ...Array.from({ length: 2 }, (_, i) => ({
          x: -(0.95) / 2 + i * 0.95, y: 0, z:  1.62,
        })),
        ...Array.from({ length: 2 }, (_, i) => ({
          x: -(0.95) / 2 + i * 0.95, y: 0, z: -1.62,
        })),
      ],
      TAILPIPE_END: [BLOCK_X / 2 - 0.10, -0.10, 2.15],
      FUEL_RAIL_Y: BLOCK_Y_FLAT() + 0.30,
      FUEL_RAIL_Z: 0,
    }
  }
  // gengine1 / inline-6 — original anchors
  const BLOCK_X = 6 * 0.78 + 0.4   // 5.08
  const BLOCK_Z = 1.10
  return {
    BLOCK_X, BLOCK_Z,
    SPARK_COUNT: 6,
    SPARK_POSITIONS: Array.from({ length: 6 }, (_, i) => ({
      x: -(5 * 0.78) / 2 + i * 0.78, y: 1.42, z: 0,
    })),
    TAILPIPE_END: [BLOCK_X / 2 - 0.2, 0.55, -(BLOCK_Z / 2 + 1.95)],
    FUEL_RAIL_Y: 1.02,
    FUEL_RAIL_Z: BLOCK_Z / 2 + 0.20,
  }
}

// Helper kept local so the static export above stays consistent
function BLOCK_Y_FLAT() { return 0.95 }

// Back-compat default dims (inline-6) — older code still imports ENGINE_DIMS
export const ENGINE_DIMS = {
  BLOCK_X: 5.08,
  BLOCK_Y: 1.20,
  BLOCK_Z: 1.10,
  CYL_COUNT: 6,
  CYL_SPACING: 0.78,
  CYL_RADIUS: 0.30,
  CYL_HEIGHT: 1.10,
}
