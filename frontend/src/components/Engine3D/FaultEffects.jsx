import { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Sparkles, Trail } from '@react-three/drei'
import * as THREE from 'three'
import { getEngineDims } from './EngineBlock'

function FlickerLight({ color, baseIntensity, position, rate = 8 }) {
  const lightRef = useRef()
  useFrame((state) => {
    if (!lightRef.current) return
    const t = state.clock.elapsedTime
    const f = 0.7 + 0.3 * Math.sin(t * rate + position[0])
    lightRef.current.intensity = baseIntensity * f
  })
  return <pointLight ref={lightRef} color={color} intensity={baseIntensity} position={position} distance={4} />
}

/**
 * Visual sphere that loops along the fuel rail, leaving a trail.
 * Mounted only when fuel-trim correction is active.
 */
function FuelFlowTrail({ direction = 1, dims }) {
  const ref = useRef()
  useFrame((state) => {
    if (!ref.current) return
    const t = state.clock.elapsedTime * direction
    const half = (dims.BLOCK_X - 0.6) / 2
    ref.current.position.x = Math.sin(t * 1.4) * half
  })
  return (
    <Trail width={0.10} length={4} color={'#00ff88'} attenuation={(t) => t * t}>
      <mesh ref={ref} position={[0, dims.FUEL_RAIL_Y + 0.04, dims.FUEL_RAIL_Z + 0.04]}>
        <sphereGeometry args={[0.05, 12, 12]} />
        <meshStandardMaterial color={'#00ff88'} emissive={'#00ff88'} emissiveIntensity={1.2} />
      </mesh>
    </Trail>
  )
}

/**
 * Per-cylinder spark arc (ignition fault).  Rendered as a short jittered
 * line above each spark plug.
 */
function SparkArcs({ dims }) {
  const linesRef = useRef([])
  const geometries = useMemo(() => {
    return dims.SPARK_POSITIONS.map(({ x, y, z }) => {
      const geo = new THREE.BufferGeometry()
      const pts = new Float32Array(6)
      pts[0] = x;       pts[1] = y;       pts[2] = z
      pts[3] = x;       pts[4] = y + 0.32; pts[5] = z
      geo.setAttribute('position', new THREE.BufferAttribute(pts, 3))
      return { geo, x, y, z }
    })
  }, [dims])

  useFrame(() => {
    linesRef.current.forEach((line, i) => {
      if (!line) return
      const g = geometries[i]
      const pos = line.geometry.attributes.position.array
      pos[3] = g.x + (Math.random() - 0.5) * 0.08
      pos[5] = g.z + (Math.random() - 0.5) * 0.08
      pos[4] = g.y + 0.32 + (Math.random() - 0.5) * 0.06
      line.geometry.attributes.position.needsUpdate = true
    })
  })

  return (
    <group>
      {geometries.map((g, i) => (
        <line key={`spark-${i}`} ref={el => (linesRef.current[i] = el)}>
          <bufferGeometry attach="geometry" {...g.geo} />
          <lineBasicMaterial color={'#aaccff'} linewidth={2} transparent opacity={0.85} />
        </line>
      ))}
    </group>
  )
}

export default function FaultEffects({ engineId, faultClass, severity, lambdaCurrent, correctionActive, fuelTrim, sparkAdv }) {
  const dims = getEngineDims(engineId)
  const { BLOCK_X, BLOCK_Z, TAILPIPE_END } = dims
  // Mild ambient cyan for healthy engines
  if (faultClass === 0 && !correctionActive) {
    return (
      <group>
        <pointLight color={'#00d4ff'} intensity={0.45} position={[0, 0.8, 0]} distance={3} />
        <Sparkles
          count={18}
          scale={[1.4, 0.4, 0.4]}
          size={2.5}
          speed={0.4}
          opacity={0.35}
          color={'#aac8d4'}
          position={TAILPIPE_END}
        />
      </group>
    )
  }

  const sev = Math.max(0.15, Math.min(1.5, severity))

  return (
    <group>
      {/* Rich (1): black/grey smoke + orange combustion flicker */}
      {faultClass === 1 && (
        <>
          <Sparkles
            count={Math.floor(40 * sev) + 20}
            scale={[1.6, 0.7, 0.7]}
            size={4 * sev}
            speed={1.0}
            opacity={0.75}
            color={'#2a2a2a'}
            position={TAILPIPE_END}
          />
          <FlickerLight color={'#ff8800'} baseIntensity={1.6 * sev} position={[0, 0.5, 0]} rate={11} />
        </>
      )}

      {/* Lean (2): pale-yellow exhaust shimmer; emissive heads handled in EngineBlock */}
      {faultClass === 2 && (
        <>
          <Sparkles
            count={Math.floor(30 * sev) + 15}
            scale={[1.6, 0.5, 0.5]}
            size={2.5 * sev}
            speed={0.7}
            opacity={0.6}
            color={'#ffe082'}
            position={TAILPIPE_END}
          />
          <FlickerLight color={'#ffe082'} baseIntensity={0.8 * sev} position={[0, 0.7, -BLOCK_Z / 2 - 0.3]} rate={5} />
        </>
      )}

      {/* Ignition (3): spark arcs, white-blue puffs */}
      {faultClass === 3 && (
        <>
          <SparkArcs dims={dims} />
          <Sparkles
            count={Math.floor(50 * sev) + 25}
            scale={[BLOCK_X * 0.6, 0.5, 0.5]}
            size={3 * sev}
            speed={1.4}
            opacity={0.8}
            color={'#88ccff'}
            position={[0, 1.5, 0]}
          />
        </>
      )}

      {/* Correction in progress — green fuel-flow trail + cyan pulse */}
      {correctionActive && Math.abs(fuelTrim) > 0.001 && <FuelFlowTrail direction={fuelTrim >= 0 ? 1 : -1} dims={dims} />}
      {correctionActive && Math.abs(sparkAdv) > 0.001 && (
        <FlickerLight color={'#00d4ff'} baseIntensity={1.0} position={[0, 1.2, 0]} rate={6} />
      )}
    </group>
  )
}
