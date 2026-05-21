import { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

/**
 * Animated piston.  Y-position oscillates with the crank, advanced by
 * its `phase` so the inline-6 fires evenly.  On Ignition fault we
 * randomly skip a frame's update to produce a visible stutter.
 */
const STROKE = 0.42

export default function Piston({ x, phase, rpmHz, radius, faultClass, severity, index }) {
  const groupRef = useRef()
  const skipRef  = useRef(false)
  const rodRef   = useRef()
  const rngOffset = useMemo(() => Math.random(), [])

  const mat = useMemo(
    () => new THREE.MeshStandardMaterial({
      color: '#c8ccd0',
      metalness: 0.92,
      roughness: 0.30,
    }), [])
  const rodMat = useMemo(
    () => new THREE.MeshStandardMaterial({
      color: '#6a6e74',
      metalness: 0.85,
      roughness: 0.40,
    }), [])

  useFrame((state) => {
    if (!groupRef.current) return

    // Misfire stutter on ignition fault — skip update with probability
    // proportional to severity (0..1).  Each piston has independent draws.
    if (faultClass === 3 && severity > 0) {
      const p = Math.min(0.25, severity * 0.20)
      if (Math.random() < p) {
        skipRef.current = true
        return
      }
    }

    const t = state.clock.elapsedTime + rngOffset * 0.5
    const y = Math.sin(t * rpmHz * 2 * Math.PI + phase) * STROKE
    groupRef.current.position.y = y

    // Connecting rod scales/tilts subtly with stroke for life
    if (rodRef.current) {
      rodRef.current.position.y = -0.55 + y * 0.5
      rodRef.current.scale.y = 1.0 + y * 0.05
    }
  })

  return (
    <group position={[x, 0.55, 0]}>
      <group ref={groupRef}>
        {/* Piston body */}
        <mesh material={mat}>
          <cylinderGeometry args={[radius, radius, 0.22, 18]} />
        </mesh>
        {/* Top ring detail */}
        <mesh material={rodMat} position={[0, 0.12, 0]}>
          <cylinderGeometry args={[radius + 0.005, radius + 0.005, 0.02, 18]} />
        </mesh>
      </group>
      {/* Connecting rod toward crankshaft */}
      <mesh ref={rodRef} material={rodMat} position={[0, -0.55, 0]}>
        <boxGeometry args={[0.10, 0.50, 0.10]} />
      </mesh>
    </group>
  )
}
