import { Suspense, useEffect, useMemo, useRef, useState } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Environment, ContactShadows } from '@react-three/drei'
import { EffectComposer, Bloom } from '@react-three/postprocessing'
import { useSimulationContext } from '../../context/SimulationContext'
import EngineBlock from './EngineBlock'
import FaultEffects from './FaultEffects'
import './EngineViewer.css'

const CAMERA_VIEWS = {
  iso:     { label: 'ISOMETRIC',     position: [ 5.2, 3.0,  6.4] },
  front:   { label: 'FRONT',         position: [ 0.0, 1.4,  7.8] },
  back:    { label: 'BACK',          position: [ 0.0, 1.4, -7.8] },
  left:    { label: 'LEFT',          position: [-7.8, 1.4,  0.0] },
  right:   { label: 'RIGHT',         position: [ 7.8, 1.4,  0.0] },
  top:     { label: 'TOP',           position: [ 0.0, 8.5,  0.01] },
  cinema:  { label: 'CINEMATIC',     position: [-4.8, 2.2,  6.6] },
}

/**
 * Snaps the camera + OrbitControls target to a named preset whenever
 * `view` changes. After the snap the user can still orbit manually —
 * this just resets the framing.
 */
function CameraRig({ view, controlsRef }) {
  const { camera } = useThree()
  useEffect(() => {
    const preset = CAMERA_VIEWS[view]
    if (!preset) return
    camera.position.set(...preset.position)
    camera.lookAt(0, 0, 0)
    const controls = controlsRef.current
    if (controls) {
      controls.target.set(0, 0, 0)
      controls.update()
    }
  }, [view, camera, controlsRef])
  return null
}

/**
 * Procedural engine vibration. Wraps the engine and applies tiny
 * per-frame translations and rotations driven by layered sine waves
 * (smooth high-frequency idle vibration) plus a stochastic jolt channel
 * used for misfire-style fault shake.
 *
 *   intensity — 0..~2.5, scales smooth vibration amplitude
 *   jolt      — 0..~2.0, probability + amplitude of irregular kicks
 *   active    — when false the group eases back to rest
 */
function ShakingEngineGroup({ children, intensity, jolt, active }) {
  const ref = useRef()
  const offset = useRef({ x: 0, y: 0, z: 0, rx: 0, rz: 0 })

  useFrame((state, delta) => {
    const g = ref.current
    if (!g) return

    // Smooth target → current with critically-damped lerp so the shake
    // doesn't snap on/off when isRunning toggles.
    const k = Math.min(1, delta * 14)

    if (!active || intensity <= 0) {
      offset.current.x  *= 1 - k
      offset.current.y  *= 1 - k
      offset.current.z  *= 1 - k
      offset.current.rx *= 1 - k
      offset.current.rz *= 1 - k
    } else {
      const t = state.clock.elapsedTime
      const amp = intensity * 0.012

      // Layered sines at coprime-ish frequencies give an organic feel
      // without expensive perlin noise.
      const tx = Math.sin(t * 47 + 1.3) * amp * 0.8 + Math.sin(t * 11 + 2.1) * amp * 0.30
      const ty = Math.sin(t * 53 + 0.7) * amp * 1.0 + Math.sin(t * 13 + 1.4) * amp * 0.40
      const tz = Math.sin(t * 41 + 2.5) * amp * 0.6 + Math.sin(t *  9 + 0.4) * amp * 0.30
      const trx = Math.sin(t * 31 + 1.0) * amp * 0.18
      const trz = Math.sin(t * 37 + 0.5) * amp * 0.18

      // Stochastic kicks — misfire pulses, with severity-scaled probability
      let jx = 0, jy = 0, jz = 0
      if (jolt > 0 && Math.random() < 0.03 + jolt * 0.05) {
        const j = jolt * 0.045
        jx = (Math.random() - 0.5) * j
        jy = (Math.random() - 0.5) * j * 1.4
        jz = (Math.random() - 0.5) * j
      }

      offset.current.x  += ((tx + jx) - offset.current.x ) * k
      offset.current.y  += ((ty + jy) - offset.current.y ) * k
      offset.current.z  += ((tz + jz) - offset.current.z ) * k
      offset.current.rx += (trx - offset.current.rx) * k
      offset.current.rz += (trz - offset.current.rz) * k
    }

    g.position.set(offset.current.x, offset.current.y, offset.current.z)
    g.rotation.set(offset.current.rx, 0, offset.current.rz)
  })

  return <group ref={ref}>{children}</group>
}

const HOTSPOTS = {
  block:    { label: 'ENGINE BLOCK' },
  exhaust:  { label: 'EXHAUST'      },
  plugs:    { label: 'SPARK PLUGS'  },
  rail:     { label: 'FUEL RAIL'    },
}

function TooltipOverlay({ hovered, state }) {
  if (!hovered) return null
  const { currentFault, lambdaCurrent, coCurrent, hcCurrent, noxCurrent, sliders } = state
  const rows = hovered === 'block' ? [
    { label: 'FAULT',  value: currentFault.name.toUpperCase(), color: currentFault.class === 0 ? '#00ff88' : '#ffaa00' },
    { label: 'LAMBDA', value: `${lambdaCurrent.toFixed(4)} σ`, color: '#00d4ff' },
    { label: 'RPM',    value: `${sliders.rpm.toFixed(2)} σ`,   color: '#00ff88' },
    { label: 'LOAD',   value: `${sliders.load.toFixed(2)} σ`,  color: '#00ff88' },
  ] : hovered === 'exhaust' ? [
    { label: 'CO',  value: coCurrent.toFixed(4),  color: '#ff3355' },
    { label: 'HC',  value: hcCurrent.toFixed(4),  color: '#ffaa00' },
    { label: 'NOx', value: noxCurrent.toFixed(4), color: '#ff9944' },
  ] : hovered === 'plugs' ? [
    { label: 'IGNITION', value: `${sliders.ignitionAngle.toFixed(2)} σ`, color: '#7b68ee' },
    { label: 'FAULT',    value: currentFault.class === 3 ? 'MISFIRE' : 'OK', color: currentFault.class === 3 ? '#ff3355' : '#00ff88' },
  ] : [
    { label: 'LAMBDA', value: `${lambdaCurrent.toFixed(4)} σ`, color: '#00d4ff' },
    { label: 'TRIM',   value: 'see indicator',                 color: '#00ff88' },
  ]
  return (
    <div className="engine-tooltip">
      <div className="engine-tooltip-title">{HOTSPOTS[hovered].label}</div>
      {rows.map(r => (
        <div key={r.label} className="engine-tooltip-row">
          <span className="engine-tooltip-label">{r.label}</span>
          <span className="engine-tooltip-value" style={{ color: r.color }}>{r.value}</span>
        </div>
      ))}
    </div>
  )
}

export default function EngineViewer() {
  const { state } = useSimulationContext()
  const [hovered, setHovered] = useState(null)
  const [view, setView] = useState('iso')
  const controlsRef = useRef()

  const faultClass        = state.currentFault.class
  const lambdaCurrent     = state.lambdaCurrent
  const fuelTrim          = state.lastFuelTrim ?? 0
  const sparkAdv          = state.lastSparkAdv ?? 0
  const correctionActive  = state.autoCorrection && (Math.abs(fuelTrim) > 0.001 || Math.abs(sparkAdv) > 0.001)

  // Crank speed — RPM slider is roughly standardised so we map it to a
  // useful audio-visual range, not literal engine RPM.
  const rpmHz = useMemo(() => {
    const x = (state.sliders.rpm + 2) * 0.85 + 1.2
    return Math.max(0.4, Math.min(6.0, x))
  }, [state.sliders.rpm])

  // Cylinder-head emissive driven by fault state
  const headColor    = faultClass === 2 ? '#ff3322' : faultClass === 1 ? '#ff8800' : '#000000'
  const headStrength = faultClass === 2 ? Math.min(1.6, Math.abs(lambdaCurrent) * 0.8 + 0.3)
                     : faultClass === 1 ? 0.35 : 0

  // Severity drives FX magnitude — use |lambda| or ignition offset proxy
  const severity = faultClass === 3
    ? Math.min(1.4, Math.abs(state.sliders.ignitionAngle) * 0.5 + 0.6)
    : Math.min(1.4, Math.abs(lambdaCurrent) * 0.7 + 0.35)

  // ── Shake parameters ───────────────────────────────────────────────────
  // Baseline idle vibration scales with RPM. Rich/lean faults add steady
  // amplitude (rougher idle). Load and large lambda deviations contribute.
  // Misfire (fault 3) is the only source of the irregular "jolt" channel.
  const shakeIntensity = useMemo(() => {
    if (!state.isRunning) return 0
    const rpmTerm  = 0.35 + Math.max(0, (rpmHz - 1.2)) * 0.18    // 0.35..~1.2
    const loadTerm = Math.abs(state.sliders.load) * 0.18
    let v = rpmTerm + loadTerm
    if (faultClass === 1) v += 0.35 + severity * 0.45   // rich → heavier idle
    if (faultClass === 2) v += 0.30 + severity * 0.40   // lean → rough
    if (faultClass === 3) v += 0.25 + severity * 0.30   // misfire baseline
    // Auto-correction reduces vibration as control closes the loop
    if (correctionActive) v *= 0.75
    return Math.max(0, Math.min(2.5, v))
  }, [state.isRunning, rpmHz, state.sliders.load, faultClass, severity, correctionActive])

  const joltIntensity = useMemo(() => {
    if (!state.isRunning) return 0
    if (faultClass !== 3) return 0
    return Math.min(2.0, 0.45 + severity * 1.2)
  }, [state.isRunning, faultClass, severity])

  return (
    <div className="engine-viewer-panel panel">
      <div className="panel-title">
        LIVE ENGINE — {state.currentFault.name.toUpperCase()}
        {correctionActive && <span className="engine-correction-tag">CORRECTING</span>}
        <select
          className="engine-view-select"
          value={view}
          onChange={(e) => setView(e.target.value)}
          aria-label="Camera view"
        >
          {Object.entries(CAMERA_VIEWS).map(([k, v]) => (
            <option key={k} value={k}>{v.label}</option>
          ))}
        </select>
      </div>
      <div className="engine-canvas-wrap">
        <Suspense fallback={<div className="engine-loading">RENDERING...</div>}>
          <Canvas
            shadows
            dpr={[1, 1.7]}
            camera={{ position: [5.2, 3.0, 6.4], fov: 38 }}
            gl={{ antialias: true, powerPreference: 'high-performance' }}
          >
            <color attach="background" args={['#0a0d11']} />
            <fog attach="fog" args={['#0a0d11', 8, 18]} />
            <ambientLight intensity={0.35} />
            <directionalLight position={[5, 6, 4]} intensity={1.1} castShadow shadow-mapSize={[1024, 1024]} />
            <directionalLight position={[-4, 3, -3]} intensity={0.3} color={'#88aaff'} />
            <Environment preset="warehouse" />
            <ContactShadows position={[0, -0.85, 0]} opacity={0.55} scale={9} blur={2.4} far={3} />

            <ShakingEngineGroup
              intensity={shakeIntensity}
              jolt={joltIntensity}
              active={state.isRunning}
            >
              <group
                onPointerOver={(e) => { e.stopPropagation(); setHovered('block') }}
                onPointerOut={() => setHovered(null)}
              >
                <EngineBlock
                  engineId={state.engineId}
                  rpmHz={rpmHz}
                  faultClass={faultClass}
                  severity={severity}
                  headEmissiveColor={headColor}
                  headEmissiveStrength={headStrength}
                />
              </group>
            </ShakingEngineGroup>

            <FaultEffects
              engineId={state.engineId}
              faultClass={faultClass}
              severity={severity}
              lambdaCurrent={lambdaCurrent}
              correctionActive={correctionActive}
              fuelTrim={fuelTrim}
              sparkAdv={sparkAdv}
            />

            <EffectComposer disableNormalPass>
              <Bloom intensity={0.55} mipmapBlur luminanceThreshold={0.35} luminanceSmoothing={0.6} />
            </EffectComposer>

            <OrbitControls
              ref={controlsRef}
              enablePan={false}
              minDistance={4}
              maxDistance={14}
              minPolarAngle={Math.PI / 6}
              maxPolarAngle={Math.PI / 2.1}
            />
            <CameraRig view={view} controlsRef={controlsRef} />
          </Canvas>
        </Suspense>
        <TooltipOverlay hovered={hovered} state={state} />
        <div className="engine-hint">Drag to rotate · Scroll to zoom</div>
      </div>

      {/* Status strip — same layout as the old CarViewer */}
      <div className="engine-status-strip">
        <div className="engine-status-item">
          <span className="engine-status-label">FAULT</span>
          <span className="engine-status-value" style={{
            color: faultClass === 0 ? '#00ff88' : faultClass === 3 ? '#ff3355' : '#ffaa00'
          }}>
            {state.currentFault.name.toUpperCase()}
          </span>
        </div>
        <div className="engine-status-item">
          <span className="engine-status-label">λ</span>
          <span className="engine-status-value mono" style={{ color: '#00d4ff' }}>
            {state.lambdaCurrent.toFixed(3)}σ
          </span>
        </div>
        <div className="engine-status-item">
          <span className="engine-status-label">CO</span>
          <span className="engine-status-value mono" style={{ color: '#ff3355' }}>
            {state.coCurrent.toFixed(3)}
          </span>
        </div>
        <div className="engine-status-item">
          <span className="engine-status-label">NOx</span>
          <span className="engine-status-value mono" style={{ color: '#ff9944' }}>
            {state.noxCurrent.toFixed(3)}
          </span>
        </div>
      </div>
    </div>
  )
}
