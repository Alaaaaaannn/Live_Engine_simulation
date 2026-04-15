import { useRef, useState, Suspense } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { useGLTF, OrbitControls, Environment, ContactShadows } from '@react-three/drei'
import { useSimulationContext } from '../../context/SimulationContext'
import './CarViewer.css'

const CAR_URL = 'https://threejs.org/examples/models/gltf/ferrari.glb'

const PARTS = {
  body:    { label: 'ENGINE',   params: ['lambda', 'rpm', 'load', 'ignition'] },
  exhaust: { label: 'EXHAUST',  params: ['co', 'hc', 'nox'] },
  wheels:  { label: 'DRIVETRAIN', params: ['rpm', 'load'] },
}

function CarModel({ faultClass, lambdaCurrent, emissionLevel, hovered, onHover }) {
  const groupRef = useRef()
  const { scene } = useGLTF(CAR_URL)

  // Slow auto-rotation
  useFrame((_, delta) => {
    if (groupRef.current) {
      groupRef.current.rotation.y += delta * 0.3
    }
  })

  // Color based on fault state
  const bodyColor = faultClass === 0 ? '#00ff88'
    : faultClass === 1 ? '#ffaa00'
    : faultClass === 2 ? '#00d4ff'
    : '#ff3355'

  return (
    <group ref={groupRef} dispose={null}>
      <primitive
        object={scene}
        scale={[0.01, 0.01, 0.01]}
        position={[0, -0.4, 0]}
        onPointerOver={(e) => { e.stopPropagation(); onHover('body') }}
        onPointerOut={() => onHover(null)}
      />
    </group>
  )
}

function TooltipOverlay({ hovered, state }) {
  if (!hovered) return null

  const { currentFault, lambdaCurrent, coCurrent, hcCurrent, noxCurrent,
          sliders } = state

  const rows = hovered === 'body' ? [
    { label: 'FAULT',   value: currentFault.name.toUpperCase(), color: currentFault.class === 0 ? '#00ff88' : '#ffaa00' },
    { label: 'LAMBDA',  value: `${lambdaCurrent.toFixed(4)} σ`,  color: '#00d4ff' },
    { label: 'RPM',     value: `${sliders.rpm.toFixed(2)} σ`,    color: '#00ff88' },
    { label: 'LOAD',    value: `${sliders.load.toFixed(2)} σ`,   color: '#00ff88' },
  ] : hovered === 'exhaust' ? [
    { label: 'CO',  value: coCurrent.toFixed(4),  color: '#ff3355' },
    { label: 'HC',  value: hcCurrent.toFixed(4),  color: '#ffaa00' },
    { label: 'NOx', value: noxCurrent.toFixed(4), color: '#ff9944' },
  ] : [
    { label: 'RPM',  value: `${sliders.rpm.toFixed(2)} σ`,  color: '#00ff88' },
    { label: 'LOAD', value: `${sliders.load.toFixed(2)} σ`,  color: '#00ff88' },
  ]

  return (
    <div className="car-tooltip">
      <div className="car-tooltip-title">{PARTS[hovered]?.label}</div>
      {rows.map(r => (
        <div key={r.label} className="car-tooltip-row">
          <span className="car-tooltip-label">{r.label}</span>
          <span className="car-tooltip-value" style={{ color: r.color }}>{r.value}</span>
        </div>
      ))}
    </div>
  )
}

export default function CarViewer() {
  const { state } = useSimulationContext()
  const [hovered, setHovered] = useState(null)
  const [loadError, setLoadError] = useState(false)

  const faultClass = state.currentFault.class
  const emissionLevel = Math.abs(state.lambdaCurrent)

  if (loadError) return null

  return (
    <div className="car-viewer-panel panel">
      <div className="panel-title">LIVE ENGINE MODEL</div>
      <div className="car-canvas-wrap">
        <Suspense fallback={<div className="car-loading">LOADING MODEL...</div>}>
          <Canvas
            camera={{ position: [4, 2, 6], fov: 35 }}
            onError={() => setLoadError(true)}
          >
            <ambientLight intensity={0.4} />
            <directionalLight position={[5, 5, 5]} intensity={1} />
            <Environment preset="city" />
            <ContactShadows
              position={[0, -0.5, 0]}
              opacity={0.4}
              scale={10}
              blur={2}
            />
            <CarModel
              faultClass={faultClass}
              lambdaCurrent={state.lambdaCurrent}
              emissionLevel={emissionLevel}
              hovered={hovered}
              onHover={setHovered}
            />
            <OrbitControls
              enablePan={false}
              minDistance={3}
              maxDistance={12}
              minPolarAngle={Math.PI / 6}
              maxPolarAngle={Math.PI / 2.2}
            />
          </Canvas>
        </Suspense>
        <TooltipOverlay hovered={hovered} state={state} />
        <div className="car-hint">Drag to rotate · Hover for data</div>
      </div>

      {/* Status strip */}
      <div className="car-status-strip">
        <div className="car-status-item">
          <span className="car-status-label">FAULT</span>
          <span className="car-status-value" style={{
            color: faultClass === 0 ? '#00ff88' : faultClass === 3 ? '#ff3355' : '#ffaa00'
          }}>
            {state.currentFault.name.toUpperCase()}
          </span>
        </div>
        <div className="car-status-item">
          <span className="car-status-label">λ</span>
          <span className="car-status-value mono" style={{ color: '#00d4ff' }}>
            {state.lambdaCurrent.toFixed(3)}σ
          </span>
        </div>
        <div className="car-status-item">
          <span className="car-status-label">CO</span>
          <span className="car-status-value mono" style={{ color: '#ff3355' }}>
            {state.coCurrent.toFixed(3)}
          </span>
        </div>
        <div className="car-status-item">
          <span className="car-status-label">NOx</span>
          <span className="car-status-value mono" style={{ color: '#ff9944' }}>
            {state.noxCurrent.toFixed(3)}
          </span>
        </div>
      </div>
    </div>
  )
}

useGLTF.preload(CAR_URL)
