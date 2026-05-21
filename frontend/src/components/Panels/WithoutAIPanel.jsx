import { useSimulationContext } from '../../context/SimulationContext'
import './Panels.css'

const FAULT_DATA = {
  1: {
    label:      'Rich Mixture',
    obdDelay:   '~45s',
    riskLevel:  'High',
    riskColor:  'var(--accent-amber)',
    primary:    'Catalyst poisoning & O2 sensor damage',
    consequences: [
      'CO emissions 4x above Euro 6 limit (1.0 g/km)',
      'Catalytic converter overheating from excess HC',
      'O2 sensor contamination — permanent damage',
      'Fuel economy degrades 8–12% per minute',
    ],
    projection: 'Without correction, lambda drifts to −4σ (~40% rich) within 90s',
    extraNote:  '~50 ml/min of unburnt fuel exhausted',
  },
  2: {
    label:      'Lean Mixture',
    obdDelay:   '~60s',
    riskLevel:  'Critical',
    riskColor:  'var(--accent-red)',
    primary:    'Engine knock & piston overheating',
    consequences: [
      'NOx emissions 3x above Euro 6 limit (0.08 g/km)',
      'Cylinder / piston thermal stress — knock risk',
      'Potential engine misfire → rough idle',
      'Pre-ignition damage if sustained >2 min',
    ],
    projection: 'Lambda drifts to +4σ (lean surge) — misfires likely within 60s',
    extraNote:  'NOx: a primary cause of smog & respiratory disease',
  },
  3: {
    label:      'Ignition Fault',
    obdDelay:   '~20s',
    riskLevel:  'Critical',
    riskColor:  'var(--accent-red)',
    primary:    'Catalytic converter meltdown risk',
    consequences: [
      'Unburnt fuel combusts inside catalyst (>900°C)',
      'HC emissions 5x legal limit → substrate melt',
      'Cylinder misfires — engine roughness & vibration',
      'Permanent catalyst damage possible in 3 min',
    ],
    projection: 'Catalyst substrate temp reaches 900–1100°C within 3 minutes',
    extraNote:  '~30 ml/min of unburnt fuel reaches the catalyst',
  },
}

export default function WithoutAIPanel() {
  const { state } = useSimulationContext()
  const { faultSnapshot, healedSnapshot } = state

  // Show only while fault is active (not after healing — HealingReport covers that)
  if (!faultSnapshot || healedSnapshot) return null

  const nameMap = { 'Rich Mixture': 1, 'Lean Mixture': 2, 'Ignition Fault': 3 }
  const faultClass = nameMap[faultSnapshot.faultName] ?? 1
  const data = FAULT_DATA[faultClass]

  return (
    <div className="panel without-ai-panel">
      <div className="panel-title" style={{ color: 'var(--accent-red)' }}>Without AI twin</div>

      {/* Detection speed comparison */}
      <div className="wai-detect-row">
        <div className="wai-detect-box">
          <div className="wai-detect-label">AI twin</div>
          <div className="wai-detect-time" style={{ color: 'var(--accent-green)' }}>0.5s</div>
          <div className="wai-detect-sub">detected</div>
        </div>
        <div className="wai-vs">vs</div>
        <div className="wai-detect-box">
          <div className="wai-detect-label">OBD-II</div>
          <div className="wai-detect-time" style={{ color: 'var(--accent-red)' }}>{data.obdDelay}</div>
          <div className="wai-detect-sub">detected</div>
        </div>
        <div className="wai-vs">vs</div>
        <div className="wai-detect-box">
          <div className="wai-detect-label">Mechanic</div>
          <div className="wai-detect-time" style={{ color: 'var(--accent-red)' }}>days</div>
          <div className="wai-detect-sub">detected</div>
        </div>
      </div>

      <div className="wai-divider" />

      <div className="wai-risk-row">
        <span className="fault-meta-label">Uncontrolled risk</span>
        <span className="fault-meta-value mono" style={{ color: data.riskColor }}>
          {data.riskLevel}
        </span>
      </div>

      <div className="wai-primary">{data.primary}</div>

      <div className="wai-consequences">
        {data.consequences.map((c, i) => (
          <div key={i} className="wai-item">
            <span className="wai-bullet" style={{ color: 'var(--accent-red)' }}>&#9642;</span>
            <span>{c}</span>
          </div>
        ))}
      </div>

      <div className="wai-divider" />

      <div className="wai-projection-box">
        <span className="fault-meta-label">Uncontrolled trajectory</span>
        <div className="wai-projection-text">{data.projection}</div>
        <div className="wai-extra-note">{data.extraNote}</div>
      </div>
    </div>
  )
}
