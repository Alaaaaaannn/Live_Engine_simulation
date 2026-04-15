import { useSpring, animated } from '@react-spring/web'
import { useSimulationContext } from '../../context/SimulationContext'
import './Panels.css'

const FAULT_COLORS = {
  0: { bg: '#001a0d', border: '#00ff88', text: '#00ff88', glow: '0 0 20px #00ff8844' },
  1: { bg: '#1a0a00', border: '#ffaa00', text: '#ffaa00', glow: '0 0 20px #ffaa0044' },
  2: { bg: '#1a0a00', border: '#ffaa00', text: '#ffaa00', glow: '0 0 20px #ffaa0044' },
  3: { bg: '#1a0008', border: '#ff3355', text: '#ff3355', glow: '0 0 20px #ff335544' },
}

// How far lambda is from stoichiometric 0 → healing progress 0–100%
function healProgress(lambdaCurrent, cls) {
  if (cls === 0) return 100
  const dist = Math.abs(lambdaCurrent)
  // fault injected at ±1.5σ — so 1.5 = 0%, 0.05 = 100%
  const maxDist = 1.5
  const prog = Math.max(0, Math.min(100, (1 - (dist - 0.05) / (maxDist - 0.05)) * 100))
  return prog
}

function healPhase(prog, cls) {
  if (cls === 0) return { label: 'NOMINAL', color: '#00ff88' }
  if (prog < 25)  return { label: 'FAULT DETECTED', color: '#ff3355' }
  if (prog < 60)  return { label: 'CORRECTING...', color: '#ffaa00' }
  if (prog < 95)  return { label: 'CONVERGING...', color: '#00d4ff' }
  return { label: 'HEALED', color: '#00ff88' }
}

export default function FaultStatusPanel() {
  const { state } = useSimulationContext()
  const { class: cls, name, confidence } = state.currentFault
  const colors = FAULT_COLORS[cls] ?? FAULT_COLORS[0]

  const prog = healProgress(state.lambdaCurrent, cls)
  const phase = healPhase(prog, cls)

  const bgSpring = useSpring({
    background:  colors.bg,
    borderColor: colors.border,
    boxShadow:   colors.glow,
    config: { tension: 100, friction: 18 },
  })

  const confSpring = useSpring({
    conf: confidence * 100,
    config: { mass: 1, tension: 200, friction: 30 },
  })

  const barSpring = useSpring({
    width: prog,
    config: { mass: 1, tension: 120, friction: 20 },
  })

  return (
    <animated.div className="panel fault-status-panel" style={bgSpring}>
      <div className="panel-title" style={{ color: colors.text }}>Fault Status</div>

      <animated.div className="fault-name" style={{ color: colors.text }}>
        {name.toUpperCase()}
      </animated.div>

      <div className="fault-meta">
        <span className="fault-meta-label">CONFIDENCE</span>
        <animated.span className="fault-meta-value mono" style={{ color: colors.text }}>
          {confSpring.conf.to(v => v.toFixed(1) + '%')}
        </animated.span>
      </div>

      <div className="fault-meta">
        <span className="fault-meta-label">LAMBDA</span>
        <span className="fault-meta-value mono text-cyan">
          {state.lambdaCurrent.toFixed(4)} σ
        </span>
      </div>

      {/* Self-healing progress */}
      <div className="heal-section">
        <div className="heal-header">
          <span className="fault-meta-label">SELF-HEALING</span>
          <span className="heal-phase" style={{ color: phase.color }}>{phase.label}</span>
        </div>
        <div className="heal-track">
          <animated.div
            className="heal-fill"
            style={{
              width: barSpring.width.to(w => `${w}%`),
              background: cls === 0
                ? '#00ff88'
                : prog < 25 ? '#ff3355'
                : prog < 60 ? '#ffaa00'
                : prog < 95 ? '#00d4ff'
                : '#00ff88',
              boxShadow: `0 0 8px ${phase.color}88`,
            }}
          />
        </div>
        <div className="heal-pct mono" style={{ color: phase.color }}>
          {Math.round(prog)}%
        </div>
      </div>

      {state.isRunning && cls !== 0 && (
        <div className="heal-stats">
          <span className="fault-meta-label">CORRECTIONS APPLIED</span>
          <span className="fault-meta-value mono" style={{ color: '#00d4ff' }}>
            {state.twinLog.filter(e => e.approved && e.faultName !== 'Normal').length}
          </span>
        </div>
      )}

      {(state.converged && cls === 0) || prog >= 95 ? (
        <div className="convergence-badge">SYSTEM NOMINAL</div>
      ) : null}
    </animated.div>
  )
}
