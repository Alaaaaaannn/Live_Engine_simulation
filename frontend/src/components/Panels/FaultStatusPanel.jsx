import { useSpring, animated } from '@react-spring/web'
import { useSimulationContext } from '../../context/SimulationContext'
import { formatBoth } from '../../utils/units'
import './Panels.css'

// Subdued status colors that match the rest of the dashboard palette.
// `bg` is a very low-opacity tint on top of the panel surface, not a
// hardcoded near-black like the previous neon design.
const FAULT_COLORS = {
  0: { tint: 'rgba(34,197,94,0.06)',  border: 'rgba(34,197,94,0.35)',  text: 'var(--accent-green)' },
  1: { tint: 'rgba(245,158,11,0.06)', border: 'rgba(245,158,11,0.35)', text: 'var(--accent-amber)' },
  2: { tint: 'rgba(245,158,11,0.06)', border: 'rgba(245,158,11,0.35)', text: 'var(--accent-amber)' },
  3: { tint: 'rgba(239,68,68,0.06)',  border: 'rgba(239,68,68,0.35)',  text: 'var(--accent-red)'   },
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
  if (cls === 0) return { label: 'Nominal',         color: 'var(--accent-green)' }
  if (prog < 25)  return { label: 'Fault detected', color: 'var(--accent-red)'   }
  if (prog < 60)  return { label: 'Correcting',     color: 'var(--accent-amber)' }
  if (prog < 95)  return { label: 'Converging',     color: 'var(--accent-cyan)'  }
  return { label: 'Healed', color: 'var(--accent-green)' }
}

// Severity → colour ramp for the parameter-state badge. Distinct from
// the fault-class colours so the operator can tell at a glance which
// signal is firing.
function paramStateColor(severity) {
  if (severity < 0.001) return 'var(--accent-green)'
  if (severity < 0.34)  return 'var(--accent-cyan)'
  if (severity < 0.67)  return 'var(--accent-amber)'
  return 'var(--accent-red)'
}

export default function FaultStatusPanel() {
  const { state } = useSimulationContext()
  const { class: cls, name, confidence } = state.currentFault
  const colors = FAULT_COLORS[cls] ?? FAULT_COLORS[0]
  const ps = state.parameterState ?? { label: 'Nominal', severity: 0, param: null }
  const psColor = paramStateColor(ps.severity)

  const prog = healProgress(state.lambdaCurrent, cls)
  const phase = healPhase(prog, cls)

  const bgSpring = useSpring({
    background:  colors.tint,
    borderColor: colors.border,
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

  const barColor =
    cls === 0       ? 'var(--accent-green)' :
    prog < 25       ? 'var(--accent-red)'   :
    prog < 60       ? 'var(--accent-amber)' :
    prog < 95       ? 'var(--accent-cyan)'  :
                      'var(--accent-green)'

  return (
    <animated.div className="panel fault-status-panel" style={bgSpring}>
      <div className="panel-title" style={{ color: colors.text }}>Fault status</div>

      <animated.div className="fault-name" style={{ color: colors.text }}>
        {name}
      </animated.div>

      <div className="fault-meta">
        <span className="fault-meta-label">Confidence</span>
        <animated.span className="fault-meta-value mono" style={{ color: colors.text }}>
          {confSpring.conf.to(v => v.toFixed(1) + '%')}
        </animated.span>
      </div>

      <div className="fault-meta">
        <span className="fault-meta-label">Lambda</span>
        <span className="fault-meta-value mono text-cyan">
          {formatBoth('lambda', state.lambdaCurrent)}
        </span>
      </div>

      <div className="fault-meta">
        <span className="fault-meta-label">Parameter state</span>
        <span className="fault-meta-value mono" style={{ color: psColor }}>
          {ps.label}
          {ps.severity > 0 && ` · ${(ps.severity * 100).toFixed(0)}%`}
        </span>
      </div>

      {/* Self-healing progress */}
      <div className="heal-section">
        <div className="heal-header">
          <span className="fault-meta-label">Self-healing</span>
          <span className="heal-phase" style={{ color: phase.color }}>{phase.label}</span>
        </div>
        <div className="heal-track">
          <animated.div
            className="heal-fill"
            style={{
              width: barSpring.width.to(w => `${w}%`),
              background: barColor,
            }}
          />
        </div>
        <div className="heal-pct mono" style={{ color: phase.color }}>
          {Math.round(prog)}%
        </div>
      </div>

      {state.isRunning && cls !== 0 && (
        <div className="heal-stats">
          <span className="fault-meta-label">Corrections applied</span>
          <span className="fault-meta-value mono" style={{ color: 'var(--accent-cyan)' }}>
            {state.twinLog.filter(e => e.approved && e.faultName !== 'Normal').length}
          </span>
        </div>
      )}

      {(state.converged && cls === 0) || prog >= 95 ? (
        <div className="convergence-badge">System nominal</div>
      ) : null}
    </animated.div>
  )
}
