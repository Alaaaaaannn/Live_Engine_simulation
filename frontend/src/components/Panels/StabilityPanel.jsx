import { useSimulationContext } from '../../context/SimulationContext'
import './StabilityPanel.css'

const NAMES = {
  0: 'Normal',
  1: 'Rich mixture',
  2: 'Lean mixture',
  3: 'Ignition fault',
}

export default function StabilityPanel() {
  const { state } = useSimulationContext()
  const agreement = state.stabilityAgreement ?? 0
  const label     = state.stabilityLabel ?? 0
  const pct       = Math.round(agreement * 100)

  const color =
    agreement < 0.60 ? 'var(--accent-red)'   :
    agreement < 0.85 ? 'var(--accent-amber)' :
                       'var(--accent-green)'

  const phase =
    agreement < 0.60 ? 'Unstable' :
    agreement < 0.85 ? 'Settling' :
                       'Stable'

  return (
    <div className="stability-panel panel">
      <div className="panel-title">Temporal stability</div>

      <div className="stability-row">
        <span className="stability-label">Majority</span>
        <span className="stability-majority" style={{ color }}>{NAMES[label] ?? '—'}</span>
      </div>

      <div className="stability-bar">
        <div
          className="stability-fill"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>

      <div className="stability-row">
        <span className="stability-label">Agreement</span>
        <span className="stability-pct" style={{ color }}>{pct}%</span>
      </div>

      <div className="stability-phase" style={{ color }}>
        {phase}
      </div>
    </div>
  )
}
