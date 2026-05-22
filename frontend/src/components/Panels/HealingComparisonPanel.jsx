import { useRef, useEffect } from 'react'
import { useSimulationContext } from '../../context/SimulationContext'
import { formatPhysical } from '../../utils/units'
import './Panels.css'

const ROWS = [
  { key: 'lambda', channel: 'lambda', label: 'Lambda λ',     target: 0.0, desc: 'Air-fuel ratio' },
  { key: 'co',     channel: 'co',     label: 'CO Emissions', target: 0.0, desc: 'Carbon monoxide' },
  { key: 'hc',     channel: 'hc',     label: 'HC Emissions', target: 0.0, desc: 'Hydrocarbons' },
  { key: 'nox',    channel: 'nox',    label: 'NOx Emissions',target: 0.0, desc: 'Nitrogen oxides' },
]

function DeltaArrow({ before, after }) {
  const improved = Math.abs(after) < Math.abs(before)
  const pct = before !== 0
    ? Math.round(Math.abs((Math.abs(after) - Math.abs(before)) / Math.abs(before)) * 100)
    : 0
  return (
    <span style={{ color: improved ? 'var(--accent-green)' : 'var(--accent-red)', fontSize: 10, fontFamily: 'var(--font-mono)' }}>
      {improved ? '▼' : '▲'} {pct}%
    </span>
  )
}

function ValueBar({ value, max }) {
  const pct = Math.min(100, (Math.abs(value) / max) * 100)
  const color = Math.abs(value) < 0.1 ? 'var(--accent-green)'
    : Math.abs(value) < 0.5 ? 'var(--accent-cyan)'
    : Math.abs(value) < 1.0 ? 'var(--accent-amber)'
    : 'var(--accent-red)'
  return (
    <div style={{ height: 3, background: '#ffffff0f', borderRadius: 2, marginTop: 2 }}>
      <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 2,
                    minWidth: 2, transition: 'width 0.6s ease' }} />
    </div>
  )
}

export default function HealingComparisonPanel() {
  const { state } = useSimulationContext()
  const { faultSnapshot, healedSnapshot } = state
  const panelRef = useRef(null)

  // Scroll the right column to show this panel when it first appears
  useEffect(() => {
    if (faultSnapshot && healedSnapshot && panelRef.current) {
      const scrollContainer = panelRef.current.closest('.dashboard-right')
      if (scrollContainer) {
        scrollContainer.scrollTo({ top: scrollContainer.scrollHeight, behavior: 'smooth' })
      }
    }
  }, [faultSnapshot, healedSnapshot])

  // Only show when we have both snapshots (fault was detected and healed)
  if (!faultSnapshot || !healedSnapshot) return null

  const before = {
    lambda: faultSnapshot.lambda,
    co:     faultSnapshot.co,
    hc:     faultSnapshot.hc,
    nox:    faultSnapshot.nox,
  }
  const after = {
    lambda: healedSnapshot.lambda,
    co:     healedSnapshot.co,
    hc:     healedSnapshot.hc,
    nox:    healedSnapshot.nox,
  }

  const cyclesTaken = healedSnapshot.cycle - faultSnapshot.cycle

  return (
    <div ref={panelRef} className="panel comparison-panel">
      <div className="panel-title" style={{ color: 'var(--accent-green)' }}>Healing report</div>

      <div className="comparison-meta">
        <span className="fault-meta-label">Fault detected</span>
        <span className="fault-meta-value mono" style={{ color: 'var(--accent-red)' }}>
          {faultSnapshot.faultName}
        </span>
      </div>
      <div className="comparison-meta">
        <span className="fault-meta-label">Cycles to heal</span>
        <span className="fault-meta-value mono" style={{ color: 'var(--accent-cyan)' }}>
          {cyclesTaken} cycles ({(cyclesTaken * 0.5).toFixed(1)}s)
        </span>
      </div>

      <div className="comparison-divider" />

      {/* Column headers */}
      <div className="comparison-header-row">
        <span className="comparison-col-param">Parameter</span>
        <span className="comparison-col-val" style={{ color: 'var(--accent-red)' }}>Fault</span>
        <span className="comparison-col-val" style={{ color: 'var(--accent-green)' }}>Healed</span>
        <span className="comparison-col-delta">Change</span>
      </div>

      {/* Data rows */}
      {ROWS.map(row => {
        const b = before[row.key]
        const a = after[row.key]
        const maxAbs = Math.max(Math.abs(b), 2.0)
        return (
          <div key={row.key} className="comparison-row">
            <div className="comparison-col-param">
              <div className="comparison-param-name">{row.label}</div>
              <div className="comparison-param-desc">{row.desc}</div>
            </div>
            <div className="comparison-col-val">
              <span className="mono" style={{ color: 'var(--accent-red)', fontSize: 11 }}>
                {formatPhysical(row.channel, b)}
              </span>
              <ValueBar value={b} max={maxAbs} />
            </div>
            <div className="comparison-col-val">
              <span className="mono" style={{ color: 'var(--accent-green)', fontSize: 11 }}>
                {formatPhysical(row.channel, a)}
              </span>
              <ValueBar value={a} max={maxAbs} />
            </div>
            <div className="comparison-col-delta">
              <DeltaArrow before={b} after={a} />
            </div>
          </div>
        )
      })}

      <div className="comparison-divider" />
      <div className="convergence-badge" style={{ marginTop: 6 }}>
        Self-healing complete
      </div>
    </div>
  )
}
