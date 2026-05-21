import { useSimulationContext } from '../../context/SimulationContext'
import './Panels.css'

const ROWS = [
  { key: 'coSavedG',           label: 'CO avoided',           fmt: v => v.toFixed(2),  unit: ' g',  color: 'var(--accent-green)', desc: 'Carbon monoxide prevented' },
  { key: 'hcSavedMg',          label: 'HC avoided',           fmt: v => Math.round(v), unit: ' mg', color: 'var(--accent-cyan)',  desc: 'Unburnt hydrocarbons' },
  { key: 'noxSavedMg',         label: 'NOx avoided',          fmt: v => Math.round(v), unit: ' mg', color: 'var(--accent-amber)', desc: 'Nitrogen oxides' },
  { key: 'fuelSavedMl',        label: 'Fuel waste prevented', fmt: v => v.toFixed(1),  unit: ' ml', color: 'var(--accent-green)', desc: 'Wasted fuel (rich faults)' },
  { key: 'catalystProtectedS', label: 'Catalyst protected',   fmt: v => Math.round(v), unit: ' s',  color: 'var(--accent-cyan)',  desc: 'From overtemp exposure' },
]

export default function ImpactCounterPanel() {
  const { state } = useSimulationContext()
  const { impactStats = {}, faultSnapshot } = state

  if (!impactStats) return null

  return (
    <div className="panel impact-panel">
      <div className="panel-title">Environmental impact</div>
      <div className="impact-subtitle">Damage prevented by real-time correction</div>

      {!faultSnapshot ? (
        <div className="dt-empty mono" style={{ padding: '8px 0' }}>
          Inject a fault to track impact
        </div>
      ) : (
        <>
          <div className="impact-grid">
            {ROWS.map(row => {
              const val = impactStats[row.key] ?? 0
              return (
                <div key={row.key} className="impact-row">
                  <div className="impact-info">
                    <span className="impact-label">{row.label}</span>
                    <span className="impact-desc">{row.desc}</span>
                    <span className="impact-value" style={{ color: row.color }}>
                      {row.fmt(val)}<span className="impact-unit">{row.unit}</span>
                    </span>
                  </div>
                </div>
              )
            })}
          </div>
          <div className="impact-note">vs. ~45s undetected by conventional OBD-II</div>
        </>
      )}
    </div>
  )
}
