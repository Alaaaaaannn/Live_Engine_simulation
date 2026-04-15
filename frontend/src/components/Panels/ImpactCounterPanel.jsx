import { useSimulationContext } from '../../context/SimulationContext'
import './Panels.css'

const ROWS = [
  { key: 'coSavedG',           label: 'CO Avoided',              fmt: v => v.toFixed(2),  unit: ' g',   color: '#00ff88',  desc: 'Carbon monoxide prevented' },
  { key: 'hcSavedMg',          label: 'HC Avoided',              fmt: v => Math.round(v), unit: ' mg',  color: '#00d4ff',  desc: 'Unburnt hydrocarbons' },
  { key: 'noxSavedMg',         label: 'NOx Avoided',             fmt: v => Math.round(v), unit: ' mg',  color: '#ffaa00',  desc: 'Nitrogen oxides' },
  { key: 'fuelSavedMl',        label: 'Fuel Waste Prevented',    fmt: v => v.toFixed(1),  unit: ' ml',  color: '#00ff88',  desc: 'Wasted fuel (rich faults)' },
  { key: 'catalystProtectedS', label: 'Catalyst Protected',      fmt: v => Math.round(v), unit: ' s',   color: '#00d4ff',  desc: 'From overtemp exposure' },
]

export default function ImpactCounterPanel() {
  const { state } = useSimulationContext()
  const { impactStats = {}, faultSnapshot } = state

  if (!impactStats) return null

  return (
    <div className="panel impact-panel">
      <div className="panel-title" style={{ color: '#00d4ff' }}>AI Environmental Impact</div>
      <div className="impact-subtitle mono">DAMAGE PREVENTED BY REAL-TIME CORRECTION</div>

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
