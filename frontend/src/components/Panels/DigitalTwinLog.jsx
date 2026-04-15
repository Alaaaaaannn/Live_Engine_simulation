import { useAutoAnimate } from '@formkit/auto-animate/react'
import { useSimulationContext } from '../../context/SimulationContext'
import './Panels.css'

function describeAction(entry) {
  const { fuelTrim, sparkAdv, faultName } = entry
  if (faultName === 'Normal' || faultName === 'normal') {
    return { action: 'Monitoring nominal operation', detail: null, color: '#00ff88' }
  }
  const parts = []
  if (Math.abs(fuelTrim) > 0.001) {
    const dir = fuelTrim > 0 ? 'increasing' : 'reducing'
    const what = faultName === 'Rich Mixture' ? 'fuel (rich correction)' :
                 faultName === 'Lean Mixture' ? 'fuel (lean correction)' : 'fuel trim'
    parts.push(`${dir.charAt(0).toUpperCase() + dir.slice(1)} ${what}`)
  }
  if (Math.abs(sparkAdv) > 0.001) {
    const dir = sparkAdv > 0 ? 'advancing' : 'retarding'
    parts.push(`${dir.charAt(0).toUpperCase() + dir.slice(1)} ignition angle`)
  }
  if (parts.length === 0) return { action: 'ECU holding current state', detail: null, color: '#888' }
  return {
    action: parts[0],
    detail: parts[1] ?? null,
    color: faultName === 'Rich Mixture' ? '#ffaa00'
         : faultName === 'Lean Mixture' ? '#ffaa00'
         : '#ff3355',
  }
}

export default function DigitalTwinLog() {
  const { state } = useSimulationContext()
  const [parent] = useAutoAnimate({ duration: 250 })

  return (
    <div className="panel dt-log-panel">
      <div className="panel-title">ECU Action Log</div>

      {state.twinLog.length === 0 ? (
        <div className="dt-empty mono">No cycles yet</div>
      ) : (
        <div className="dt-scroll" ref={parent}>
          {state.twinLog.map((entry, i) => {
            const desc = describeAction(entry)
            return (
              <div key={`${entry.cycle}-${i}`} className="dt-entry">
                <span className="dt-cycle mono">#{entry.cycle}</span>

                <div className="dt-details">
                  <span className="dt-action" style={{ color: desc.color }}>
                    {desc.action}
                  </span>
                  {desc.detail && (
                    <span className="dt-pred text-muted">{desc.detail}</span>
                  )}
                  <span className="dt-pred mono text-muted">
                    {Math.abs(entry.fuelTrim) > 0.001 && (
                      <>Fuel {entry.fuelTrim > 0 ? '+' : ''}{entry.fuelTrim.toFixed(3)}σ &nbsp;</>
                    )}
                    {Math.abs(entry.sparkAdv) > 0.001 && (
                      <>Spark {entry.sparkAdv > 0 ? '+' : ''}{entry.sparkAdv.toFixed(3)}σ &nbsp;</>
                    )}
                    &lambda;&rarr;{entry.lambdaPred.toFixed(3)}
                  </span>
                </div>

                <span className={`badge ${entry.approved ? 'badge-approved' : 'badge-rejected'}`}>
                  {entry.approved ? 'APR' : 'REJ'}
                </span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
