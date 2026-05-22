import { useMemo } from 'react'
import {
  VisXYContainer, VisLine, VisArea, VisAxis, VisCrosshair, VisTooltip
} from '@unovis/react'
import { useSimulationContext } from '../../context/SimulationContext'
import { toPhysical, CALIBRATION } from '../../utils/units'
import './Charts.css'

export default function LambdaChart({ data: dataProp, height = 240, title = 'Lambda Convergence' }) {
  const ctx = useSimulationContext()
  const data = dataProp ?? ctx.state.lambdaHistory

  // Unovis accessors — project the stored z-scores into physical λ on the fly.
  const x    = useMemo(() => (d) => d.cycle,                            [])
  const yAct = useMemo(() => (d) => toPhysical('lambda', d.actual),     [])
  const yPrd = useMemo(() => (d) => toPhysical('lambda', d.predicted),  [])

  // Stoichiometric control band: ±0.02 around λ=1.0 (tight closed-loop target).
  const bandData = useMemo(() => {
    if (data.length === 0) return [{ cycle: 0 }, { cycle: 1 }]
    return [{ cycle: data[0].cycle }, { cycle: data[data.length - 1].cycle }]
  }, [data])
  const bandY0 = () => CALIBRATION.lambda.mean - 0.02
  const bandY1 = () => CALIBRATION.lambda.mean + 0.02

  const tooltipTemplate = (d) => {
    const a = toPhysical('lambda', d.actual)
    const p = toPhysical('lambda', d.predicted)
    return `<div style="font-family:Inter,system-ui,sans-serif;font-size:12px">
      <b>Cycle ${d.cycle}</b><br/>
      <span style="color:#8b939d">λ actual</span> <span style="color:#3b82f6;font-family:monospace">${a?.toFixed(3) ?? '—'}</span> <span style="color:#5b6470;font-size:10px">(${d.actual?.toFixed(2) ?? '—'}σ)</span><br/>
      <span style="color:#8b939d">λ predicted</span> <span style="color:#8b5cf6;font-family:monospace">${p?.toFixed(3) ?? '—'}</span> <span style="color:#5b6470;font-size:10px">(${d.predicted?.toFixed(2) ?? '—'}σ)</span>
    </div>`
  }

  return (
    <div className="chart-card panel">
      <div className="panel-title">{title}</div>
      {data.length === 0 ? (
        <div className="chart-empty">No data yet</div>
      ) : (
        <VisXYContainer data={data} height={height}>
          {/* Stoichiometric band */}
          <VisArea
            data={bandData}
            x={x}
            y={[bandY0, bandY1]}
            color="#22c55e"
            opacity={0.08}
          />
          {/* Actual Lambda */}
          <VisLine
            x={x} y={yAct}
            color="#3b82f6"
            lineWidth={1.6}
            curveType="monotoneX"
          />
          {/* Twin-predicted Lambda */}
          <VisLine
            x={x} y={yPrd}
            color="#8b5cf6"
            lineWidth={1.4}
            lineDash={[4, 3]}
            curveType="monotoneX"
          />
          <VisAxis type="x" label="Cycle" tickCount={6} />
          <VisAxis type="y" label="λ (air-fuel ratio)" tickCount={5} />
          <VisCrosshair template={tooltipTemplate} />
        </VisXYContainer>
      )}
      <div className="chart-legend">
        <span><span className="legend-dot" style={{ background: '#3b82f6' }} /> Actual</span>
        <span><span className="legend-dash" style={{ borderColor: '#8b5cf6' }} /> Twin predicted</span>
        <span><span className="legend-band" /> Stoich. band</span>
      </div>
    </div>
  )
}
