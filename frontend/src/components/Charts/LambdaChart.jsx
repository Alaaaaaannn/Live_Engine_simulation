import { useMemo } from 'react'
import {
  VisXYContainer, VisLine, VisArea, VisAxis, VisCrosshair, VisTooltip
} from '@unovis/react'
import { useSimulationContext } from '../../context/SimulationContext'
import './Charts.css'

export default function LambdaChart() {
  const { state } = useSimulationContext()
  const data = state.lambdaHistory

  // Unovis accessor functions
  const x    = useMemo(() => (d) => d.cycle,     [])
  const yAct = useMemo(() => (d) => d.actual,    [])
  const yPrd = useMemo(() => (d) => d.predicted, [])

  // Stoichiometric band as area between -0.05 and +0.05
  const bandData = useMemo(() => {
    if (data.length === 0) return [{ cycle: 0 }, { cycle: 1 }]
    return [{ cycle: data[0].cycle }, { cycle: data[data.length - 1].cycle }]
  }, [data])
  const bandY0 = () => -0.05
  const bandY1 = () =>  0.05

  const tooltipTemplate = (d) =>
    `<div style="font-family:monospace;font-size:11px">
      <b>Cycle ${d.cycle}</b><br/>
      λ actual: <span style="color:#00d4ff">${d.actual?.toFixed(4) ?? '—'}</span><br/>
      λ pred:   <span style="color:#ff3355">${d.predicted?.toFixed(4) ?? '—'}</span>
    </div>`

  return (
    <div className="chart-card panel panel-scan">
      <div className="panel-title">Lambda Convergence</div>
      {data.length === 0 ? (
        <div className="chart-empty">Start simulation to see live data</div>
      ) : (
        <VisXYContainer data={data} height={200}>
          {/* Stoichiometric band */}
          <VisArea
            data={bandData}
            x={x}
            y={[bandY0, bandY1]}
            color="#00ff88"
            opacity={0.12}
          />
          {/* Actual Lambda */}
          <VisLine
            x={x} y={yAct}
            color="#00d4ff"
            lineWidth={2}
            curveType="monotoneX"
          />
          {/* Twin-predicted Lambda */}
          <VisLine
            x={x} y={yPrd}
            color="#ff3355"
            lineWidth={1.5}
            lineDash={[4, 3]}
            curveType="monotoneX"
          />
          <VisAxis type="x" label="Cycle" tickCount={6} />
          <VisAxis type="y" label="λ (σ)" tickCount={5} />
          <VisCrosshair template={tooltipTemplate} />
        </VisXYContainer>
      )}
      <div className="chart-legend">
        <span><span className="legend-dot" style={{ background: '#00d4ff' }} /> Actual</span>
        <span><span className="legend-dash" style={{ borderColor: '#ff3355' }} /> Twin Predicted</span>
        <span><span className="legend-band" /> Stoich. Band</span>
      </div>
    </div>
  )
}
