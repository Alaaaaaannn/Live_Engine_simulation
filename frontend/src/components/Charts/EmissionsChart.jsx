import { useMemo } from 'react'
import { VisXYContainer, VisLine, VisAxis, VisCrosshair } from '@unovis/react'
import { useSimulationContext } from '../../context/SimulationContext'
import './Charts.css'

export default function EmissionsChart() {
  const { state } = useSimulationContext()
  const data = state.emissionsHistory

  const x   = useMemo(() => (d) => d.cycle, [])
  const yCO = useMemo(() => (d) => d.co,    [])
  const yHC = useMemo(() => (d) => d.hc,    [])
  const yNO = useMemo(() => (d) => d.nox,   [])

  const tooltipTemplate = (d) =>
    `<div style="font-family:monospace;font-size:11px">
      <b>Cycle ${d.cycle}</b><br/>
      CO:  <span style="color:#ff3355">${d.co?.toFixed(3) ?? '—'}</span><br/>
      HC:  <span style="color:#ffaa00">${d.hc?.toFixed(3) ?? '—'}</span><br/>
      NOx: <span style="color:#7b68ee">${d.nox?.toFixed(3) ?? '—'}</span>
    </div>`

  return (
    <div className="chart-card panel panel-scan">
      <div className="panel-title">Emission Levels</div>
      {data.length === 0 ? (
        <div className="chart-empty">Start simulation to see live data</div>
      ) : (
        <VisXYContainer data={data} height={180}>
          <VisLine x={x} y={yCO} color="#ff3355" lineWidth={1.8} curveType="monotoneX" />
          <VisLine x={x} y={yHC} color="#ffaa00" lineWidth={1.8} curveType="monotoneX" />
          <VisLine x={x} y={yNO} color="#7b68ee" lineWidth={1.8} curveType="monotoneX" />
          <VisAxis type="x" label="Cycle" tickCount={6} />
          <VisAxis type="y" label="Std. Value" tickCount={5} />
          <VisCrosshair template={tooltipTemplate} />
        </VisXYContainer>
      )}
      <div className="chart-legend">
        <span><span className="legend-dot" style={{ background: '#ff3355' }} /> CO</span>
        <span><span className="legend-dot" style={{ background: '#ffaa00' }} /> HC</span>
        <span><span className="legend-dot" style={{ background: '#7b68ee' }} /> NOx</span>
      </div>
    </div>
  )
}
