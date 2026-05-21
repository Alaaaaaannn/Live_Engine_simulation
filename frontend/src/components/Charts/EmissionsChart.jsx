import { useMemo } from 'react'
import { VisXYContainer, VisLine, VisAxis, VisCrosshair } from '@unovis/react'
import { useSimulationContext } from '../../context/SimulationContext'
import './Charts.css'

export default function EmissionsChart({ data: dataProp, height = 240, title = 'Emission Levels' }) {
  const ctx = useSimulationContext()
  const data = dataProp ?? ctx.state.emissionsHistory

  const x   = useMemo(() => (d) => d.cycle, [])
  const yCO = useMemo(() => (d) => d.co,    [])
  const yHC = useMemo(() => (d) => d.hc,    [])
  const yNO = useMemo(() => (d) => d.nox,   [])

  const tooltipTemplate = (d) =>
    `<div style="font-family:Inter,system-ui,sans-serif;font-size:12px">
      <b>Cycle ${d.cycle}</b><br/>
      <span style="color:#8b939d">CO</span>  <span style="color:#ef4444;font-family:monospace">${d.co?.toFixed(3) ?? '—'}</span><br/>
      <span style="color:#8b939d">HC</span>  <span style="color:#f59e0b;font-family:monospace">${d.hc?.toFixed(3) ?? '—'}</span><br/>
      <span style="color:#8b939d">NOx</span> <span style="color:#8b5cf6;font-family:monospace">${d.nox?.toFixed(3) ?? '—'}</span>
    </div>`

  return (
    <div className="chart-card panel">
      <div className="panel-title">{title}</div>
      {data.length === 0 ? (
        <div className="chart-empty">No data yet</div>
      ) : (
        <VisXYContainer data={data} height={height}>
          <VisLine x={x} y={yCO} color="#ef4444" lineWidth={1.5} curveType="monotoneX" />
          <VisLine x={x} y={yHC} color="#f59e0b" lineWidth={1.5} curveType="monotoneX" />
          <VisLine x={x} y={yNO} color="#8b5cf6" lineWidth={1.5} curveType="monotoneX" />
          <VisAxis type="x" label="Cycle" tickCount={6} />
          <VisAxis type="y" label="Std. value" tickCount={5} />
          <VisCrosshair template={tooltipTemplate} />
        </VisXYContainer>
      )}
      <div className="chart-legend">
        <span><span className="legend-dot" style={{ background: '#ef4444' }} /> CO</span>
        <span><span className="legend-dot" style={{ background: '#f59e0b' }} /> HC</span>
        <span><span className="legend-dot" style={{ background: '#8b5cf6' }} /> NOx</span>
      </div>
    </div>
  )
}
