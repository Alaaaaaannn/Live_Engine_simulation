import { useMemo } from 'react'
import { VisXYContainer, VisLine, VisAxis, VisCrosshair } from '@unovis/react'
import { useSimulationContext } from '../../context/SimulationContext'
import { formatPhysical } from '../../utils/units'
import './Charts.css'

export default function EmissionsChart({ data: dataProp, height = 240, title = 'Emission Levels' }) {
  const ctx = useSimulationContext()
  const data = dataProp ?? ctx.state.emissionsHistory

  // CO, HC and NOx have incompatible physical units (% vol vs ppm), so the
  // shared y-axis stays in σ for relative comparison; the tooltip surfaces
  // the physical reading per series.
  const x   = useMemo(() => (d) => d.cycle, [])
  const yCO = useMemo(() => (d) => d.co,    [])
  const yHC = useMemo(() => (d) => d.hc,    [])
  const yNO = useMemo(() => (d) => d.nox,   [])

  const tooltipTemplate = (d) =>
    `<div style="font-family:Inter,system-ui,sans-serif;font-size:12px">
      <b>Cycle ${d.cycle}</b><br/>
      <span style="color:#8b939d">CO</span>  <span style="color:#ef4444;font-family:monospace">${formatPhysical('co', d.co)}</span> <span style="color:#5b6470;font-size:10px">(${d.co?.toFixed(2) ?? '—'}σ)</span><br/>
      <span style="color:#8b939d">HC</span>  <span style="color:#f59e0b;font-family:monospace">${formatPhysical('hc', d.hc)}</span> <span style="color:#5b6470;font-size:10px">(${d.hc?.toFixed(2) ?? '—'}σ)</span><br/>
      <span style="color:#8b939d">NOx</span> <span style="color:#8b5cf6;font-family:monospace">${formatPhysical('nox', d.nox)}</span> <span style="color:#5b6470;font-size:10px">(${d.nox?.toFixed(2) ?? '—'}σ)</span>
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
          <VisAxis type="y" label="Deviation (σ)" tickCount={5} />
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
