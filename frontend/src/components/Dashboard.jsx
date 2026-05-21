import { useEffect, useRef } from 'react'
import { gsap } from 'gsap'
import SimulationControls from './Controls/SimulationControls'
import TweakablesPanel    from './Controls/TweakablesPanel'
import LambdaChart        from './Charts/LambdaChart'
import EmissionsChart     from './Charts/EmissionsChart'
import EngineViewer       from './Engine3D/EngineViewer'
import FaultStatusPanel      from './Panels/FaultStatusPanel'
import DigitalTwinLog        from './Panels/DigitalTwinLog'
import ShapPanel             from './Panels/ShapPanel'
import HealingComparisonPanel from './Panels/HealingComparisonPanel'
import HealthScorePanel      from './Panels/HealthScorePanel'
import ImpactCounterPanel    from './Panels/ImpactCounterPanel'
import StabilityPanel        from './Panels/StabilityPanel'
import WithoutAIPanel        from './Panels/WithoutAIPanel'
import './Dashboard.css'

export default function Dashboard() {
  const colLeft   = useRef(null)
  const colCenter = useRef(null)
  const colRight  = useRef(null)

  useEffect(() => {
    const tl = gsap.timeline({ delay: 0.2 })
    tl.from(colLeft.current,   { x: -20, duration: 0.5, ease: 'power3.out', clearProps: 'transform' })
      .from(colCenter.current, { y:  10, duration: 0.4, ease: 'power2.out', clearProps: 'transform' }, '-=0.2')
      .from(colRight.current,  { x:  20, duration: 0.5, ease: 'power3.out', clearProps: 'transform' }, '-=0.3')
  }, [])

  return (
    <div className="dashboard">
      {/* LEFT — controls */}
      <aside ref={colLeft} className="dashboard-left">
        <SimulationControls />
        <TweakablesPanel />
      </aside>

      {/* CENTER — 3D engine + charts + below-chart panels */}
      <main ref={colCenter} className="dashboard-center">
        <EngineViewer />
        <LambdaChart />
        <EmissionsChart />
        {/* Below-chart panels: side by side */}
        <div className="center-bottom-row">
          <ImpactCounterPanel />
          <ShapPanel />
        </div>
      </main>

      {/* RIGHT — always-visible status column */}
      <aside ref={colRight} className="dashboard-right">
        <HealthScorePanel />
        <FaultStatusPanel />
        <StabilityPanel />
        <WithoutAIPanel />
        <DigitalTwinLog />
        <HealingComparisonPanel />
      </aside>
    </div>
  )
}
