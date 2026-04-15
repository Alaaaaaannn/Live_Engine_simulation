import { useEffect, useRef } from 'react'
import { gsap } from 'gsap'
import { SimulationProvider } from './context/SimulationContext'
import Header    from './components/Layout/Header'
import Dashboard from './components/Dashboard'
import './styles/App.css'

function AppContent() {
  const rootRef = useRef(null)

  // Boot sequence: scan line reveal
  useEffect(() => {
    const el = rootRef.current
    if (!el) return
    gsap.from(el, { duration: 0.3, ease: 'power2.out', clearProps: 'all' })
  }, [])

  return (
    <div ref={rootRef} className="app-root">
      <Header />
      <div className="app-body">
        <Dashboard />
      </div>
    </div>
  )
}

export default function App() {
  return (
    <SimulationProvider>
      <AppContent />
    </SimulationProvider>
  )
}
