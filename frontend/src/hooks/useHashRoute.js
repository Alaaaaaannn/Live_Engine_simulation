import { useEffect, useState, useCallback } from 'react'

function readHash() {
  const h = window.location.hash || '#/'
  return h.startsWith('#') ? h.slice(1) : h
}

export function useHashRoute() {
  const [path, setPath] = useState(readHash())

  useEffect(() => {
    const onChange = () => setPath(readHash())
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])

  const navigate = useCallback((to) => {
    const next = to.startsWith('/') ? to : `/${to}`
    if (window.location.hash !== `#${next}`) {
      window.location.hash = next
    }
  }, [])

  return { path, navigate }
}
