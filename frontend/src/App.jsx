import { useEffect, useState } from 'react'

export default function App() {
  const [message, setMessage] = useState('載入中...')

  useEffect(() => {
    fetch('/api/message')
      .then((res) => res.json())
      .then((data) => setMessage(data.message))
      .catch(() => setMessage('連不上後端'))
  }, [])

  return (
    <main style={{ fontFamily: 'sans-serif', padding: '2rem' }}>
      <h1>MDR S-map Classifier</h1>
      <p>
        後端回應:<strong>{message}</strong>
      </p>
    </main>
  )
}
