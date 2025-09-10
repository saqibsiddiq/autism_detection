import React from 'react'
import axios from 'axios'

export default function Overview() {
  const [status, setStatus] = React.useState('Checking...')
  React.useEffect(() => {
    axios.get('/api/health').then(() => setStatus('Backend connected')).catch(() => setStatus('Backend not reachable'))
  }, [])

  return (
    <div>
      <h3>Assessment Overview</h3>
      <p>{status}</p>
      <ul>
        <li>Face Recognition Test</li>
        <li>Social Attention Test</li>
        <li>Visual Pattern Test</li>
        <li>Motion Tracking Test</li>
      </ul>
    </div>
  )
}


