import { useEffect, useState } from 'react'

const HORIZONS = [5, 10, 20]
const STATUS_TEXT = {
  queued: '排隊中',
  running: '執行中',
  succeeded: '完成',
  failed: '失敗',
}
const STAGE_TEXT = { fetch: '抓資料', search: 'Search', predict: '逐日預測' }
const POLL_INTERVAL_MS = 2000

export default function App() {
  const [stockId, setStockId] = useState('2330')
  const [horizon, setHorizon] = useState(10)
  const [job, setJob] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const inProgress = job?.status === 'queued' || job?.status === 'running'

  async function start(event) {
    event.preventDefault()
    setJob(null)
    setResult(null)
    setError(null)
    try {
      const response = await fetch('/api/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stock_id: stockId.trim(), horizon }),
      })
      if (!response.ok) {
        setError(`無法發起執行(HTTP ${response.status})`)
        return
      }
      setJob(await response.json())
    } catch {
      setError('連不上後端')
    }
  }

  // 工作還沒結束時,每隔一段時間重新查一次它的狀態
  useEffect(() => {
    if (!inProgress) return
    const timer = setTimeout(async () => {
      try {
        const response = await fetch(`/api/jobs/${job.id}`)
        setJob(await response.json())
      } catch {
        setError('連不上後端')
      }
    }, POLL_INTERVAL_MS)
    return () => clearTimeout(timer)
  }, [job, inProgress])

  useEffect(() => {
    if (job?.status !== 'succeeded') return
    fetch(`/api/results/${job.stock_id}/${job.horizon}`)
      .then((response) => response.json())
      .then(setResult)
      .catch(() => setError('連不上後端'))
  }, [job?.status, job?.stock_id, job?.horizon])

  return (
    <main style={{ fontFamily: 'sans-serif', padding: '1rem', maxWidth: '40rem' }}>
      <h1>MDR S-map Classifier</h1>

      <form onSubmit={start} style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
        <label>
          股票代號{' '}
          <input
            value={stockId}
            onChange={(event) => setStockId(event.target.value)}
            size={8}
            required
          />
        </label>
        <label>
          Horizon{' '}
          <select value={horizon} onChange={(event) => setHorizon(Number(event.target.value))}>
            {HORIZONS.map((days) => (
              <option key={days} value={days}>
                {days} 個交易日
              </option>
            ))}
          </select>
        </label>
        <button type="submit" disabled={inProgress}>
          執行
        </button>
      </form>

      {error && <p style={{ color: 'crimson' }}>{error}</p>}
      {job && <JobStatus job={job} />}
      {result && <ResultView result={result} />}
    </main>
  )
}

function JobStatus({ job }) {
  return (
    <section>
      <p>
        {job.stock_id} · Horizon {job.horizon}:<strong>{STATUS_TEXT[job.status]}</strong>
        {job.status === 'running' && job.stage && `(${STAGE_TEXT[job.stage]})`}
      </p>
      {job.status === 'running' && job.stage === 'search' && (
        <p>Search 通常需要幾分鐘,請稍候。</p>
      )}
      {job.status === 'failed' && <p style={{ color: 'crimson' }}>失敗原因:{job.error}</p>}
    </section>
  )
}

function ResultView({ result }) {
  const { accuracy, scored_days: scoredDays } = result.backtest
  return (
    <section>
      <h2>Backtest</h2>
      <p>
        準確率 <strong>{accuracy === null ? '—' : `${(accuracy * 100).toFixed(1)}%`}</strong>
        ,計分 {scoredDays} 天
      </p>

      <h2>Forecast</h2>
      <table style={{ borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th style={{ textAlign: 'left', paddingRight: '2rem' }}>交易日</th>
            <th style={{ textAlign: 'left' }}>Direction</th>
          </tr>
        </thead>
        <tbody>
          {result.forecast.map((day) => (
            <tr key={day.date}>
              <td style={{ paddingRight: '2rem' }}>{day.date}</td>
              <td>{day.direction}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  )
}
