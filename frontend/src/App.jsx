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

// 選到的結果記在網址的 hash(例如 #2330/10):重新整理後能從 API 載回同一個結果,
// 上一頁也能回到前一個看過的結果
function parseHash(hash) {
  const match = /^#([^/]+)\/(\d+)$/.exec(hash)
  return match ? { stockId: decodeURIComponent(match[1]), horizon: Number(match[2]) } : null
}

function resultHash(stockId, horizon) {
  return `#${encodeURIComponent(stockId)}/${horizon}`
}

function percent(ratio) {
  return ratio === null ? '—' : `${(ratio * 100).toFixed(1)}%`
}

async function fetchJson(path) {
  let response
  try {
    response = await fetch(path)
  } catch {
    throw new Error('連不上後端')
  }
  const body = await response.json().catch(() => null)
  if (!response.ok) {
    throw new Error(typeof body?.detail === 'string' ? body.detail : `HTTP ${response.status}`)
  }
  return body
}

/** 讀取 path 的 JSON;path 是 null 時不讀。version 改變時重新讀取同一個 path。 */
function useApi(path, version) {
  const [state, setState] = useState({ data: null, error: null })
  useEffect(() => {
    if (path === null) {
      setState({ data: null, error: null })
      return
    }
    // 換了 path 之後,舊 path 晚到的回應不能蓋掉新的
    let current = true
    fetchJson(path).then(
      (data) => current && setState({ data, error: null }),
      (error) => current && setState({ data: null, error: error.message }),
    )
    return () => {
      current = false
    }
  }, [path, version])
  return state
}

export default function App() {
  const [stockId, setStockId] = useState('2330')
  const [horizon, setHorizon] = useState(10)
  const [job, setJob] = useState(null)
  const [error, setError] = useState(null)
  const [selected, setSelected] = useState(() => parseHash(window.location.hash))
  // 每次有工作成功就加一,讓清單與結果重新讀取
  const [resultsVersion, setResultsVersion] = useState(0)

  const summaries = useApi('/api/results', resultsVersion)
  const result = useApi(
    selected && `/api/results/${encodeURIComponent(selected.stockId)}/${selected.horizon}`,
    resultsVersion,
  )

  const inProgress = job?.status === 'queued' || job?.status === 'running'

  async function start(event) {
    event.preventDefault()
    setJob(null)
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

  useEffect(() => {
    const onHashChange = () => setSelected(parseHash(window.location.hash))
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

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
    // hash 沒有變時不會觸發 hashchange,所以這裡直接選取
    window.location.hash = resultHash(job.stock_id, job.horizon)
    setSelected({ stockId: job.stock_id, horizon: job.horizon })
    setResultsVersion((version) => version + 1)
  }, [job?.status, job?.stock_id, job?.horizon])

  return (
    <main>
      <h1>MDR S-map Classifier</h1>

      <form onSubmit={start}>
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

      {error && <p className="error">{error}</p>}
      {job && <JobStatus job={job} />}
      {result.error && <p className="error">{result.error}</p>}
      {result.data && <ResultView result={result.data} />}

      <StockList summaries={summaries} selected={selected} />

      <footer>
        <p>
          資料來源:<a href="https://finmindtrade.com/">FinMind</a>
          。本服務的 Forecast 與 Backtest 只是模型的計算結果,不構成投資建議。
        </p>
      </footer>
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
      {job.status === 'failed' && <p className="error">失敗原因:{job.error}</p>}
    </section>
  )
}

function ResultView({ result }) {
  const { horizon } = result
  const backtest = result.backtest
  const misses = backtest.days.filter((day) => day.direction !== day.actual).length
  return (
    <section>
      <h2>
        {result.stock_id} · Horizon {horizon}
      </h2>
      <p>
        資料用到 <strong>{result.data_date}</strong> 這個交易日。
      </p>
      <p className="note">
        某個交易日的 Direction 為 1,表示它之後 {horizon} 個交易日的平均收盤價不低於當天的收盤價;
        低於則為 0。它比的是接下來一段期間的平均價,不是「明天漲或跌」。
      </p>

      <h3>Forecast</h3>
      <p className="note">最近 {horizon} 個交易日的實際 Direction 還沒揭曉,每一天各有一筆 Forecast。</p>
      <table>
        <thead>
          <tr>
            <th>交易日</th>
            <th>Direction</th>
          </tr>
        </thead>
        <tbody>
          {result.forecast.map((day) => (
            <tr key={day.date}>
              <td>{day.date}</td>
              <td>{day.direction}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3>Backtest</h3>
      <p>
        準確率 <strong>{percent(backtest.accuracy)}</strong>,計分 {backtest.scored_days} 天
      </p>
      <p>
        同期間 Direction 為 1 的天數比例 <strong>{percent(backtest.direction_one_ratio)}</strong>
      </p>
      <p className="note">
        每天都猜 1 的準確率就是 Direction 為 1 的天數比例;準確率要高於這個比例,模型才比每天都猜 1 好。
      </p>
      <details>
        <summary>
          逐日明細({backtest.scored_days} 天,猜錯 {misses} 天)
        </summary>
        <table>
          <thead>
            <tr>
              <th>交易日</th>
              <th>預測</th>
              <th>實際</th>
              <th>對錯</th>
            </tr>
          </thead>
          <tbody>
            {backtest.days.map((day) => {
              const hit = day.direction === day.actual
              return (
                <tr key={day.date} className={hit ? undefined : 'miss'}>
                  <td>{day.date}</td>
                  <td>{day.direction}</td>
                  <td>{day.actual}</td>
                  <td>{hit ? '對' : '錯'}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </details>
    </section>
  )
}

function StockList({ summaries, selected }) {
  return (
    <section>
      <h2>執行過的股票</h2>
      {summaries.error && <p className="error">{summaries.error}</p>}
      {summaries.data?.length === 0 && <p className="note">還沒有成功的執行。</p>}
      <ul className="stock-list">
        {summaries.data?.map((summary) => {
          const isSelected =
            selected?.stockId === summary.stock_id && selected?.horizon === summary.horizon
          return (
            <li key={`${summary.stock_id}/${summary.horizon}`}>
              <a
                href={resultHash(summary.stock_id, summary.horizon)}
                aria-current={isSelected ? 'true' : undefined}
              >
                <strong>
                  {summary.stock_id} · Horizon {summary.horizon}
                </strong>
                <span>資料到 {summary.data_date}</span>
                <span>當天的 Forecast {summary.latest_forecast.direction}</span>
                <span>Backtest 準確率 {percent(summary.backtest.accuracy)}</span>
              </a>
            </li>
          )
        })}
      </ul>
    </section>
  )
}
