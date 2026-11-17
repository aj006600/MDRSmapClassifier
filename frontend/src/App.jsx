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

const isActive = (job) => job.status === 'queued' || job.status === 'running'

// 回傳合併後的工作清單:updates 裡的工作取代同一個 id 的舊資料。
// 清單依 id 排列,id 的順序就是發起與執行的順序
function mergeJobs(jobs, updates) {
  const byId = new Map(jobs.map((job) => [job.id, job]))
  for (const job of updates) byId.set(job.id, job)
  return [...byId.values()].sort((a, b) => a.id - b.id)
}

export default function App() {
  const [stockId, setStockId] = useState('2330')
  const [horizon, setHorizon] = useState(10)
  const [jobs, setJobs] = useState([])
  const [polls, setPolls] = useState(0)
  const [offline, setOffline] = useState(false)
  const [error, setError] = useState(null)
  const [selected, setSelected] = useState(() => parseHash(window.location.hash))
  // 每次有工作成功就加一,讓清單與結果重新讀取
  const [resultsVersion, setResultsVersion] = useState(0)

  const summaries = useApi('/api/results', resultsVersion)
  const result = useApi(
    selected && `/api/results/${encodeURIComponent(selected.stockId)}/${selected.horizon}`,
    resultsVersion,
  )

  async function start(event) {
    event.preventDefault()
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
      // 這支股票這個 Horizon 已經在排隊或執行時,回傳的是原本那個工作
      const job = await response.json()
      setJobs((jobs) => mergeJobs(jobs, [job]))
    } catch {
      setError('連不上後端')
    }
  }

  // 選取這個工作的結果,並重新讀取股票清單與結果,讓剛完成的結果出現。
  // hash 沒有變時不會觸發 hashchange,所以這裡直接選取
  function showResult(job) {
    window.location.hash = resultHash(job.stock_id, job.horizon)
    setSelected({ stockId: job.stock_id, horizon: job.horizon })
    setResultsVersion((version) => version + 1)
  }

  useEffect(() => {
    const onHashChange = () => setSelected(parseHash(window.location.hash))
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  // 這個 effect 在網頁打開時立刻向後端查一次排隊中與執行中的工作,之後定期再查:
  // 重新整理或重開網頁時,畫面才能恢復這些工作。
  // 離開進行中清單的工作要個別再查一次,才拿得到它最後的狀態。
  // polls 讓 effect 在每一輪之後重跑:後端暫時連不上(例如重新啟動)時工作清單不會變,
  // 少了它就不會再重試
  useEffect(() => {
    const watched = jobs.filter(isActive)
    const timer = setTimeout(async () => {
      try {
        const active = await fetchJson('/api/jobs')
        const activeIds = new Set(active.map((job) => job.id))
        const ended = await Promise.all(
          watched
            .filter((job) => !activeIds.has(job.id))
            .map((job) => fetchJson(`/api/jobs/${job.id}`)),
        )
        setJobs((jobs) => mergeJobs(jobs, [...active, ...ended]))
        setOffline(false)
        const succeeded = ended.filter((job) => job.status === 'succeeded')
        if (succeeded.length > 0) showResult(succeeded[succeeded.length - 1])
      } catch {
        setOffline(true)
      } finally {
        setPolls((polls) => polls + 1)
      }
    }, polls === 0 ? 0 : POLL_INTERVAL_MS)
    return () => clearTimeout(timer)
  }, [jobs, polls])

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
        <button type="submit">執行</button>
      </form>

      {error && <p className="error">{error}</p>}
      {offline && <p className="error">連不上後端,稍後會自動重試</p>}
      {jobs.length > 0 && <JobList jobs={jobs} />}
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

function JobList({ jobs }) {
  const active = jobs.filter(isActive)
  return (
    <section>
      <h2>工作</h2>
      <ul>
        {jobs.map((job) => (
          <li key={job.id}>
            <JobStatus job={job} ahead={active.filter((other) => other.id < job.id).length} />
          </li>
        ))}
      </ul>
    </section>
  )
}

function JobStatus({ job, ahead }) {
  return (
    <>
      {job.stock_id} · Horizon {job.horizon}:<strong>{STATUS_TEXT[job.status]}</strong>
      {job.status === 'running' && job.stage && `(${STAGE_TEXT[job.stage]})`}
      {job.status === 'queued' && ahead > 0 && `(前面還有 ${ahead} 個)`}
      {job.status === 'running' && job.stage === 'search' && (
        <div>Search 通常需要幾分鐘,請稍候。</div>
      )}
      {job.status === 'failed' && <div className="error">失敗原因:{job.error}</div>}
    </>
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
