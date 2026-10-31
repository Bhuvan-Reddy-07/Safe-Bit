/**
 * Safe-Bit Frontend API & WebSocket Client.
 * Communicates with FastAPI streaming backend.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const WS_BASE_URL = import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8000';

export async function scoreTransaction(payload) {
  const response = await fetch(`${API_BASE_URL}/api/v1/score_transaction`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.statusText}`);
  }
  return await response.json();
}

export async function fetchMetrics() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/metrics`);
    if (!response.ok) return null;
    return await response.json();
  } catch (err) {
    console.warn('Metrics polling error:', err);
    return null;
  }
}

export async function fetchSubgraph(accountId) {
  const response = await fetch(`${API_BASE_URL}/api/v1/subgraph/${accountId}`);
  if (!response.ok) {
    throw new Error(`Failed to load subgraph: ${response.statusText}`);
  }
  return await response.json();
}

export async function fetchModels() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/models`);
    if (!response.ok) return null;
    return await response.json();
  } catch (err) {
    console.warn('Fetch models error:', err);
    return null;
  }
}

export async function switchModel(modelName) {
  const response = await fetch(`${API_BASE_URL}/api/v1/models/switch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_name: modelName }),
  });
  if (!response.ok) {
    throw new Error(`Failed to switch model: ${response.statusText}`);
  }
  return await response.json();
}

export async function triggerAMLSim(typology, amount = 45000.0) {
  const response = await fetch(`${API_BASE_URL}/api/v1/simulate/amlsim`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ typology, amount }),
  });
  if (!response.ok) {
    throw new Error(`Failed to trigger AMLSim: ${response.statusText}`);
  }
  return await response.json();
}

export async function fetchFeedzaiBenchmark() {
  const response = await fetch(`${API_BASE_URL}/api/v1/benchmark/feedzai_drift`);
  if (!response.ok) {
    throw new Error(`Failed to load Feedzai benchmark: ${response.statusText}`);
  }
  return await response.json();
}

export async function fetchDatasetsStatus() {
  const response = await fetch(`${API_BASE_URL}/api/v1/datasets/status`);
  if (!response.ok) {
    throw new Error(`Failed to load datasets status: ${response.statusText}`);
  }
  return await response.json();
}

export async function fetchIBMPatterns() {
  const response = await fetch(`${API_BASE_URL}/api/v1/datasets/ibm_patterns`);
  if (!response.ok) {
    throw new Error(`Failed to load IBM patterns: ${response.statusText}`);
  }
  return await response.json();
}

export async function replayDataset(datasetName, count = 50) {
  const response = await fetch(`${API_BASE_URL}/api/v1/datasets/replay`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset_name: datasetName, count }),
  });
  if (!response.ok) {
    throw new Error(`Failed to replay dataset: ${response.statusText}`);
  }
  return await response.json();
}

export async function fetchCrossDatasetBenchmark() {
  const response = await fetch(`${API_BASE_URL}/api/v1/datasets/benchmark_all`);
  if (!response.ok) {
    throw new Error(`Failed to load benchmark: ${response.statusText}`);
  }
  return await response.json();
}

export const fetchDatasetsBenchmark = fetchCrossDatasetBenchmark;

export async function fetchHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/health`);
    if (!response.ok) return null;
    return await response.json();
  } catch (err) {
    console.warn('Health check error:', err);
    return null;
  }
}

export async function fetchTransactionHistory(params = {}) {
  const query = new URLSearchParams();
  if (params.limit !== undefined) query.set('limit', params.limit);
  if (params.offset !== undefined) query.set('offset', params.offset);
  if (params.search) query.set('search', params.search);
  if (params.decision && params.decision !== 'ALL') query.set('decision', params.decision);
  if (params.typology && params.typology !== 'ALL') query.set('typology', params.typology);
  if (params.sort_by) query.set('sort_by', params.sort_by);

  const url = `${API_BASE_URL}/api/v1/transactions/history?${query.toString()}`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load transaction history: ${response.statusText}`);
  }
  return await response.json();
}

export async function fetchTransactionDetail(txId) {
  const response = await fetch(`${API_BASE_URL}/api/v1/transactions/${encodeURIComponent(txId)}`);
  if (!response.ok) {
    throw new Error(`Failed to load transaction details: ${response.statusText}`);
  }
  return await response.json();
}

export function createStreamWebSocket(onTransaction, onError) {
  let ws = null;
  let isClosedManually = false;

  function connect() {
    try {
      ws = new WebSocket(`${WS_BASE_URL}/api/v1/ws/stream`);

      ws.onopen = () => {
        console.log('[Safe-Bit Stream] WebSocket Connected');
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          onTransaction(data);
        } catch (e) {
          console.error('[Safe-Bit Stream] Parse error:', e);
        }
      };

      ws.onerror = (err) => {
        console.warn('[Safe-Bit Stream] Connection error', err);
        if (onError) onError(err);
      };

      ws.onclose = () => {
        if (!isClosedManually) {
          console.log('[Safe-Bit Stream] Reconnecting in 2s...');
          setTimeout(connect, 2000);
        }
      };
    } catch (e) {
      if (onError) onError(e);
    }
  }

  connect();

  return () => {
    isClosedManually = true;
    if (ws) ws.close();
  };
}
