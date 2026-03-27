const BASE_URL = 'http://localhost:8000';

export async function predictImage(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${BASE_URL}/api/predict`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorBody = await response.text();
    throw new Error(`Prediction failed (${response.status}): ${errorBody}`);
  }

  return response.json();
}

export async function getHealth() {
  const response = await fetch(`${BASE_URL}/api/health`);

  if (!response.ok) {
    throw new Error(`Health check failed (${response.status})`);
  }

  return response.json();
}

export async function getMetrics() {
  const response = await fetch(`${BASE_URL}/api/metrics`);

  if (!response.ok) {
    throw new Error(`Metrics fetch failed (${response.status})`);
  }

  return response.json();
}
