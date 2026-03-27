import React, { useState, useEffect } from 'react';
import { getMetrics } from '../utils/api';
import './ClinicalMetrics.css';

export default function ClinicalMetrics() {
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getMetrics()
      .then((data) => {
        const display = [];
        if (data.classifier) {
          const fields = [
            ['accuracy', 'Accuracy', true],
            ['precision', 'Precision', true],
            ['recall', 'Recall', true],
            ['f1_score', 'F1-Score', false],
          ];
          for (const [key, label, isPercent] of fields) {
            if (data.classifier[key] != null) {
              display.push({
                label,
                value: isPercent
                  ? `${(data.classifier[key] * 100).toFixed(1)}%`
                  : data.classifier[key].toFixed(3),
              });
            }
          }
        }
        setMetrics(display.length > 0 ? display : null);
      })
      .catch(() => {
        setError('Metrics unavailable');
      });
  }, []);

  if (error) {
    return (
      <div className="clinical-metrics">
        <div className="clinical-metrics-header">
          <div className="clinical-metrics-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="20" x2="18" y2="10" />
              <line x1="12" y1="20" x2="12" y2="4" />
              <line x1="6" y1="20" x2="6" y2="14" />
            </svg>
          </div>
          <span className="clinical-metrics-title">Clinical Metrics</span>
        </div>
        <div className="clinical-metrics-empty">{error}</div>
      </div>
    );
  }

  if (!metrics) {
    return null;
  }

  return (
    <div className="clinical-metrics">
      <div className="clinical-metrics-header">
        <div className="clinical-metrics-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="20" x2="18" y2="10" />
            <line x1="12" y1="20" x2="12" y2="4" />
            <line x1="6" y1="20" x2="6" y2="14" />
          </svg>
        </div>
        <span className="clinical-metrics-title">Clinical Metrics</span>
      </div>

      <div className="clinical-metrics-grid" role="table" aria-label="Clinical performance metrics">
        {metrics.map((metric, i) => (
          <div key={metric.label} className="clinical-metric-box" role="row">
            <div className="clinical-metric-label" role="rowheader">{metric.label}</div>
            <div className={`clinical-metric-value${i === 0 ? ' highlight' : ''}`} role="cell">
              {metric.value}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
