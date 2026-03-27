import React from 'react';
import './ClinicalMetrics.css';

const DEFAULT_METRICS = [
  { label: 'Accuracy', value: '98.4%' },
  { label: 'Precision', value: '96.1%' },
  { label: 'Recall', value: '94.8%' },
  { label: 'F1-Score', value: '0.95' },
];

export default function ClinicalMetrics({ metrics }) {
  const displayMetrics = metrics || DEFAULT_METRICS;

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
        {displayMetrics.map((metric, i) => (
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
