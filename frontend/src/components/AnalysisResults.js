import React from 'react';
import './AnalysisResults.css';

export default function AnalysisResults({ predictions, appState }) {
  const hasResults = appState === 'success' || appState === 'no-caries';
  const isCaries = appState === 'success';

  const confidence = predictions?.confidence ?? 0;
  const confidencePercent = Math.round(confidence * 100);

  return (
    <div className="analysis-results">
      <div className="analysis-header">
        <div className="analysis-header-left">
          <div className="analysis-header-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
            </svg>
          </div>
          <span className="analysis-header-title">Analysis Results</span>
        </div>
        <span className="analysis-version-badge">v2.4.1</span>
      </div>

      {!hasResults ? (
        <div className="analysis-empty">
          <div className="analysis-empty-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
          </div>
          Upload an X-ray image to see analysis results
        </div>
      ) : (
        <div className="analysis-body">
          <div className="analysis-label">PRIMARY FINDING</div>
          <div className={`analysis-finding ${isCaries ? 'caries' : 'no-caries'}`}>
            <span className={`analysis-finding-icon ${isCaries ? 'caries' : 'no-caries'}`}>
              {isCaries ? (
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                  <line x1="12" y1="9" x2="12" y2="13" />
                  <line x1="12" y1="17" x2="12.01" y2="17" />
                </svg>
              ) : (
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                  <polyline points="22 4 12 14.01 9 11.01" />
                </svg>
              )}
            </span>
            {isCaries ? 'Caries Detected' : 'No Caries Detected'}
          </div>

          <div className="analysis-confidence">
            <div className="analysis-confidence-header">
              <span className="analysis-confidence-label">Confidence Score</span>
              <span className="analysis-confidence-value">{confidencePercent}%</span>
            </div>
            <div className="analysis-confidence-bar" role="progressbar" aria-valuenow={confidencePercent} aria-valuemin={0} aria-valuemax={100}>
              <div
                className="analysis-confidence-fill"
                style={{ width: `${confidencePercent}%` }}
              />
            </div>
          </div>

          <div className="analysis-meta">
            <div className="analysis-meta-label">Neural Architecture</div>
            <div className="analysis-meta-value">MobileNetV2</div>
          </div>

          {predictions?.inference_time_ms && (
            <div className="analysis-meta">
              <div className="analysis-meta-label">Inference Time</div>
              <div className="analysis-meta-value">{predictions.inference_time_ms}ms</div>
            </div>
          )}

          <button className="analysis-download-btn" aria-label="Download clinical report">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="7 10 12 15 17 10" />
              <line x1="12" y1="15" x2="12" y2="3" />
            </svg>
            Download Clinical Report
          </button>
        </div>
      )}
    </div>
  );
}
