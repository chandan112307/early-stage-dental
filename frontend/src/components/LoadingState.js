import React from 'react';
import './LoadingState.css';

export default function LoadingState({ message = 'Analyzing X-ray...' }) {
  return (
    <div className="loading-overlay" role="status" aria-live="polite">
      <div className="loading-spinner" aria-hidden="true" />
      <div className="loading-text">{message}</div>
      <div className="loading-subtext">Running neural network inference</div>
      <div className="loading-progress-dots" aria-hidden="true">
        <span className="loading-progress-dot" />
        <span className="loading-progress-dot" />
        <span className="loading-progress-dot" />
      </div>
    </div>
  );
}
