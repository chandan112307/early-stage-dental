import React, { useState, useCallback } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import XRayViewer from './components/XRayViewer';
import AnalysisResults from './components/AnalysisResults';
import ClinicalMetrics from './components/ClinicalMetrics';
import DiagnosticHistory from './components/DiagnosticHistory';
import LoadingState from './components/LoadingState';
import { predictImage } from './utils/api';
import './App.css';

// idle | uploading | processing | success | no-caries | error
const INITIAL_STATE = {
  appState: 'idle',
  imageUrl: null,
  predictions: null,
  error: null,
  activeNav: 'upload',
  sidebarOpen: false,
};

export default function App() {
  const [state, setState] = useState(INITIAL_STATE);

  const { appState, imageUrl, predictions, error, activeNav, sidebarOpen } = state;

  const handleNavChange = useCallback((nav) => {
    setState((prev) => ({ ...prev, activeNav: nav, sidebarOpen: false }));
  }, []);

  const toggleSidebar = useCallback(() => {
    setState((prev) => ({ ...prev, sidebarOpen: !prev.sidebarOpen }));
  }, []);

  const handleFileSelect = useCallback(async (file) => {
    const objectUrl = URL.createObjectURL(file);

    setState((prev) => ({
      ...prev,
      appState: 'uploading',
      imageUrl: objectUrl,
      predictions: null,
      error: null,
    }));

    // Transition quickly to processing state
    setState((prev) => ({ ...prev, appState: 'processing' }));

    try {
      const result = await predictImage(file);

      const hasCaries =
        result.classification === 'caries' ||
        result.classification === 'Caries' ||
        (result.detections && result.detections.length > 0);

      setState((prev) => ({
        ...prev,
        appState: hasCaries ? 'success' : 'no-caries',
        predictions: result,
      }));
    } catch (err) {
      setState((prev) => ({
        ...prev,
        appState: 'error',
        error: err.message || 'An unexpected error occurred during analysis.',
      }));
    }
  }, []);

  const dismissError = useCallback(() => {
    setState((prev) => ({
      ...prev,
      appState: 'idle',
      error: null,
      imageUrl: null,
      predictions: null,
    }));
  }, []);

  const isLoading = appState === 'uploading' || appState === 'processing';

  return (
    <div className="app">
      <Sidebar
        activeNav={activeNav}
        onNavChange={handleNavChange}
        isOpen={sidebarOpen}
        onToggle={toggleSidebar}
      />

      <main className="app-main">
        <Header />

        <div className="app-content">
          {error && appState === 'error' && (
            <div className="app-error-banner" role="alert">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10" />
                <line x1="15" y1="9" x2="9" y2="15" />
                <line x1="9" y1="9" x2="15" y2="15" />
              </svg>
              {error}
              <button className="app-error-dismiss" onClick={dismissError} aria-label="Dismiss error">
                Try Again
              </button>
            </div>
          )}

          <div className="app-content-top">
            <div className="app-viewer-wrapper">
              <XRayViewer
                imageUrl={imageUrl}
                predictions={predictions}
                appState={appState}
                onFileSelect={handleFileSelect}
              />
              {isLoading && <LoadingState />}
            </div>

            <div className="app-results-column">
              <AnalysisResults predictions={predictions} appState={appState} />
              <ClinicalMetrics />
            </div>
          </div>

          <DiagnosticHistory />
        </div>
      </main>
    </div>
  );
}
