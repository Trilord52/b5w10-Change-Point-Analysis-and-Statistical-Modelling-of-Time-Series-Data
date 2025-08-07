import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';

// API configuration
const API_BASE_URL = 'http://localhost:5000/api';
const API_TIMEOUT = 30000; // 30 seconds

// Configure axios defaults
axios.defaults.timeout = API_TIMEOUT;
axios.defaults.baseURL = API_BASE_URL;

function App() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [priceData, setPriceData] = useState([]);
  const [eventsData, setEventsData] = useState([]);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [businessInsights, setBusinessInsights] = useState(null);
  const [summaryStats, setSummaryStats] = useState(null);
  const [apiHealth, setApiHealth] = useState(null);
  const [filters, setFilters] = useState({
    startDate: '',
    endDate: '',
    categories: [],
    impactLevels: []
  });

  // Check API health on component mount
  useEffect(() => {
    checkApiHealth();
  }, []);

  // Load dashboard data after API health check
  useEffect(() => {
    if (apiHealth?.status === 'healthy') {
      loadDashboardData();
    }
  }, [apiHealth]);

  const checkApiHealth = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await axios.get('/health');
      setApiHealth(response.data);
      
      if (response.data.status !== 'healthy') {
        throw new Error('API is not healthy');
      }
    } catch (err) {
      setError(`API Health Check Failed: ${err.message}. Please ensure the backend server is running.`);
      setLoading(false);
    }
  };

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Load all data in parallel
      const [priceResponse, eventsResponse, insightsResponse, summaryResponse] = await Promise.all([
        axios.get('/data/price'),
        axios.get('/data/events'),
        axios.get('/insights/business'),
        axios.get('/visualizations/summary')
      ]);

      // Validate responses
      if (priceResponse.data.success) {
        setPriceData(priceResponse.data.data);
        setSummaryStats(priceResponse.data.summary);
      } else {
        throw new Error('Failed to load price data');
      }

      if (eventsResponse.data.success) {
        setEventsData(eventsResponse.data.data);
      } else {
        throw new Error('Failed to load events data');
      }

      if (insightsResponse.data.success) {
        setBusinessInsights(insightsResponse.data.insights);
      }

      if (summaryResponse.data.success) {
        setSummaryStats(prev => ({ ...prev, ...summaryResponse.data.summary }));
      }

      setLoading(false);
    } catch (err) {
      setError(`Failed to load dashboard data: ${err.message}`);
      setLoading(false);
    }
  };

  const runAnalysis = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await axios.post('/analysis/change-points', {
        model_type: 'single',
        draws: 1000,
        tune: 500
      });
      
      if (response.data.success) {
        setAnalysisResults(response.data.results);
        
        // Show success message
        console.log('Analysis completed successfully');
      } else {
        throw new Error('Analysis failed');
      }
      setLoading(false);
    } catch (err) {
      setError(`Analysis failed: ${err.message}`);
      setLoading(false);
    }
  };

  const applyFilters = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await axios.post('/data/filtered', filters);
      
      if (response.data.success) {
        setPriceData(response.data.price_data);
        setEventsData(response.data.events_data);
        console.log('Filters applied successfully');
      } else {
        throw new Error('Failed to apply filters');
      }
      setLoading(false);
    } catch (err) {
      setError(`Failed to apply filters: ${err.message}`);
      setLoading(false);
    }
  };

  const handleFilterChange = (field, value) => {
    setFilters(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const resetFilters = () => {
    setFilters({
      startDate: '',
      endDate: '',
      categories: [],
      impactLevels: []
    });
    loadDashboardData();
  };

  // Loading component
  if (loading && !priceData.length) {
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '100vh',
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        color: 'white',
        fontSize: '1.2rem'
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ marginBottom: '1rem' }}>🔄</div>
          <div>Loading Brent Oil Analysis Dashboard...</div>
          <div style={{ fontSize: '0.9rem', marginTop: '0.5rem', opacity: 0.8 }}>
            {apiHealth ? 'API Connected' : 'Checking API Health...'}
          </div>
        </div>
      </div>
    );
  }

  // Error component
  if (error) {
    return (
      <div style={{ 
        minHeight: '100vh', 
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        padding: '2rem'
      }}>
        <div style={{ 
          background: 'rgba(255, 255, 255, 0.95)', 
          padding: '2rem', 
          borderRadius: '12px',
          maxWidth: '600px',
          textAlign: 'center'
        }}>
          <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>⚠️</div>
          <h2 style={{ color: '#c33', marginBottom: '1rem' }}>Error</h2>
          <p style={{ marginBottom: '1.5rem', lineHeight: '1.6' }}>{error}</p>
          <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
            <button 
              onClick={checkApiHealth}
              style={{ 
                background: '#667eea', 
                color: 'white', 
                border: 'none', 
                padding: '0.5rem 1rem', 
                borderRadius: '6px', 
                cursor: 'pointer' 
              }}
            >
              Retry Connection
            </button>
            <button 
              onClick={() => window.location.reload()}
              style={{ 
                background: '#6c757d', 
                color: 'white', 
                border: 'none', 
                padding: '0.5rem 1rem', 
                borderRadius: '6px', 
                cursor: 'pointer' 
              }}
            >
              Reload Page
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
      {/* Header */}
      <header style={{ 
        background: 'rgba(255, 255, 255, 0.95)', 
        padding: '1rem 2rem', 
        boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
        backdropFilter: 'blur(10px)'
      }}>
        <h1 style={{ margin: '0 0 0.5rem 0', color: '#333' }}>Brent Oil Price Change Point Analysis Dashboard</h1>
        <p style={{ margin: 0, color: '#666' }}>Interactive visualization of geopolitical events and oil price dynamics</p>
        {apiHealth && (
          <div style={{ 
            marginTop: '0.5rem', 
            fontSize: '0.9rem', 
            color: apiHealth.status === 'healthy' ? '#28a745' : '#dc3545' 
          }}>
            API Status: {apiHealth.status === 'healthy' ? '✅ Connected' : '❌ Disconnected'}
          </div>
        )}
      </header>

      {/* Main Content */}
      <main style={{ padding: '2rem', maxWidth: '1400px', margin: '0 auto' }}>
        {/* Summary Metrics */}
        {summaryStats && (
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', 
            gap: '1rem', 
            marginBottom: '2rem' 
          }}>
            <div style={{ 
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 
              color: 'white', 
              padding: '1.5rem', 
              borderRadius: '12px', 
              textAlign: 'center',
              boxShadow: '0 4px 20px rgba(0,0,0,0.1)'
            }}>
              <div style={{ fontSize: '2rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>
                {summaryStats.total_observations?.toLocaleString() || priceData.length.toLocaleString()}
              </div>
              <div style={{ fontSize: '0.9rem', opacity: 0.9 }}>Total Observations</div>
            </div>
            <div style={{ 
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 
              color: 'white', 
              padding: '1.5rem', 
              borderRadius: '12px', 
              textAlign: 'center',
              boxShadow: '0 4px 20px rgba(0,0,0,0.1)'
            }}>
              <div style={{ fontSize: '2rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>
                {eventsData.length}
              </div>
              <div style={{ fontSize: '0.9rem', opacity: 0.9 }}>Geopolitical Events</div>
            </div>
            <div style={{ 
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 
              color: 'white', 
              padding: '1.5rem', 
              borderRadius: '12px', 
              textAlign: 'center',
              boxShadow: '0 4px 20px rgba(0,0,0,0.1)'
            }}>
              <div style={{ fontSize: '2rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>
                ${summaryStats.price_range?.mean?.toFixed(2) || 'N/A'}
              </div>
              <div style={{ fontSize: '0.9rem', opacity: 0.9 }}>Average Price</div>
            </div>
            <div style={{ 
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 
              color: 'white', 
              padding: '1.5rem', 
              borderRadius: '12px', 
              textAlign: 'center',
              boxShadow: '0 4px 20px rgba(0,0,0,0.1)'
            }}>
              <div style={{ fontSize: '2rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>
                {analysisResults ? '✅' : '⏳'}
              </div>
              <div style={{ fontSize: '0.9rem', opacity: 0.9 }}>
                {analysisResults ? 'Analysis Complete' : 'Ready for Analysis'}
              </div>
            </div>
          </div>
        )}

        {/* Filters Section */}
        <div style={{ 
          background: 'white', 
          borderRadius: '12px', 
          padding: '1.5rem', 
          marginBottom: '2rem',
          boxShadow: '0 4px 20px rgba(0,0,0,0.1)'
        }}>
          <h3 style={{ marginTop: 0, marginBottom: '1rem' }}>Data Filters</h3>
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
            gap: '1rem',
            marginBottom: '1rem'
          }}>
            <div>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '600' }}>Start Date</label>
              <input
                type="date"
                value={filters.startDate}
                onChange={(e) => handleFilterChange('startDate', e.target.value)}
                style={{ 
                  width: '100%', 
                  padding: '0.5rem', 
                  border: '1px solid #ddd', 
                  borderRadius: '4px' 
                }}
              />
            </div>
            <div>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '600' }}>End Date</label>
              <input
                type="date"
                value={filters.endDate}
                onChange={(e) => handleFilterChange('endDate', e.target.value)}
                style={{ 
                  width: '100%', 
                  padding: '0.5rem', 
                  border: '1px solid #ddd', 
                  borderRadius: '4px' 
                }}
              />
            </div>
            <div>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '600' }}>Event Categories</label>
              <select
                multiple
                value={filters.categories}
                onChange={(e) => handleFilterChange('categories', Array.from(e.target.selectedOptions, option => option.value))}
                style={{ 
                  width: '100%', 
                  padding: '0.5rem', 
                  border: '1px solid #ddd', 
                  borderRadius: '4px',
                  minHeight: '80px'
                }}
              >
                <option value="War/Conflict">War/Conflict</option>
                <option value="OPEC Policy">OPEC Policy</option>
                <option value="Economic Crisis">Economic Crisis</option>
                <option value="Natural Disaster">Natural Disaster</option>
                <option value="Terrorism">Terrorism</option>
                <option value="Political Unrest">Political Unrest</option>
                <option value="Technology/Supply">Technology/Supply</option>
                <option value="Pandemic">Pandemic</option>
                <option value="Policy Change">Policy Change</option>
              </select>
            </div>
            <div>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '600' }}>Impact Levels</label>
              <select
                multiple
                value={filters.impactLevels}
                onChange={(e) => handleFilterChange('impactLevels', Array.from(e.target.selectedOptions, option => option.value))}
                style={{ 
                  width: '100%', 
                  padding: '0.5rem', 
                  border: '1px solid #ddd', 
                  borderRadius: '4px',
                  minHeight: '80px'
                }}
              >
                <option value="Very High">Very High</option>
                <option value="High">High</option>
                <option value="Medium">Medium</option>
                <option value="Low">Low</option>
              </select>
            </div>
          </div>
          <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
            <button 
              onClick={applyFilters}
              disabled={loading}
              style={{ 
                background: '#667eea', 
                color: 'white', 
                border: 'none', 
                padding: '0.5rem 1rem', 
                borderRadius: '6px', 
                cursor: loading ? 'not-allowed' : 'pointer',
                opacity: loading ? 0.6 : 1
              }}
            >
              {loading ? 'Applying...' : 'Apply Filters'}
            </button>
            <button 
              onClick={resetFilters}
              disabled={loading}
              style={{ 
                background: '#6c757d', 
                color: 'white', 
                border: 'none', 
                padding: '0.5rem 1rem', 
                borderRadius: '6px', 
                cursor: loading ? 'not-allowed' : 'pointer',
                opacity: loading ? 0.6 : 1
              }}
            >
              Reset Filters
            </button>
          </div>
        </div>

        {/* Price Chart */}
        <div style={{ 
          background: 'white', 
          borderRadius: '12px', 
          boxShadow: '0 4px 20px rgba(0,0,0,0.1)', 
          padding: '1.5rem', 
          marginBottom: '1.5rem' 
        }}>
          <h3 style={{ marginTop: 0 }}>Brent Oil Price Series</h3>
          <div style={{ height: '400px', width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={priceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="Date" 
                  tick={{ fontSize: 12 }}
                  angle={-45}
                  textAnchor="end"
                  height={80}
                />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip 
                  formatter={(value) => [`$${value}`, 'Price']}
                  labelFormatter={(label) => `Date: ${label}`}
                />
                <Line 
                  type="monotone" 
                  dataKey="Price" 
                  stroke="#8884d8" 
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Analysis Results */}
        {analysisResults && (
          <div style={{ 
            background: 'white', 
            borderRadius: '12px', 
            boxShadow: '0 4px 20px rgba(0,0,0,0.1)', 
            padding: '1.5rem', 
            marginBottom: '1.5rem' 
          }}>
            <h3 style={{ marginTop: 0 }}>Change Point Analysis Results</h3>
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
              gap: '1rem' 
            }}>
              <div>
                <strong>Change Point Date:</strong> {analysisResults.single_change_point?.change_point_date || 'N/A'}
              </div>
              <div>
                <strong>Mean Change:</strong> {analysisResults.single_change_point?.mean_change?.toFixed(6) || 'N/A'}
              </div>
              <div>
                <strong>Volatility Change:</strong> {analysisResults.single_change_point?.volatility_change?.toFixed(6) || 'N/A'}
              </div>
              <div>
                <strong>Model Convergence:</strong> {analysisResults.model_performance?.single_cp_converged ? '✅' : '❌'}
              </div>
            </div>
          </div>
        )}

        {/* Business Insights */}
        {businessInsights && (
          <div style={{ 
            background: 'white', 
            borderRadius: '12px', 
            boxShadow: '0 4px 20px rgba(0,0,0,0.1)', 
            padding: '1.5rem', 
            marginBottom: '1.5rem' 
          }}>
            <h3 style={{ marginTop: 0 }}>Business Insights & Recommendations</h3>
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', 
              gap: '2rem' 
            }}>
              <div>
                <h4 style={{ color: '#667eea', marginBottom: '1rem' }}>Key Findings</h4>
                <ul style={{ paddingLeft: '1.5rem', lineHeight: '1.6' }}>
                  {businessInsights.key_findings?.map((finding, index) => (
                    <li key={index} style={{ marginBottom: '0.5rem' }}>{finding}</li>
                  ))}
                </ul>
              </div>
              <div>
                <h4 style={{ color: '#667eea', marginBottom: '1rem' }}>Investment Implications</h4>
                <ul style={{ paddingLeft: '1.5rem', lineHeight: '1.6' }}>
                  {businessInsights.investment_implications?.map((implication, index) => (
                    <li key={index} style={{ marginBottom: '0.5rem' }}>{implication}</li>
                  ))}
                </ul>
              </div>
              <div>
                <h4 style={{ color: '#667eea', marginBottom: '1rem' }}>Risk Management</h4>
                <ul style={{ paddingLeft: '1.5rem', lineHeight: '1.6' }}>
                  {businessInsights.risk_management_recommendations?.map((rec, index) => (
                    <li key={index} style={{ marginBottom: '0.5rem' }}>{rec}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div style={{ 
          background: 'white', 
          borderRadius: '12px', 
          boxShadow: '0 4px 20px rgba(0,0,0,0.1)', 
          padding: '1.5rem' 
        }}>
          <h3 style={{ marginTop: 0 }}>Analysis Actions</h3>
          <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
            <button 
              onClick={runAnalysis}
              disabled={loading}
              style={{ 
                background: '#667eea', 
                color: 'white', 
                border: 'none', 
                padding: '0.5rem 1rem', 
                borderRadius: '6px', 
                cursor: loading ? 'not-allowed' : 'pointer',
                opacity: loading ? 0.6 : 1
              }}
            >
              {loading ? 'Running Analysis...' : 'Run Change Point Analysis'}
            </button>
            <button 
              onClick={loadDashboardData}
              disabled={loading}
              style={{ 
                background: '#6c757d', 
                color: 'white', 
                border: 'none', 
                padding: '0.5rem 1rem', 
                borderRadius: '6px', 
                cursor: loading ? 'not-allowed' : 'pointer',
                opacity: loading ? 0.6 : 1
              }}
            >
              {loading ? 'Refreshing...' : 'Refresh Data'}
            </button>
            <button 
              onClick={checkApiHealth}
              disabled={loading}
              style={{ 
                background: '#28a745', 
                color: 'white', 
                border: 'none', 
                padding: '0.5rem 1rem', 
                borderRadius: '6px', 
                cursor: loading ? 'not-allowed' : 'pointer',
                opacity: loading ? 0.6 : 1
              }}
            >
              Check API Health
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
