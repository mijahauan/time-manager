/**
 * Discrimination Charts for time-manager
 * 
 * Visualization components for WWV/WWVH discrimination analysis.
 * Uses Plotly.js for rendering.
 */

// Chart rendering functions are embedded in discrimination.html
// This file provides additional utilities and shared functions

/**
 * Format a number with specified decimal places
 * @param {number} value - Value to format
 * @param {number} digits - Number of decimal places
 * @returns {string} Formatted number
 */
function formatNumber(value, digits = 1) {
    if (typeof value !== 'number' || !isFinite(value)) {
        return '--';
    }
    return value.toFixed(digits);
}

/**
 * Convert timestamp to UTC string for Plotly
 * @param {string|Date|number} date - Date to convert
 * @returns {string} ISO string
 */
function toUTCString(date) {
    if (!date) return null;
    const d = date instanceof Date ? date : new Date(date);
    return d.toISOString();
}

/**
 * Get color for station
 * @param {string} station - Station name (WWV, WWVH, etc.)
 * @returns {string} Color hex code
 */
function getStationColor(station) {
    const colors = {
        'WWV': '#22c55e',
        'WWVH': '#ef4444',
        'CHU': '#3b82f6',
        'BALANCED': '#8b5cf6',
        'NONE': '#64748b'
    };
    return colors[station?.toUpperCase()] || colors.NONE;
}

/**
 * Common Plotly layout settings for dark theme
 */
const darkThemeLayout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'rgba(30, 41, 59, 0.5)',
    font: { color: '#e0e0e0' },
    xaxis: {
        gridcolor: '#334155',
        tickcolor: '#64748b',
        tickfont: { color: '#94a3b8' }
    },
    yaxis: {
        gridcolor: '#334155',
        tickcolor: '#64748b',
        tickfont: { color: '#94a3b8' }
    },
    legend: {
        font: { color: '#e0e0e0' },
        bgcolor: 'rgba(30, 41, 59, 0.8)'
    }
};

/**
 * Common Plotly config
 */
const plotlyConfig = {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d']
};

// Export for use in other scripts
if (typeof window !== 'undefined') {
    window.formatNumber = formatNumber;
    window.toUTCString = toUTCString;
    window.getStationColor = getStationColor;
    window.darkThemeLayout = darkThemeLayout;
    window.plotlyConfig = plotlyConfig;
}
