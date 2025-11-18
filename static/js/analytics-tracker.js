// Auto-track every page visit
window.addEventListener('load', async function() {
    try {
        const locationResponse = await fetch('https://ipapi.co/json/');
        const locationData = await locationResponse.json();
        
        await fetch('/api/track', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                country: locationData.country_name,
                city: locationData.city,
                region: locationData.region,
                timezone: locationData.timezone,
                isp: locationData.org,
                page_url: window.location.href,
                referrer: document.referrer
            })
        });
    } catch (e) {
        console.log('Analytics tracking skipped');
    }
});

// Helper function to track searches (call this from your search pages)
window.trackSearch = async function(query, resultsCount, category = 'drug') {
    try {
        await fetch('/api/track/search', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                query: query,
                results_count: resultsCount,
                category: category
            })
        });
    } catch (e) {}
};

// Helper function to track drug views (call this from drug detail pages)
window.trackDrugView = async function(drugId, duration) {
    try {
        await fetch('/api/track/drug-view', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                drug_id: drugId,
                duration: duration
            })
        });
    } catch (e) {}
};

// Helper function to track interaction checks
window.trackInteraction = async function(drugIds, interactionsFound, severityLevels) {
    try {
        await fetch('/api/track/interaction', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                drug_ids: drugIds,
                interactions_found: interactionsFound,
                severity_levels: severityLevels
            })
        });
    } catch (e) {}
};