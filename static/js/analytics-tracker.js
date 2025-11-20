// Enhanced Analytics Tracker
(function() {
    let sessionStartTime = Date.now();
    let pageStartTime = Date.now();
    let lastUrl = location.href;
    let clickCount = 0;
    let scrollCount = 0;
    
    // Get comprehensive device info
    function getDeviceInfo() {
        return {
            screen_width: window.screen.width,
            screen_height: window.screen.height,
            viewport_width: window.innerWidth,
            viewport_height: window.innerHeight,
            screen_resolution: `${window.screen.width}x${window.screen.height}`,
            color_depth: window.screen.colorDepth,
            pixel_ratio: window.devicePixelRatio || 1,
            language: navigator.language || navigator.userLanguage,
            platform: navigator.platform,
            connection_type: navigator.connection?.effectiveType || 'unknown',
            device_memory: navigator.deviceMemory || null,
            hardware_concurrency: navigator.hardwareConcurrency || null,
            online: navigator.onLine,
            cookies_enabled: navigator.cookieEnabled,
            do_not_track: navigator.doNotTrack || 'unknown'
        };
    }
    
    // Get geolocation if permitted
    async function getGeolocation() {
        return new Promise((resolve) => {
            if ('geolocation' in navigator) {
                navigator.geolocation.getCurrentPosition(
                    (position) => {
                        resolve({
                            latitude: position.coords.latitude,
                            longitude: position.coords.longitude,
                            accuracy: position.coords.accuracy
                        });
                    },
                    () => resolve({})
                );
            } else {
                resolve({});
            }
        });
    }
    
    // Auto-track every page visit with enhanced data
    window.addEventListener('load', async function() {
        try {
            const [locationData, geoData] = await Promise.all([
                fetch('https://ipapi.co/json/').then(r => r.json()).catch(() => ({})),
                getGeolocation()
            ]);
            
            const deviceInfo = getDeviceInfo();
            
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
                    page_title: document.title,
                    referrer: document.referrer,
                    ...deviceInfo,
                    ...geoData
                })
            });
        } catch (e) {
            console.log('Analytics tracking skipped');
        }
    });
    
    // Track page visibility (when user leaves/returns to tab)
    document.addEventListener('visibilitychange', function() {
        if (document.hidden) {
            const timeOnPage = Math.floor((Date.now() - pageStartTime) / 1000);
            window.trackEvent('page_visibility', 'engagement', {
                action: 'hidden',
                time_on_page: timeOnPage
            });
        } else {
            window.trackEvent('page_visibility', 'engagement', {
                action: 'visible'
            });
            pageStartTime = Date.now();
        }
    });
    
    // Track scroll depth with more detail
    let maxScroll = 0;
    let scrollTracked = {25: false, 50: false, 75: false, 100: false};
    
    window.addEventListener('scroll', function() {
        scrollCount++;
        const scrollPercent = Math.round((window.scrollY / (document.documentElement.scrollHeight - window.innerHeight)) * 100);
        
        if (scrollPercent > maxScroll) {
            maxScroll = scrollPercent;
        }
        
        // Track milestone scrolls
        [25, 50, 75, 100].forEach(milestone => {
            if (scrollPercent >= milestone && !scrollTracked[milestone]) {
                scrollTracked[milestone] = true;
                
                fetch('/api/track/scroll', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        page_url: window.location.href,
                        scroll_depth: window.scrollY,
                        scroll_percentage: scrollPercent
                    })
                }).catch(() => {});
            }
        });
    });
    
    // Track time on page before leaving
    window.addEventListener('beforeunload', function() {
        const timeOnPage = Math.floor((Date.now() - pageStartTime) / 1000);
        const sessionDuration = Math.floor((Date.now() - sessionStartTime) / 1000);
        
        // Use fetch with keepalive instead of sendBeacon for better compatibility
        fetch('/api/track/event', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                event_type: 'page_exit',
                event_category: 'engagement',
                page_url: window.location.href,
                event_data: {
                    page: window.location.pathname,
                    time_on_page: timeOnPage,
                    session_duration: sessionDuration,
                    max_scroll: maxScroll,
                    total_clicks: clickCount,
                    total_scrolls: scrollCount
                }
            }),
            keepalive: true
        }).catch(() => {});
    });
    
    // Track clicks with position
    document.addEventListener('click', function(e) {
        clickCount++;
        
        const target = e.target.closest('a, button, [data-track], input[type="submit"]');
        
        fetch('/api/track/click', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                element_id: target?.id || null,
                element_class: target?.className || null,
                element_tag: target?.tagName || e.target.tagName,
                element_text: (target?.innerText || e.target.innerText || '').substring(0, 100),
                page_url: window.location.href,
                x_position: e.clientX,
                y_position: e.clientY
            })
        }).catch(() => {});
        
        // Additional tracking for specific elements
        if (target) {
            const trackData = {
                element: target.tagName,
                text: target.innerText?.substring(0, 50) || '',
                href: target.href || '',
                id: target.id || '',
                class: target.className || '',
                x: e.clientX,
                y: e.clientY
            };
            
            window.trackEvent('element_click', 'interaction', trackData);
        }
    });
    
    // Track form interactions
    let formStartTimes = new Map();
    
    document.addEventListener('focus', function(e) {
        if (e.target.form && !formStartTimes.has(e.target.form)) {
            formStartTimes.set(e.target.form, Date.now());
        }
    }, true);
    
    document.addEventListener('submit', function(e) {
        if (e.target.tagName === 'FORM') {
            const formStartTime = formStartTimes.get(e.target);
            const timeToSubmit = formStartTime ? Math.floor((Date.now() - formStartTime) / 1000) : null;
            
            const fields = Array.from(e.target.elements)
                .filter(el => el.name)
                .map(el => ({
                    name: el.name,
                    type: el.type,
                    filled: !!el.value
                }));
            
            fetch('/api/track/form', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    form_name: e.target.name || e.target.id || 'unnamed_form',
                    page_url: window.location.href,
                    fields_filled: fields,
                    submission_success: true,
                    time_to_submit: timeToSubmit
                })
            }).catch(() => {});
            
            formStartTimes.delete(e.target);
        }
    });
    
    // Track SPA page changes
    let urlCheckInterval = setInterval(() => {
        const url = location.href;
        if (url !== lastUrl) {
            const timeOnPreviousPage = Math.floor((Date.now() - pageStartTime) / 1000);
            
            // Update page view with exit data
            fetch('/api/track', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    page_url: url,
                    page_title: document.title,
                    referrer: lastUrl,
                    time_on_page: timeOnPreviousPage,
                    scroll_depth: maxScroll,
                    ...getDeviceInfo()
                })
            }).catch(() => {});
            
            lastUrl = url;
            pageStartTime = Date.now();
            maxScroll = 0;
            clickCount = 0;
            scrollCount = 0;
            scrollTracked = {25: false, 50: false, 75: false, 100: false};
            
            window.trackEvent('spa_navigation', 'navigation', {
                from: lastUrl,
                to: url
            });
        }
    }, 1000);
    
    // Helper function to track searches
    window.trackSearch = async function(query, resultsCount, category = 'drug', options = {}) {
        try {
            await fetch('/api/track/search', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    query: query,
                    results_count: resultsCount,
                    category: category,
                    clicked_result_position: options.clickedPosition || null,
                    clicked_result_id: options.clickedResultId || null,
                    time_to_first_click: options.timeToClick || null,
                    refined_query: options.isRefined || false
                })
            });
        } catch (e) {}
    };
    
    // Helper function to track drug views with comprehensive data
    window.trackDrugView = function(drugId, options = {}) {
        const startTime = Date.now();
        let sectionsViewed = new Set();
        let interactionCount = 0;
        let maxScrollDepth = 0;
        
        // Track section views
        const sectionObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting && entry.target.dataset.section) {
                    sectionsViewed.add(entry.target.dataset.section);
                }
            });
        }, { threshold: 0.5 });
        
        document.querySelectorAll('[data-section]').forEach(section => {
            sectionObserver.observe(section);
        });
        
        // Track interactions
        const trackInteraction = () => interactionCount++;
        document.addEventListener('click', trackInteraction);
        
        // Track scroll
        const trackScroll = () => {
            const depth = Math.round((window.scrollY / (document.documentElement.scrollHeight - window.innerHeight)) * 100);
            if (depth > maxScrollDepth) maxScrollDepth = depth;
        };
        window.addEventListener('scroll', trackScroll);
        
        // Send data on page exit
        window.addEventListener('beforeunload', function() {
            const duration = Math.floor((Date.now() - startTime) / 1000);
            
            // Use fetch with keepalive instead of sendBeacon
            fetch('/api/track/drug-view', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    drug_id: drugId,
                    duration: duration,
                    sections_viewed: Array.from(sectionsViewed),
                    scroll_depth: maxScrollDepth,
                    interactions: interactionCount,
                    shared: options.shared || false,
                    bookmarked: options.bookmarked || false
                }),
                keepalive: true
            }).catch(() => {});
            
            document.removeEventListener('click', trackInteraction);
            window.removeEventListener('scroll', trackScroll);
            sectionObserver.disconnect();
        });
        
        return {
            markShared: () => options.shared = true,
            markBookmarked: () => options.bookmarked = true
        };
    };
    
    // Helper function to track interaction checks
    window.trackInteraction = async function(drugIds, interactionsFound, severityLevels, options = {}) {
        try {
            await fetch('/api/track/interaction', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    drug_ids: drugIds,
                    interactions_found: interactionsFound,
                    severity_levels: severityLevels,
                    checked_food_interactions: options.checkedFood || false,
                    checked_disease_interactions: options.checkedDisease || false,
                    checked_lab_interactions: options.checkedLab || false,
                    time_spent: options.timeSpent || null,
                    exported: options.exported || false
                })
            });
        } catch (e) {}
    };
    
    // Helper function to track custom events
    window.trackEvent = async function(eventType, eventCategory, eventData, eventValue = null) {
        try {
            await fetch('/api/track/event', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    event_type: eventType,
                    event_category: eventCategory,
                    event_data: eventData,
                    page_url: window.location.href,
                    event_value: eventValue
                })
            });
        } catch (e) {}
    };
    
    // Track page performance
    window.addEventListener('load', function() {
        setTimeout(function() {
            if (window.performance && window.performance.timing) {
                const perfData = window.performance.timing;
                const pageLoadTime = perfData.loadEventEnd - perfData.navigationStart;
                const connectTime = perfData.responseEnd - perfData.requestStart;
                const renderTime = perfData.domComplete - perfData.domLoading;
                const dnsTime = perfData.domainLookupEnd - perfData.domainLookupStart;
                const ttfb = perfData.responseStart - perfData.navigationStart;
                
                window.trackEvent('page_performance', 'performance', {
                    page_load_time: pageLoadTime,
                    connect_time: connectTime,
                    render_time: renderTime,
                    dns_time: dnsTime,
                    time_to_first_byte: ttfb,
                    page: window.location.pathname
                }, pageLoadTime);
            }
            
            // Track resource timing
            if (window.performance && window.performance.getEntriesByType) {
                const resources = window.performance.getEntriesByType('resource');
                const slowResources = resources
                    .filter(r => r.duration > 1000)
                    .map(r => ({
                        name: r.name,
                        duration: Math.round(r.duration),
                        type: r.initiatorType
                    }));
                
                if (slowResources.length > 0) {
                    window.trackEvent('slow_resources', 'performance', {
                        resources: slowResources,
                        page: window.location.pathname
                    });
                }
            }
        }, 0);
    });
    
    // Track errors with more detail
    window.addEventListener('error', function(e) {
        const deviceInfo = getDeviceInfo();
        
        fetch('/api/track/error', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                error_message: e.message,
                error_stack: e.error?.stack || 'No stack trace',
                page_url: window.location.href,
                browser: navigator.userAgent,
                os: navigator.platform
            })
        }).catch(() => {});
        
        window.trackEvent('javascript_error', 'error', {
            message: e.message,
            filename: e.filename,
            line: e.lineno,
            column: e.colno,
            stack: e.error?.stack,
            page: window.location.pathname,
            userAgent: navigator.userAgent
        });
    });
    
    // Track unhandled promise rejections
    window.addEventListener('unhandledrejection', function(e) {
        window.trackEvent('promise_rejection', 'error', {
            reason: e.reason?.toString() || 'Unknown',
            page: window.location.pathname
        });
    });
    
    // Track network status
    window.addEventListener('online', () => {
        window.trackEvent('network_status', 'system', { status: 'online' });
    });
    
    window.addEventListener('offline', () => {
        window.trackEvent('network_status', 'system', { status: 'offline' });
    });
    
    // Track connection speed changes
    if (navigator.connection) {
        navigator.connection.addEventListener('change', () => {
            window.trackEvent('connection_change', 'system', {
                effectiveType: navigator.connection.effectiveType,
                downlink: navigator.connection.downlink,
                rtt: navigator.connection.rtt
            });
        });
    }
    
    // Track battery status if available
    if (navigator.getBattery) {
        navigator.getBattery().then(battery => {
            window.trackEvent('battery_status', 'system', {
                level: Math.round(battery.level * 100),
                charging: battery.charging
            });
        });
    }
    
    // Track memory usage if available
    if (performance.memory) {
        setInterval(() => {
            const memoryUsage = Math.round((performance.memory.usedJSHeapSize / performance.memory.jsHeapSizeLimit) * 100);
            if (memoryUsage > 90) {
                window.trackEvent('high_memory_usage', 'performance', {
                    usage_percent: memoryUsage,
                    used_mb: Math.round(performance.memory.usedJSHeapSize / 1048576),
                    limit_mb: Math.round(performance.memory.jsHeapSizeLimit / 1048576)
                });
            }
        }, 60000); // Check every minute
    }
    
    // Track orientation changes
    window.addEventListener('orientationchange', () => {
        window.trackEvent('orientation_change', 'system', {
            orientation: window.orientation,
            viewport: `${window.innerWidth}x${window.innerHeight}`
        });
    });
    
    // Cleanup on page unload
    window.addEventListener('unload', () => {
        clearInterval(urlCheckInterval);
    });
    
})();