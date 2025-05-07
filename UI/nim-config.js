/**
 * NVIDIA Inference Microservices (NIM) Configuration
 *
 * This file contains settings for connecting to NVIDIA NIM services.
 */

const NIM_CONFIG = {
    // Your NVIDIA NIM API key
    API_KEY: "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",

    // API endpoints
    ENDPOINTS: {
        BASE_URL: "http://127.0.0.1:3000",
        CHAT_COMPLETIONS: "/api/rag_chat",
        COMPLETIONS: "/completions",
        EMBEDDINGS: "/embeddings"
    },

    // Models available through NIM
    MODELS: {
        CHAT: "mistralai/mixtral-8x7b-instruct-v0.1",
        TEXT: "mistralai/mixtral-8x7b-instruct-v0.1",
        EMBEDDINGS: "nvidia/nv-embed-v1"
    },

    // Default parameters for API calls
    DEFAULTS: {
        MAX_TOKENS: 4096,
        TEMPERATURE: 0.6,
        TOP_P: 0.7,
        FREQUENCY_PENALTY: 0,
        PRESENCE_PENALTY: 0,
        STREAM: true
    },

    // Settings for connection and retries
    CONNECTION: {
        TIMEOUT_MS: 30000,
        MAX_RETRIES: 3,
        RETRY_DELAY_MS: 1000
    }
};

/**
 * Sets the API key for NIM services
 * @param {string} apiKey - The API key to use for authentication
 */
function setNimApiKey(apiKey) {
    if (!apiKey || typeof apiKey !== 'string' || apiKey.trim() === '') {
        console.error("Invalid API key provided");
        return false;
    }

    // Store the API key in the configuration
    NIM_CONFIG.API_KEY = apiKey;

    // Also save to localStorage for persistence
    localStorage.setItem('nim_api_key', apiKey);

    console.log("NIM API key has been set successfully");
    return true;
}

/**
 * Loads a saved API key from localStorage
 * @returns {string|null} The saved API key or null if not found
 */
function loadSavedNimApiKey() {
    const savedKey = localStorage.getItem('nim_api_key');
    if (savedKey) {
        NIM_CONFIG.API_KEY = savedKey;
        return savedKey;
    }
    return null;
}

// Test NIM connection
async function testNimConnection() {
    try {
        const response = await fetch(`${NIM_CONFIG.ENDPOINTS.BASE_URL}${NIM_CONFIG.ENDPOINTS.CHAT_COMPLETIONS}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: "Hello, are you connected?"
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            console.error("NIM connection test failed:", errorData);
            return false;
        }

        console.log("NIM connection test successful");
        return true;
    } catch (error) {
        console.error("NIM connection test error:", error);
        return false;
    }
}

// Send the message to the backend
fetch('/api/chat', {  // Using /api/chat endpoint
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        messages: [{ role: 'user', content: messageText }]  // Using messages format
    })
})

// Response handling without RAG debug info
.then(data => {
    console.log("Response data:", data);

    // Hide typing indicator
    if (typingIndicator) {
        typingIndicator.style.display = 'none';
    }

    // Add the bot's response to the chat
    if (data.status === 'success' && data.message) {
        addMessageToChat(data.message, 'bot');
    } else {
        addMessageToChat("Error: " + (data.message || "Unknown error"), 'bot');
    }
})