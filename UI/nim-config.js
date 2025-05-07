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
        BASE_URL: "https://integrate.api.nvidia.com/v1",
        CHAT_COMPLETIONS: "/chat/completions",
        COMPLETIONS: "/completions",
        EMBEDDINGS: "/embeddings"
    },

    // Models available through NIM
    MODELS: {
        CHAT: "deepseek-ai/deepseek-r1", // Updated model
        TEXT: "deepseek-ai/deepseek-r1",
        EMBEDDINGS: "text-embedding-ada-002"
    },

    // Default parameters for API calls - Updated based on your Python code
    DEFAULTS: {
        MAX_TOKENS: 4096,    // Updated max_tokens
        TEMPERATURE: 0.6,    // Updated temperature
        TOP_P: 0.7,          // Updated top_p
        FREQUENCY_PENALTY: 0,
        PRESENCE_PENALTY: 0,
        STREAM: true  // Added streaming support
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
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${NIM_CONFIG.API_KEY}`
            },
            body: JSON.stringify({
                model: NIM_CONFIG.MODELS.CHAT,
                messages: [
                    { role: "system", content: "You are a helpful assistant." },
                    { role: "user", content: "Hello, are you connected?" }
                ],
                max_tokens: 5
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