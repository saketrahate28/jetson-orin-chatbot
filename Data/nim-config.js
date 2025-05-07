/**
 * NVIDIA Inference Microservices (NIM) Configuration
 *
 * This file contains settings for connecting to NVIDIA NIM services.
 * Replace the placeholders with your actual API key and preferred settings.
 */

const NIM_CONFIG = {
    // Your NVIDIA NIM API key - keep this secure and never expose it in client-side code
    API_KEY: "", // Will be populated from .env or user input

    // API endpoints
    ENDPOINTS: {
        BASE_URL: "https://api.nim.nvidia.com",
        CHAT_COMPLETIONS: "/v1/chat/completions",
        COMPLETIONS: "/v1/completions",
        EMBEDDINGS: "/v1/embeddings"
    },

    // Models available through NIM
    MODELS: {
        CHAT: "llama3-70b-instruct", // Default chat model
        TEXT: "llama3-70b-instruct",  // Default text completion model
        EMBEDDINGS: "embedding-001"   // Default embeddings model
    },

    // Default parameters for API calls
    DEFAULTS: {
        MAX_TOKENS: 2048,
        TEMPERATURE: 0.7,
        TOP_P: 0.95,
        FREQUENCY_PENALTY: 0,
        PRESENCE_PENALTY: 0
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
    NIM_CONFIG.API_KEY = "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";

    // Also save to localStorage for persistence (encrypt in a production app)
    localStorage.setItem('nim_api_key', apiKey);

    console.log("NIM API key has been set successfully");
    return true;
}

/**
 * Loads a previously saved API key if it exists
 * @returns {boolean} Whether a saved API key was loaded
 */
function loadSavedNimApiKey() {
    const savedKey = localStorage.getItem('nim_api_key');
    if (savedKey) {
        NIM_CONFIG.API_KEY = savedKey;
        return true;
    }
    return false;
}

/**
 * Validates if the current API key is set and properly formatted
 * @returns {boolean} Whether the API key is valid
 */
function isNimApiKeyValid() {
    return NIM_CONFIG.API_KEY &&
           typeof NIM_CONFIG.API_KEY === 'string' &&
           NIM_CONFIG.API_KEY.trim() !== '';
}

/**
 * Clears the stored API key
 */
function clearNimApiKey() {
    NIM_CONFIG.API_KEY = "";
    localStorage.removeItem('nim_api_key');
}

// Try to load a saved API key when the file is first loaded
loadSavedNimApiKey();