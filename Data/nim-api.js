/**
 * NVIDIA Inference Microservices (NIM) API Integration
 *
 * This file contains functions to interact with NVIDIA NIM APIs for AI capabilities.
 * The module depends on nim-config.js being loaded first.
 */

/**
 * Sends a chat completion request to NIM
 * @param {Array} messages - Array of message objects with role and content
 * @param {Object} options - Optional parameters to override defaults
 * @returns {Promise} - Promise resolving to the API response
 */
async function nimChatCompletion(messages, options = {}) {
    if (!isNimApiKeyValid()) {
        throw new Error("API key not set or invalid. Please configure your NIM API key first.");
    }

    const apiUrl = NIM_CONFIG.ENDPOINTS.BASE_URL + NIM_CONFIG.ENDPOINTS.CHAT_COMPLETIONS;

    const requestBody = {
        model: options.model || NIM_CONFIG.MODELS.CHAT,
        messages: messages,
        max_tokens: options.max_tokens || NIM_CONFIG.DEFAULTS.MAX_TOKENS,
        temperature: options.temperature || NIM_CONFIG.DEFAULTS.TEMPERATURE,
        top_p: options.top_p || NIM_CONFIG.DEFAULTS.TOP_P,
        frequency_penalty: options.frequency_penalty || NIM_CONFIG.DEFAULTS.FREQUENCY_PENALTY,
        presence_penalty: options.presence_penalty || NIM_CONFIG.DEFAULTS.PRESENCE_PENALTY,
        stream: options.stream || false
    };

    try {
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${NIM_CONFIG.API_KEY}`
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(`NIM API error (${response.status}): ${error.error?.message || 'Unknown error'}`);
        }

        return await response.json();
    } catch (error) {
        console.error("Error calling NIM chat API:", error);
        throw error;
    }
}

/**
 * Streams a chat completion response from NIM
 * @param {Array} messages - Array of message objects with role and content
 * @param {Function} onChunk - Callback function for each chunk of the response
 * @param {Object} options - Optional parameters to override defaults
 * @returns {Promise} - Promise that resolves when streaming is complete
 */
async function nimStreamChatCompletion(messages, onChunk, options = {}) {
    if (!isNimApiKeyValid()) {
        throw new Error("API key not set or invalid. Please configure your NIM API key first.");
    }

    const apiUrl = NIM_CONFIG.ENDPOINTS.BASE_URL + NIM_CONFIG.ENDPOINTS.CHAT_COMPLETIONS;

    const requestBody = {
        model: options.model || NIM_CONFIG.MODELS.CHAT,
        messages: messages,
        max_tokens: options.max_tokens || NIM_CONFIG.DEFAULTS.MAX_TOKENS,
        temperature: options.temperature || NIM_CONFIG.DEFAULTS.TEMPERATURE,
        top_p: options.top_p || NIM_CONFIG.DEFAULTS.TOP_P,
        frequency_penalty: options.frequency_penalty || NIM_CONFIG.DEFAULTS.FREQUENCY_PENALTY,
        presence_penalty: options.presence_penalty || NIM_CONFIG.DEFAULTS.PRESENCE_PENALTY,
        stream: true
    };

    try {
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${NIM_CONFIG.API_KEY}`
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(`NIM API error (${response.status}): ${error.error?.message || 'Unknown error'}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");

        let buffer = '';
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // Process complete lines in the buffer
            let lines = buffer.split('\n');
            buffer = lines.pop() || ''; // Keep the last incomplete line in the buffer

            for (const line of lines) {
                const trimmedLine = line.trim();
                if (!trimmedLine || trimmedLine === 'data: [DONE]') continue;

                try {
                    // Remove 'data: ' prefix if present
                    const jsonData = trimmedLine.startsWith('data: ')
                        ? JSON.parse(trimmedLine.slice(6))
                        : JSON.parse(trimmedLine);

                    if (jsonData.choices && jsonData.choices[0]?.delta?.content) {
                        onChunk(jsonData.choices[0].delta.content);
                    }
                } catch (e) {
                    console.warn('Error parsing streaming response chunk:', e);
                }
            }
        }
    } catch (error) {
        console.error("Error streaming from NIM chat API:", error);
        throw error;
    }
}

/**
 * Tests the NIM API connection with the current API key
 * @returns {Promise<boolean>} - Promise resolving to true if connection is successful
 */
async function testNimConnection() {
    if (!isNimApiKeyValid()) {
        return false;
    }

    try {
        // Simple test request with minimal tokens
        const response = await nimChatCompletion([
            { role: "system", content: "You are a helpful assistant." },
            { role: "user", content: "Hello, are you connected?" }
        ], { max_tokens: 5 });

        return !!response.choices;
    } catch (error) {
        console.error("NIM connection test failed:", error);
        return false;
    }
}

/**
 * Gets embeddings for a text from NIM
 * @param {string} text - The text to get embeddings for
 * @returns {Promise} - Promise resolving to the embedding vectors
 */
async function nimGetEmbeddings(text) {
    if (!isNimApiKeyValid()) {
        throw new Error("API key not set or invalid. Please configure your NIM API key first.");
    }

    const apiUrl = NIM_CONFIG.ENDPOINTS.BASE_URL + NIM_CONFIG.ENDPOINTS.EMBEDDINGS;

    try {
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${NIM_CONFIG.API_KEY}`
            },
            body: JSON.stringify({
                model: NIM_CONFIG.MODELS.EMBEDDINGS,
                input: text
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(`NIM API error (${response.status}): ${error.error?.message || 'Unknown error'}`);
        }

        const data = await response.json();
        return data.data[0].embedding;
    } catch (error) {
        console.error("Error getting embeddings from NIM:", error);
        throw error;
    }
}